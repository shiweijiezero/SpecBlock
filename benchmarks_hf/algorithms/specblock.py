"""SpecBlock Shift: SpecBlock with cross-slot hidden injection between decoder layers.

与 SpecBlock 完全相同的推理流程（tree construction, verification, accept），
draft model 使用带 shift_proj 层的 SpecBlockInferenceModel。
"""

import os
import torch

from ._specblock_base import _SpecBlockAlgorithmBase


class SpecBlockAlgorithm(_SpecBlockAlgorithmBase):
    """SpecBlock Shift: SpecBlock with shift_proj between decoder layers.

    Inherits all tree construction, verification, and generation logic from
    _SpecBlockAlgorithmBase. Only overrides load_model() to use the shift inference model.
    """

    # Day 22 Pareto config — 8-bench unified avg 104.96 tp, humaneval 133.27
    # 超 Eagle3 官方 130.72 (+1.9%). Env vars here default-on so CLI/UI/offline
    # eval gets the best config out-of-box without any export. Users can still
    # override via explicit `export VAR=...` before launching.
    _PARETO_DEFAULTS = {
        # Draft kernel / tree build
        'DRAFT_COMPILE':            '2',         # torch.compile draft (mode='default') → draft 11→7.4ms
        'TREE_ATTN_TRITON':         '1',         # triton attn kernel for draft
        'TREE_ATTN_SDPA':           '0',         # 关掉 SDPA (TRITON 赢在 short prompt)
        'TREE_BUILD_TRITON':        '1',         # triton mega tree build
        'TREE_FINALIZE_CUDA':       '1',         # GPU-resident finalize
        'BFS_EVENTS':               '0',         # skip per-block CUDA event (省 launch)
        'TARGET_ATTN_IMPL':         'sdpa',       # high-throughput target attention
        # Slot branching / topk
        'ADAPTIVE_SLOT0':           '1',         # p-based slot-0 branching
        'ADAPTIVE_ALL':             '1',         # p-based all-slot branching
        'SLOT_TOPK_MODE':           'slim_r2',   # [2,4,6,4] — slim rank-2 topk
        'STREAM_COMPACT_PROMPT':    '0',         # compact off (acc drops otherwise)
        # Day 21/22 unified controls
        'SPECBLOCK_INTERNAL_PREWARM':'1',        # 2-shape internal prewarm (default on, +30-45s load)
        'SPECBLOCK_DYNAMIC_TREE':   '0',         # prompt-length dynamic tree_tokens (default off, -0.8% avg when on)
    }

    def __init__(self, *args, **kwargs):
        # Apply Pareto defaults before super().__init__ reads env.
        # setdefault so explicit user export wins.
        for k, v in self._PARETO_DEFAULTS.items():
            os.environ.setdefault(k, v)
        super().__init__(*args, **kwargs)

    def load_model(self):
        """Load target model and SpecBlock Shift draft model."""
        # Load target model (reuse parent logic for everything except draft model import)
        from .specblock_inference_model import SpecBlockInferenceModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        self._assert_target_runtime_environment()
        print(f"Loading target model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        _attn_impl = os.environ.get('TARGET_ATTN_IMPL', 'sdpa')
        # Register custom attention backends under HF's ALL_ATTENTION_FUNCTIONS
        # so the model can pick them up via attn_implementation=<name>.
        if _attn_impl == 'flashinfer_tree':
            from .target_flashinfer_attn import register as _register_flashinfer
            _register_flashinfer(name='flashinfer_tree')
        elif _attn_impl == 'triton_tree':
            from .target_triton_attn import register as _register_triton
            _register_triton(name='triton_tree')
        _load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        if _attn_impl and _attn_impl != 'sdpa':
            _load_kwargs['attn_implementation'] = _attn_impl
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **_load_kwargs
        )
        self._assert_target_runtime_contract(self.target_model)
        self.target_model.eval()

        num_layers = self.target_model.config.num_hidden_layers
        offset = 1
        self.hidden_layer_indices = [
            1 + offset,
            num_layers // 2 - 1 + offset,
            num_layers - 4 + offset,
        ]
        print(f"  Hidden layer indices: {self.hidden_layer_indices}")

        print(f"Loading SpecBlock Shift draft model from {self.draft_model_path}...")
        config = AutoConfig.from_pretrained(self.draft_model_path, trust_remote_code=True)
        config.diffspec_draft_token_num = self.K

        self.draft_model = SpecBlockInferenceModel(config)

        safetensors_path = os.path.join(self.draft_model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
            self.draft_model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"No checkpoint at {safetensors_path}")

        # Copy embed_tokens from target
        with torch.no_grad():
            target_embed = self.target_model.model.embed_tokens.weight
            self.draft_model.input_layer.embed_tokens.weight.copy_(target_embed)
        print(f"  Copied embed_tokens ({target_embed.shape})")

        self.draft_model = self.draft_model.to(device=self.device, dtype=torch.bfloat16)
        self.draft_model.eval()

        self.draft_model.prepare_for_inference(quantize=getattr(self, 'draft_quantize', None))

        # torch.compile draft forward. Pareto: DRAFT_COMPILE=2 (Triton fusion,
        # mode='default'). Cuts draft from 11.1 → 7.4 ms on humaneval by fusing
        # ~100 per-op kernel launches into ~few fused kernels. Long-prompt bench
        # (gsm8k) sees less benefit (dynamic-shape recompiles dominate).
        _compile_mode = os.environ.get('DRAFT_COMPILE', '0')
        if _compile_mode in ('1', '2'):
            _mode_name = 'reduce-overhead' if _compile_mode == '1' else 'default'
            print(f"  [DRAFT_COMPILE={_compile_mode}] torch.compile(mode='{_mode_name}')")
            _compile_kwargs = dict(dynamic=True)
            if _compile_mode == '1':
                _compile_kwargs['mode'] = 'reduce-overhead'
            # Compiling the inference copy preserves parameter identity while
            # avoiding retracing across repeated decoding iterations.
            self.draft_model.forward_with_cache = torch.compile(
                self.draft_model.forward_with_cache, **_compile_kwargs,
            )
            self.draft_model.update_cache_and_draft = torch.compile(
                self.draft_model.update_cache_and_draft, **_compile_kwargs,
            )
            # Keep the request-batched ragged path eager: active B and padded N
            # vary enough to trigger expensive mid-run recompiles.  B=1 has a
            # stable batch dimension and retains the compiled fast path.
            self.draft_model.update_cache_and_draft_ragged_b1 = torch.compile(
                self.draft_model.update_cache_and_draft_ragged, **_compile_kwargs,
            )
            self.draft_model.update_cache_and_draft_ragged_from_condition_b1 = (
                torch.compile(
                    self.draft_model.update_cache_and_draft_ragged_from_condition,
                    **_compile_kwargs,
                )
            )

        # Vocab mapping
        vocab_mapping_path = os.path.join(self.draft_model_path, "vocab_mapping.pt")
        draft_vocab = getattr(config, 'draft_vocab_size', config.vocab_size)
        if draft_vocab != config.vocab_size and os.path.exists(vocab_mapping_path):
            self.draft_model.load_vocab_mapping(vocab_mapping_path)
            print(f"  Loaded vocab mapping: draft={draft_vocab}, target={config.vocab_size}")
        elif draft_vocab != config.vocab_size:
            print(f"  Using identity mapping (draft={draft_vocab}, target={config.vocab_size}, "
                  f"vocab_mapping.pt not found)")
        else:
            print(f"  No vocab mapping needed (vocab_size={config.vocab_size})")

        # Move rank lookup table to GPU
        self._rank_to_factor = self._rank_to_factor.to(self.device)

        # Pre-compute arange buffers reused across tree builds
        import numpy as _np
        self._arange_K_np = _np.arange(self.K, dtype=_np.int64)
        self._arange_K_t = torch.arange(self.K, device=self.device)

        self._init_draft_cuda_graph()

        # Static-shape, sync-free draft tree builder (opt-in via SPECBLOCK_STATIC=1).
        # Must be initialized BEFORE _internal_prewarm() because prewarm calls
        # _generate_once which dispatches via _use_static_draft.
        self._use_static_draft = (
            not self._strict_linear
            and os.environ.get("SPECBLOCK_STATIC", "0") == "1"
        )
        if self._use_static_draft:
            try:
                from .specblock_static_draft import StaticDraftBuilder
                self._static_draft = StaticDraftBuilder(
                    draft_model=self.draft_model,
                    K=self.K,
                    total_tokens=self.total_tokens,
                    max_blocks=self.max_blocks,
                    rank_classes=self.rank_classes,
                )
                if getattr(self, '_draft_graph_cache', None) is not None:
                    self._static_draft.attach_graph_cache(self._draft_graph_cache)
                print(
                    f"  [SPECBLOCK_STATIC=1] StaticDraftBuilder enabled: "
                    f"MAX_TOPK={self._static_draft.MAX_TOPK}, "
                    f"N_PEND={self._static_draft.N_PEND}, "
                    f"MAX_NODES={self._static_draft.MAX_NODES}, "
                    f"λ={self._static_draft.rank_bias_lambda}"
                )
            except Exception as _e:
                print(
                    f"  WARNING: failed to init StaticDraftBuilder: "
                    f"{type(_e).__name__}: {_e}"
                )
                self._use_static_draft = False
                self._static_draft = None

        # 2-shape internal prewarm (Day 22): run short + mid prompt dummy
        # generations to warm torch.compile cache for common bench prompt shapes.
        # Replaces the older `--benchmark-list hu:3 <target>` CLI prewarm hack
        # and the now-removed DRAFT_WARMUP opt-in modes. Disable via
        # SPECBLOCK_INTERNAL_PREWARM=0.
        if os.environ.get('SPECBLOCK_INTERNAL_PREWARM', '1') == '1':
            self._internal_prewarm()

        print(f"SpecBlock Shift ready! K={self.K}, max_blocks={self.max_blocks}, "
              f"total_tokens={self.total_tokens}")

    def _internal_prewarm(self):
        """Run dummy generations to warm draft + target compile caches.

        Aggressive prewarm: covers cross_count range 50..2000 across 4 prompt
        lengths × 200 generated tokens. Triton kernel autotune and dynamic
        torch.compile recompile both cache by tensor shape — without seeing
        each (prompt_len + generated_pos) bucket the first real sample of
        a benchmark pays autotune cost (50-200ms each), inflating draft
        latency on small samples.

        Empirical: short variant (2 prompts × 50 tokens, ~16s) leaves
        humaneval:10 sanity at draft=19.7ms vs n=80 truth 7.7ms (2.5×).

        Total cost: ~80-120s at load time. Opt out via SPECBLOCK_INTERNAL_PREWARM=0.
        SPECBLOCK_PREWARM_LITE=1 falls back to the old 2-prompt × 50-token form.
        """
        import time
        _t0 = time.perf_counter()
        if os.environ.get("SPECBLOCK_PREWARM_LITE", "0") == "1":
            dummy_msgs = [
                [{"role": "user", "content":
                  "Write a short Python function that takes a list of integers "
                  "and returns the sum of only the even numbers."}],
                [{"role": "user", "content":
                  "Alice has 24 apples, Bob has three times as many apples as "
                  "Alice, and Carol has half. They pool all apples and divide "
                  "among 8 friends. Each friend eats 2 apples per day. How "
                  "many full days will the apples last? Walk through step by step."}],
            ]
            max_new = 50
        else:
            # 5 prompts × 200 generated tokens: cover cross_count buckets
            #   1→200, 50→250, 200→400, 500→700, 1000→1200
            #   ~5 distinct shape ranges, hits Triton autotune cache.
            # ultra-short bucket added for UI / chat use where prompt may be
            # 1-5 tokens ("hi", "hello") — without it first generation pays
            # ~15s/iter compile cost (see Day 26 UI regression).
            dummy_msgs = [
                # ~1-token ultra-short (UI chat-style)
                [{"role": "user", "content": "hi"}],
                # ~50-token short (humaneval-style)
                [{"role": "user", "content":
                  "Write a Python function `is_palindrome(s)` that returns True "
                  "if s reads the same forwards and backwards, ignoring case."}],
                # ~200-token mid (gsm/mt-style CoT)
                [{"role": "user", "content":
                  "Alice has 24 apples, Bob has three times as many as Alice, "
                  "and Carol has half as many as Bob. They pool all their "
                  "apples and divide equally among 8 friends. Each friend eats "
                  "2 apples per day. How many full days will the pooled apples "
                  "last? Show every intermediate step including subtotals."}],
                # ~500-token long (alpaca/wmt-style multi-paragraph)
                [{"role": "user", "content":
                  "Translate the following English passage to French, then "
                  "summarize it in one paragraph in English. Passage: " +
                  ("The Roman Empire reached its greatest territorial extent "
                   "under Trajan in 117 AD, stretching from Britain to "
                   "Mesopotamia. Its sophisticated infrastructure of roads, "
                   "aqueducts, and legal codes shaped European civilization "
                   "for centuries. Latin evolved into the modern Romance "
                   "languages including Spanish, Italian, French, Portuguese, "
                   "and Romanian. Roman law forms the basis of civil-law "
                   "systems used today across continental Europe and Latin "
                   "America. Architectural innovations such as the arch and "
                   "the dome were carried forward through Byzantine and "
                   "Renaissance traditions. ") * 2}],
                # ~1000-token xlong (mtbench-style multi-turn / cnn-dm-style)
                [{"role": "user", "content":
                  "Read the technical document and answer the questions below. "
                  + ("Document section: Speculative decoding accelerates "
                     "autoregressive language model inference by drafting "
                     "multiple candidate tokens with a small drafter model "
                     "and verifying them in parallel with the larger target "
                     "model. The acceptance length tau measures how many "
                     "drafted tokens pass verification on average per "
                     "iteration. Tree-structured drafting expands a beam of "
                     "alternative continuations, allowing the verifier to "
                     "accept any matching path. EAGLE-3 grows the tree depth "
                     "by depth using sequential drafter calls. Parallel "
                     "drafters such as Medusa generate alternatives across "
                     "future positions in one call but lose path coherence. ") * 4 +
                  "Question 1: How does tau relate to per-iteration speedup? "
                  "Question 2: Why do parallel drafters trade tau for speed?"}],
            ]
            max_new = 200
        try:
            with torch.no_grad():
                for idx, m in enumerate(dummy_msgs):
                    _t_i = time.perf_counter()
                    _ = self._generate_once(m, max_new_tokens=max_new, temperature=0.0)
                    print(f"  [INTERNAL_PREWARM] pass {idx+1} done in "
                          f"{time.perf_counter() - _t_i:.1f}s")
            print(f"  [INTERNAL_PREWARM] total {time.perf_counter() - _t0:.1f}s")
        except Exception as _e:
            print(f"  [INTERNAL_PREWARM] skipped: {type(_e).__name__}: {_e}")
        finally:
            self.reset_after_warmup()

    # Day 22: prompt-length dynamic tree_tokens (opt-in, OFF by default).
    #
    # Hypothesis: short prompt → small tree (save draft/target overhead);
    # long prompt → large tree (amortize per-iter overhead).
    #
    # Empirical (threshold=150 @ 8-bench parallel): avg 104.14 vs static
    # tree=90 104.96 = -0.8%. Wins on humaneval (+1.6%) and mtbench_short_turns,
    # but mtbench long multi-turn samples, alpaca, nq_rag regress 3-5%.
    # Gsm8k (long prompt, high acc) and nq_rag (long prompt, low acc) can't be
    # distinguished by prompt length alone, which caps this heuristic.
    #
    # Kept as opt-in via SPECBLOCK_DYNAMIC_TREE=1. Env knobs:
    #   SPECBLOCK_DYNAMIC_TREE=0          (default off — static tree_tokens)
    #   SPECBLOCK_DYNAMIC_TREE_THRESHOLD=150
    #   SPECBLOCK_DYNAMIC_TREE_SMALL=50
    def _chat_template_kwargs(self):
        if os.environ.get("QWEN_NO_THINK", "0") == "1":
            return {"enable_thinking": False}
        return {}

    def _batch_tree_budget_for_prompt(self, prompt_len: int) -> int:
        budget = int(self.total_tokens)
        if os.environ.get('SPECBLOCK_DYNAMIC_TREE', '0') != '1':
            return budget
        if getattr(self, '_use_static_draft', False):
            raise NotImplementedError(
                "SPECBLOCK_DYNAMIC_TREE is incompatible with the fixed-budget "
                "StaticDraftBuilder in true-batch mode"
            )
        threshold = int(os.environ.get('SPECBLOCK_DYNAMIC_TREE_THRESHOLD', '150'))
        small = int(os.environ.get('SPECBLOCK_DYNAMIC_TREE_SMALL', '50'))
        return small if prompt_len <= threshold and small < budget else budget

    def _generate_once(self, conversation, max_new_tokens, temperature, **kwargs):
        if os.environ.get('SPECBLOCK_DYNAMIC_TREE', '0') != '1':
            return super()._generate_once(
                conversation, max_new_tokens, temperature, **kwargs,
            )

        _ct_kwargs = {"enable_thinking": False} if os.environ.get("QWEN_NO_THINK", "0") == "1" else {}
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **_ct_kwargs
        )
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
        threshold = int(os.environ.get('SPECBLOCK_DYNAMIC_TREE_THRESHOLD', '150'))
        small = int(os.environ.get('SPECBLOCK_DYNAMIC_TREE_SMALL', '50'))

        _orig_tt = self.total_tokens
        if prompt_len <= threshold and small < _orig_tt:
            self.total_tokens = small
        try:
            return super()._generate_once(
                conversation, max_new_tokens, temperature, **kwargs,
            )
        finally:
            self.total_tokens = _orig_tt
