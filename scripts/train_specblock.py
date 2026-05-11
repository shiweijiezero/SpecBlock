import argparse
import hashlib
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge import (
    AutoDraftModel,
    AutoDraftModelConfig,
)
from specforge.core.specblock import OnlineSpecBlockModel
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_draft_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.target import (
    Eagle3TargetModel,
    get_eagle3_target_model,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import Tracker, create_tracker, get_tracker_class
from specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_args_with_dots,
    print_on_rank0,
    print_with_rank,
)


def parse_args() -> Tuple[ArgumentParser, Namespace]:
    parser = argparse.ArgumentParser(description="Train SpecBlock with online data")

    # model arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument("--draft-model-config", type=str, required=False)
    model_group.add_argument("--embedding-key", type=str, default="model.embed_tokens.weight")
    model_group.add_argument(
        "--target-model-backend", type=str, default="sglang",
        choices=["sglang", "hf", "custom"],
    )

    # dataset arguments
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="llama3")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--build-dataset-num-proc", type=int, default=8)
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=4)

    # training hyper params
    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=10)
    training_group.add_argument("--max-num-steps", type=int, default=None)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=1e-4)
    training_group.add_argument("--max-length", type=int, default=1024)
    training_group.add_argument("--warmup-ratio", type=float, default=0.015)
    training_group.add_argument("--warmup-steps", type=int, default=None)
    training_group.add_argument(
        "--scheduler-type", type=str, default="cosine",
        choices=["linear", "cosine"],
    )
    training_group.add_argument("--total-steps", type=int, default=None)
    training_group.add_argument("--max-grad-norm", type=float, default=0.5)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument("--ckpt-dir", type=str, default=None)
    training_group.add_argument("--eval-interval", type=int, default=5000)
    training_group.add_argument("--save-interval", type=int, default=5000)
    training_group.add_argument("--log-interval", type=int, default=50)
    training_group.add_argument("--seed", type=int, default=0)
    training_group.add_argument("--draft-accumulation-steps", type=int, default=1)

    # distributed / optimization
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)
    optimization_group.add_argument("--sp-ulysses-size", type=int, default=1)
    optimization_group.add_argument("--sp-ring-size", type=int, default=1)
    optimization_group.add_argument("--attention-backend", type=str, default="flex_attention")

    # other args
    other_group = parser.add_argument_group("others")
    other_group.add_argument("--cache-key", type=str, default=None)
    other_group.add_argument("--cache-dir", type=str, default="./cache")
    other_group.add_argument("--output-dir", type=str, required=True)
    other_group.add_argument("--vocab-mapping-override", type=str, default=None,
                             help="Force load vocab_mapping.pt from this path (base ckpt) "
                             "instead of regenerating from train data. Critical for fine-tune.")
    other_group.add_argument("--verbose", action="store_true")
    other_group.add_argument("--dist-timeout", type=int, default=1000)
    other_group.add_argument("--model-download-dir", type=str, default=None)

    # profiling
    profiling_group = parser.add_argument_group("profiling")
    profiling_group.add_argument("--profile", action="store_true")
    profiling_group.add_argument("--profile-start-step", type=int, default=30)
    profiling_group.add_argument("--profile-num-steps", type=int, default=4)
    profiling_group.add_argument("--profile-record-shapes", action="store_true")

    # sglang backend
    sglang_group = parser.add_argument_group("sglang target model backend")
    SGLangBackendArgs.add_args(sglang_group)

    # SpecBlock specific args
    specblock_group = parser.add_argument_group("specblock")
    specblock_group.add_argument(
        "--position-loss-weight", type=float, default=1.0,
        help="Position loss weight decay factor. weight = decay^k for position k.",
    )
    specblock_group.add_argument(
        "--draft-token-num", type=int, default=None,
        help="Number of draft tokens (K). If not provided, uses config value.",
    )
    specblock_group.add_argument(
        "--unlock-warmup-steps", type=int, default=None,
        help="Warmup steps for progressive position unlocking.",
    )
    specblock_group.add_argument(
        "--rank-start-step", type=int, default=2000,
        help="Step to start training rank head. Before this step, rank loss weight = 0.",
    )
    specblock_group.add_argument(
        "--num-ttt-blocks", type=int, default=None,
        help="Number of TTT blocks for block-level autoregression. If not provided, uses config value.",
    )
    specblock_group.add_argument(
        "--num-layers", type=int, default=None,
        help="Override num_hidden_layers in draft model config. If not provided, uses config value.",
    )
    specblock_group.add_argument(
        "--rank-budget-per-class", type=int, default=10,
        help="Target effective count per class for rank loss frequency reweighting (default: 10). "
             "Within each slot, weight(c) = min(1, budget / count_c). <=0 disables.",
    )
    specblock_group.add_argument(
        "--draft-loss-topk", type=int, default=-1,
        help="HASS-style top-K restriction of draft KL loss. If >0, only the top-K "
             "target-probability tokens per position contribute to the loss; long-tail "
             "tokens are ignored. -1 = full vocab KL (default, backward compatible).",
    )
    # tracker
    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    args = parser.parse_args()
    return parser, args


def build_tracker(args: Namespace, parser: ArgumentParser) -> Tracker:
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")
    tracker = create_tracker(args, args.output_dir)
    return tracker


def build_target_model(
    args: Namespace, draft_model_config: AutoDraftModelConfig
) -> Eagle3TargetModel:
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    else:
        target_model_kwargs = {}
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda",
        cache_dir=args.model_download_dir,
        **target_model_kwargs,
    )

    if (
        hasattr(draft_model_config, "eagle_config")
        and draft_model_config.eagle_config is not None
        and "eagle_aux_hidden_state_layer_ids" in draft_model_config.eagle_config
    ):
        target_model.set_aux_hidden_states_layers(
            draft_model_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
        )
    else:
        target_model.set_aux_hidden_states_layers()

    return target_model


def sanity_check(args: Namespace) -> None:
    args.dp_size = dist.get_world_size() // args.tp_size
    args.target_batch_size = args.tp_size * args.batch_size
    args.draft_accumulation_steps = (
        args.draft_accumulation_steps * args.sp_ulysses_size * args.sp_ring_size
    )


def build_draft_model(
    args: Namespace,
) -> Tuple[AutoDraftModelConfig, nn.Module, Optional[str]]:
    if args.draft_model_config is None:
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, cache_dir=args.model_download_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # Override config with command-line args
    if args.num_layers is not None:
        draft_model_config.num_hidden_layers = args.num_layers
        print_on_rank0(f"Overriding num_hidden_layers = {args.num_layers}")
    if args.draft_token_num is not None:
        draft_model_config.diffspec_draft_token_num = args.draft_token_num
        print_on_rank0(f"Overriding diffspec_draft_token_num = {args.draft_token_num}")

    draft_model_last_checkpoint = None
    resume_checkpoint_path = None

    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            # FIX: previously `draft_model_config = os.path.join(..., "config.json")`
            # clobbered the AutoDraftModelConfig object to a bare string, breaking
            # downstream attribute access in build_target_model (needs .eagle_config)
            # and train loop (needs .diffspec_draft_token_num / .num_ttt_blocks).
            # Rebuild config from ckpt's config.json, reapply CLI overrides.
            draft_model_config = AutoDraftModelConfig.from_file(
                os.path.join(args.ckpt_dir, "config.json")
            )
            if args.num_layers is not None:
                draft_model_config.num_hidden_layers = args.num_layers
            if args.draft_token_num is not None:
                draft_model_config.diffspec_draft_token_num = args.draft_token_num
            draft_model_last_checkpoint = args.ckpt_dir
            print_on_rank0(f"Finetuning from base model: {draft_model_last_checkpoint}")
        else:
            raise ValueError(
                f"Provided base model dir {args.ckpt_dir} is not a valid directory."
            )

    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")
        resume_checkpoint_path = draft_model_last_checkpoint

    if draft_model_last_checkpoint:
        draft_model = AutoDraftModel.from_pretrained(
            draft_model_last_checkpoint,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        draft_model = AutoDraftModel.from_config(
            draft_model_config,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()

    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    return draft_model_config, draft_model, resume_checkpoint_path


def build_dataloaders(
    args: Namespace,
    draft_model_config: AutoDraftModelConfig,
) -> Tuple[DataLoader, str, Optional[DataLoader]]:
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    train_cache_dir = os.path.join(args.cache_dir, "processed_dataset")
    vocab_cache_dir = os.path.join(args.cache_dir, "vocab_mapping")
    vocab_mapping_path = os.path.join(vocab_cache_dir, f"{cache_key}.pt")

    rank = dist.get_rank()
    if rank == 0:
        # Rank 0: load raw data, process, and write caches
        train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=train_cache_dir,
            cache_key=cache_key,
            is_vlm=False,
            is_preformatted=args.is_preformatted,
            processor=None,
            num_proc=args.build_dataset_num_proc,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=vocab_cache_dir,
            cache_key=cache_key,
        )
    dist.barrier()
    if rank != 0:
        # Other ranks: use per-rank cache dir to avoid FileLock conflicts on shared fs
        local_cache = os.path.join(args.cache_dir, f"hf_cache_rank{rank}")
        train_dataset = load_dataset("json", data_files=args.train_data_path, cache_dir=local_cache)["train"]
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=train_cache_dir,
            cache_key=cache_key,
            is_vlm=False,
            is_preformatted=args.is_preformatted,
            processor=None,
            num_proc=args.build_dataset_num_proc,
        )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.target_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=(
            get_draft_dp_group() if args.attention_backend == "usp" else get_dp_group()
        ),
        is_vlm=False,
    )

    eval_cache_key = hashlib.md5(
        f"{args.eval_data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}".encode()
    ).hexdigest() if args.eval_data_path else None

    if args.eval_data_path is not None:
        if rank == 0:
            eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
            eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset,
                tokenizer,
                args.chat_template,
                args.max_length,
                cache_dir=train_cache_dir,
                cache_key=eval_cache_key,
                is_vlm=False,
                processor=None,
                num_proc=args.build_dataset_num_proc,
                is_preformatted=args.is_preformatted,
            )
        dist.barrier()
        if rank != 0:
            eval_dataset = load_dataset("json", data_files=args.eval_data_path, cache_dir=local_cache)["train"]
            eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset,
                tokenizer,
                args.chat_template,
                args.max_length,
                cache_dir=train_cache_dir,
                cache_key=eval_cache_key,
                is_vlm=False,
                processor=None,
                num_proc=args.build_dataset_num_proc,
                is_preformatted=args.is_preformatted,
            )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.target_batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=(
                get_draft_dp_group()
                if args.attention_backend == "usp"
                else get_dp_group()
            ),
            is_vlm=False,
        )
        print_with_rank("Initialized eval dataloader")
    else:
        eval_dataloader = None
    return (
        train_dataloader,
        vocab_mapping_path,
        eval_dataloader,
    )


def save_checkpoints(
    args: Namespace,
    epoch: int,
    step: int,
    specblock_model: nn.Module,
    optimizer: Optimizer,
    vocab_mapping_path: str = None,
):
    epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(epoch_output_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(specblock_model, StateDictType.FULL_STATE_DICT):
        model_state_dict = specblock_model.state_dict()
        state_to_save = {
            "epoch": epoch,
            "global_step": step,
            "args": args,
        }
        state_to_save.update(optimizer.state_dict())

        draft_model_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in model_state_dict.items()
            if "draft_model." in k and "embed" not in k.lower()
        }

        if dist.get_rank() == 0:
            torch.save(
                state_to_save,
                os.path.join(epoch_output_dir, "training_state.pt"),
            )
            print_on_rank0(
                f"Saved full training state to {epoch_output_dir}/training_state.pt"
            )
            specblock_model.draft_model.save_pretrained(
                epoch_output_dir,
                state_dict=draft_model_state_dict,
            )
            if vocab_mapping_path and os.path.exists(vocab_mapping_path):
                import shutil
                shutil.copy2(vocab_mapping_path, os.path.join(epoch_output_dir, "vocab_mapping.pt"))
                print_on_rank0(f"Saved vocab_mapping.pt to {epoch_output_dir}")
            print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
        dist.barrier()


def run_forward(
    args: Namespace,
    specblock_model: nn.Module,
    data: dict,
    target_model: Eagle3TargetModel,
    draft_token_num: int = 7,
    global_step: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Run forward pass for online SpecBlock training.

    Returns:
        plosses: [draft_loss_0, ..., draft_loss_{K-1}, rank_loss]
        acces: [draft_acc_0, ..., draft_acc_{K-1}, rank_acc]
        input_ids: Input token IDs
    """
    eagle3_data = target_model.generate_eagle3_data(
        input_ids=data["input_ids"].cuda(),
        attention_mask=data["attention_mask"].cuda(),
        loss_mask=data["loss_mask"].cuda(),
    )

    input_ids = get_dp_data_shard_from_tp(eagle3_data.input_ids)
    attention_mask = get_dp_data_shard_from_tp(eagle3_data.attention_mask)
    loss_mask = get_dp_data_shard_from_tp(eagle3_data.loss_mask)
    target = get_dp_data_shard_from_tp(eagle3_data.target)
    hidden_states = get_dp_data_shard_from_tp(eagle3_data.hidden_states)

    plosses, _, acces = specblock_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target=target,
        loss_mask=loss_mask,
        hidden_states=hidden_states,
        global_step=global_step,
    )
    return plosses, acces, input_ids


def run_backward_and_update(
    args: Namespace, plosses: List[torch.Tensor], optimizer: Optimizer, global_step: int
) -> None:
    """
    Compute weighted loss and run backward pass.

    plosses = [draft_loss_0, ..., draft_loss_{K-1}, rank_loss]
    """
    # Separate draft losses and rank loss
    rank_loss = plosses[-1]
    draft_plosses = plosses[:-1]
    K = len(draft_plosses)
    device = draft_plosses[0].device

    # Draft loss: weighted by position
    k_indices = torch.arange(K, device=device, dtype=torch.float32)
    base_weights = args.position_loss_weight ** k_indices

    if args.unlock_warmup_steps is not None:
        alpha_k = (global_step / (args.unlock_warmup_steps * (k_indices + 1))).clamp(max=1.0)
    else:
        alpha_k = 1.0

    ploss_weights = alpha_k * base_weights
    draft_loss = (ploss_weights * torch.stack(draft_plosses)).sum()

    # Rank loss: step-gated
    rank_weight = 1.0 if global_step >= args.rank_start_step else 0.0
    total_loss = (draft_loss + rank_weight * rank_loss) / args.draft_accumulation_steps

    total_loss.backward()

    if global_step % args.draft_accumulation_steps == 0:
        optimizer.step()


def record_metrics(
    args: Namespace,
    accuracies: List[torch.Tensor],
    plosses: List[torch.Tensor],
    global_step: int,
    tracker: Tracker,
    optimizer: Optional[Optimizer] = None,
    mode: str = "train",
) -> None:
    """Record training/eval metrics to tracker."""
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    # Separate draft metrics and rank metrics
    rank_loss = plosses[-1]
    rank_acc_dict = accuracies[-1]  # dict: {"rank0": tensor, "rank1": tensor, ...}
    draft_plosses = plosses[:-1]
    draft_accuracies = accuracies[:-1]

    # Normalize logged loss by blocks_trained (loss is accumulated across blocks,
    # but logged value should be per-block average for smooth wandb curves)
    blocks_trained = rank_acc_dict.get("ttt_blocks_trained", torch.tensor(1.0)).item()
    blocks_trained = max(blocks_trained, 1.0)

    # Draft metrics
    draft_accs_tensor = torch.stack(draft_accuracies)
    draft_losses_tensor = torch.stack(draft_plosses)

    dist.all_reduce(draft_accs_tensor, op=dist.ReduceOp.AVG)
    draft_accs_list = draft_accs_tensor.cpu().tolist()
    for i in range(len(draft_accs_list)):
        logdict[f"{mode}/acc_{i}"] = draft_accs_list[i]
        print_on_rank0(
            f"{mode.capitalize()} - Step {global_step}, position {i}, Acc: {draft_accs_list[i]:.2f}"
        )

    dist.all_reduce(draft_losses_tensor, op=dist.ReduceOp.AVG)
    draft_losses_list = draft_losses_tensor.cpu().tolist()
    for i in range(len(draft_losses_list)):
        logdict[f"{mode}/ploss_{i}"] = draft_losses_list[i] / blocks_trained
        print_on_rank0(
            f"{mode.capitalize()} - Step {global_step}, position {i}, pLoss: {draft_losses_list[i] / blocks_trained:.4f}"
        )

    # Rank metrics (also normalize by blocks_trained)
    rank_loss_tensor = rank_loss.detach().clone()
    dist.all_reduce(rank_loss_tensor, op=dist.ReduceOp.AVG)
    logdict[f"{mode}/rank_loss"] = rank_loss_tensor.item() / blocks_trained

    # Per-class rank accuracy, counts, and TTT metrics
    rank_acc_strs = []
    rank_count_strs = []
    ttt_strs = []
    for key, val in rank_acc_dict.items():
        val_tensor = val.detach().clone()
        if key.startswith("rank_count"):
            # Counts: sum across GPUs (not average)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            v = val_tensor.cpu().item()
            logdict[f"{mode}/rank_count/{key}"] = v
            rank_count_strs.append(f"{key}={v:.0f}")
        elif key.startswith("ttt_M_"):
            # M distribution counts: sum across GPUs
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            v = val_tensor.cpu().item()
            logdict[f"{mode}/ttt/{key}"] = v
            if v > 0:
                ttt_strs.append(f"{key}={v:.0f}")
        elif key.startswith("ttt_"):
            # TTT metrics: average across GPUs
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            v = val_tensor.cpu().item()
            logdict[f"{mode}/ttt/{key}"] = v
            ttt_strs.append(f"{key}={v:.2f}")
        elif key.startswith("block") and "_acc" in key:
            # Per-block per-position accuracy: block0_acc1, block1_acc1, etc.
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            v = val_tensor.cpu().item()
            logdict[f"{mode}/block_acc/{key}"] = v
        elif key.startswith("posacc_"):
            # Absolute position accuracy: posacc_1, posacc_5, etc.
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            v = val_tensor.cpu().item()
            logdict[f"{mode}/posacc/{key}"] = v
        else:
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            v = val_tensor.cpu().item()
            logdict[f"{mode}/rank_acc/{key}"] = v
            rank_acc_strs.append(f"{key}={v:.2f}")

    # Per-block accuracy summary
    block_acc_strs = []
    for bidx in range(4):  # max 4 blocks
        block_keys = sorted([k for k in rank_acc_dict if k.startswith(f"block{bidx}_acc")])
        if not block_keys:
            break
        vals = [logdict.get(f"{mode}/block_acc/{k}", 0.0) for k in block_keys]
        block_acc_strs.append(f"B{bidx}=[{', '.join(f'{v:.2f}' for v in vals)}]")

    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}, Rank Loss: {rank_loss_tensor.cpu().item() / blocks_trained:.4f}, "
        f"Rank Acc: [{', '.join(rank_acc_strs)}]"
    )
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}, Rank Count: [{', '.join(rank_count_strs)}]"
    )
    if block_acc_strs:
        print_on_rank0(
            f"{mode.capitalize()} - Step {global_step}, Block Acc: {', '.join(block_acc_strs)}"
        )
    # Absolute position accuracy summary
    posacc_keys = sorted(
        [k for k in rank_acc_dict if k.startswith("posacc_")],
        key=lambda k: int(k.split("_")[1]),
    )
    if posacc_keys:
        posacc_strs = [f"p{k.split('_')[1]}={logdict.get(f'{mode}/posacc/{k}', 0.0):.2f}" for k in posacc_keys]
        print_on_rank0(
            f"{mode.capitalize()} - Step {global_step}, PosAcc: [{', '.join(posacc_strs)}]"
        )
    if ttt_strs:
        print_on_rank0(
            f"{mode.capitalize()} - Step {global_step}, TTT: [{', '.join(ttt_strs)}]"
        )

    tracker.log(logdict, step=global_step)


def get_dp_data_shard_from_tp(tensor: torch.Tensor) -> torch.Tensor:
    tp_size = dist.get_world_size(get_tp_group())
    tp_rank = dist.get_rank(get_tp_group())
    return tensor.chunk(tp_size, dim=0)[tp_rank]


def main():
    # ================================================
    # 1. Initialize
    # ================================================
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=args.sp_ring_size,
        sp_ulysses_size=args.sp_ulysses_size,
    )

    sanity_check(args)
    print_args_with_dots(args)
    print_with_rank("Initialized distributed environment")

    # ================================================
    # 2. Build models
    # ================================================
    draft_model_config, draft_model, resume_checkpoint_path = build_draft_model(args)
    target_model = build_target_model(args, draft_model_config)

    # ================================================
    # 3. Build dataloader
    # ================================================
    train_dataloader, vocab_mapping_path, eval_dataloader = build_dataloaders(
        args, draft_model_config
    )

    # For fine-tune: use the base checkpoint's vocab_mapping.pt rather than
    # regenerating from training data. Keeping d2t/t2d consistent with the
    # weights trained on the base vocab avoids drift at inference time.
    # Set via --vocab-mapping-override <path>.
    override = getattr(args, "vocab_mapping_override", None)
    if override:
        draft_model.load_vocab_mapping(override)
        print_with_rank(f"OVERRIDE vocab mapping loaded from {override}")
        vocab_mapping_path = override  # also propagate for save_checkpoints
    else:
        draft_model.load_vocab_mapping(vocab_mapping_path)
        print_with_rank("Loaded vocab mapping (freshly generated)")

    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
        print_with_rank(
            f"Auto-calculated total_steps: {args.total_steps} (num_epochs={args.num_epochs} * steps_per_epoch={steps_per_epoch})"
        )
    else:
        print_with_rank(f"Using provided total_steps: {args.total_steps}")

    # ================================================
    # 4. Build SpecBlock model
    # ================================================
    draft_token_num = args.draft_token_num or getattr(
        draft_model_config, "diffspec_draft_token_num", 7
    )
    num_ttt_blocks = args.num_ttt_blocks or getattr(draft_model_config, "num_ttt_blocks", 4)
    # Save num_ttt_blocks to config so inference can read it
    draft_model_config.num_ttt_blocks = num_ttt_blocks
    specblock_model = OnlineSpecBlockModel(
        draft_model=draft_model,
        length=draft_token_num,
        rank_start_step=args.rank_start_step,
        num_ttt_blocks=num_ttt_blocks,
        budget_per_class=args.rank_budget_per_class,
        draft_loss_topk=args.draft_loss_topk,
    )

    specblock_model = FSDP(
        specblock_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        process_group=dist.group.WORLD,
    )
    print_with_rank("Initialized SpecBlock FSDP model")

    # ================================================
    # 6. Build optimizer and scheduler
    # ================================================
    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        scheduler_type=args.scheduler_type,
    )
    print_with_rank("Initialized optimizer and scheduler")

    # ================================================
    # 7. Build tracker and resume training state
    # ================================================
    tracker = build_tracker(args, parser)
    global_step = 0
    start_epoch = 0

    if resume_checkpoint_path is not None:
        training_state_path = os.path.join(resume_checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            print_on_rank0(f"Loading training state from {training_state_path}")
            training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
            global_step = training_state.get("global_step", 0)
            start_epoch = training_state.get("epoch", 0)
            optimizer.load_state_dict(training_state)
            print_on_rank0(
                f"Resumed from epoch {start_epoch}, global_step {global_step}"
            )
        else:
            print_on_rank0(
                f"Warning: training_state.pt not found in {resume_checkpoint_path}, "
                "starting from scratch with loaded model weights"
            )

    dist.barrier()

    last_time = time.time()

    # ================================================
    # 8. Start training
    # ================================================
    print_on_rank0(f"Starting training from epoch {start_epoch}, global_step {global_step}")
    print_on_rank0(f"Rank head starts training at step {args.rank_start_step}")
    print_on_rank0(f"TTT blocks: {num_ttt_blocks} (Phase 1: single block, Phase 2: multi-block TTT, M ~ Uniform(1,K), no inter-block filtering)")
    print_on_rank0(f"Rank freq reweight: budget_per_class={args.rank_budget_per_class} (w=min(1, budget/count))")
    if args.draft_loss_topk > 0:
        print_on_rank0(f"Draft loss top-K restriction: draft_loss_topk={args.draft_loss_topk} (HASS-style, long-tail ignored)")
    else:
        print_on_rank0(f"Draft loss: full-vocab KL (draft_loss_topk={args.draft_loss_topk})")

    resume_global_step = global_step
    steps_per_epoch = len(train_dataloader)
    current_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for data in progress_bar:
            current_step += 1

            if current_step <= resume_global_step:
                continue

            global_step += 1

            # ================================================
            # 8.0 Profiling
            # ================================================
            if args.profile:
                if global_step == args.profile_start_step + 1:
                    print("Start profile")
                    torch_profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        record_shapes=args.profile_record_shapes,
                    )
                    torch_profiler.start()
                if global_step == args.profile_start_step + args.profile_num_steps + 1:
                    output_path = os.path.join(
                        args.output_dir,
                        f"profile_rank{torch.distributed.get_rank()}_{time.time()}.trace.json.gz",
                    )
                    print(f"End profile {output_path=}")
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(output_path)

            # ================================================
            # 8.1 Training Step
            # ================================================
            plosses, acces, batch_input_ids = run_forward(
                args, specblock_model, data, target_model,
                draft_token_num=draft_token_num,
                global_step=global_step,
            )
            run_backward_and_update(args, plosses, optimizer, global_step)

            # log training metrics
            if global_step % (args.log_interval * args.draft_accumulation_steps) == 0:
                record_metrics(
                    args,
                    acces,
                    plosses,
                    global_step // args.draft_accumulation_steps,
                    tracker,
                    optimizer,
                    mode="train",
                )

            if dist.get_rank() == 0:
                time_per_step = time.time() - last_time
                last_time = time.time()
                # Draft losses only (exclude rank loss)
                draft_plosses = plosses[:-1]
                draft_acces = acces[:-1]
                n_blocks = acces[-1].get("ttt_blocks_trained", torch.tensor(1.0)).item()
                n_blocks = max(n_blocks, 1.0)
                avg_loss = sum(pl for pl in draft_plosses) / len(draft_plosses) / n_blocks
                avg_acc = sum(draft_acces) / len(draft_acces)
                rank_loss_val = plosses[-1] / n_blocks
                rank_acc_dict = acces[-1]  # dict of per-class rank acc
                # Show metrics in progress bar (use .item() only on rank 0, outside training critical path)
                _zero = torch.tensor(0.0)
                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.2f}",
                        "acc": f"{avg_acc:.2f}",
                        "rloss": f"{rank_loss_val:.2f}",
                        "racc0": f"{rank_acc_dict.get('rank0', _zero):.2f}",
                        "blk": f"{rank_acc_dict.get('ttt_blocks_trained', _zero):.0f}",
                        "M": f"{rank_acc_dict.get('ttt_avg_M', _zero):.1f}",
                        "Mv": f"{rank_acc_dict.get('ttt_m_valid_ratio', _zero):.2f}",
                        "time": f"{time_per_step:.2f}s",
                    }
                )

            # ================================================
            # 8.2 Evaluation Step
            # ================================================
            if (
                args.eval_data_path is not None
                and global_step % args.eval_interval == 0
            ):
                draft_model.eval()
                num_metrics = draft_token_num + 1  # K draft + 1 rank
                eval_draft_acces = [[] for _ in range(draft_token_num)]
                eval_rank_acces = []  # list of dicts
                eval_plosses = [[] for _ in range(num_metrics)]

                for eval_data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                    with torch.no_grad():
                        eval_plosses_batch, eval_acces_batch, _ = run_forward(
                            args, specblock_model, eval_data, target_model,
                            draft_token_num=draft_token_num,
                            global_step=global_step,
                        )
                        # Draft accuracies (tensors)
                        for i in range(draft_token_num):
                            eval_draft_acces[i].append(eval_acces_batch[i])
                        # Rank accuracy (dict)
                        eval_rank_acces.append(eval_acces_batch[-1])
                        eval_plosses = [
                            eval_plosses[i] + [eval_plosses_batch[i]] for i in range(len(eval_plosses_batch))
                        ]

                # Average draft accuracies
                eval_acces = [torch.stack(acc).mean() for acc in eval_draft_acces]
                # Aggregate rank metrics per class across batches
                # Use union of all keys (posacc_/block_acc keys may vary across batches due to M sampling)
                all_keys = set()
                for d in eval_rank_acces:
                    all_keys.update(d.keys())
                avg_rank_acc_dict = {}
                _dev = next(iter(eval_rank_acces[0].values())).device
                zero = torch.tensor(0.0, device=_dev)
                for key in all_keys:
                    vals = torch.stack([d.get(key, zero) for d in eval_rank_acces])
                    # Counts: sum across batches; accuracies: average
                    avg_rank_acc_dict[key] = vals.sum() if key.startswith("rank_count") else vals.mean()
                eval_acces.append(avg_rank_acc_dict)
                eval_plosses = [torch.stack(pl).mean() for pl in eval_plosses]

                record_metrics(
                    args,
                    eval_acces,
                    eval_plosses,
                    global_step // args.draft_accumulation_steps,
                    tracker,
                    mode="eval",
                )

                draft_model.train()

            # ================================================
            # 8.3 Save Checkpoints
            # ================================================
            if global_step % args.save_interval == 0:
                save_checkpoints(args, epoch, global_step, specblock_model, optimizer, vocab_mapping_path)

            if args.max_num_steps is not None and global_step >= args.max_num_steps:
                break

        if args.max_num_steps is not None and global_step >= args.max_num_steps:
            break

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
