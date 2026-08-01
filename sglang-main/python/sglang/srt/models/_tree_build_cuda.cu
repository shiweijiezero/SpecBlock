// SpecBlock-specific tree-build CUDA kernels.
//
// Replaces tree_build_triton.py with two native kernels, callable via
// torch.utils.cpp_extension.load. The kernels match the triton version
// byte-identical on all outputs; the only delta is dispatch/launch
// overhead (expected savings ~100-400 µs per tree build).
//
// Design notes
//  - Block-1 kernel: <<<1, 1>>>. K=4 is tiny; serial thread keeps the
//    code straight-line and avoids synchronization between lanes. The
//    win is eliminating triton JIT + Python wrapper overhead, not
//    SIMT parallelism.
//  - BFS kernel: <<<N, 1>>>. One block per pending leaf. Each block
//    *recomputes* the cross-leaf prefix-sum (O(N) reads, trivial at
//    N ≤ 15) so we collapse triton's sizing+scatter into one launch.
//  - Sizes that Python needs are written into a small GPU int64 tensor
//    at kernel completion; caller copies to pinned host memory.
//
// Outputs byte-match the eager / triton paths (same adaptive rules,
// same hitchhike flag, same alt layout [K + cum_alt_excl[k] + j-1]).

#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


// -----------------------------------------------------------------------
//   Adaptive rules — kept as __device__ inline to mirror triton literals.
// -----------------------------------------------------------------------

__device__ __forceinline__ int64_t apply_adaptive_slot0(
    int mode, float p, int64_t beam_width)
{
    const int64_t bw = beam_width;
    if (mode == 0) return bw;
    if (mode == 1) {
        if (p >= 0.9f)  return min(bw, (int64_t)3);
        if (p >= 0.8f)  return min(bw, (int64_t)5);
        if (p >= 0.6f)  return min(bw, (int64_t)8);
        return bw;
    }
    if (mode == 2) {
        if (p >= 0.95f) return min(bw, (int64_t)2);
        if (p >= 0.85f) return min(bw, (int64_t)4);
        if (p >= 0.6f)  return min(bw, (int64_t)7);
        return bw;
    }
    if (mode == 3) {
        if (p >= 0.95f) return (int64_t)1;
        if (p >= 0.9f)  return min(bw, (int64_t)2);
        if (p >= 0.8f)  return min(bw, (int64_t)4);
        if (p >= 0.6f)  return min(bw, (int64_t)7);
        return bw;
    }
    // mode == 4 (ultra)
    if (p >= 0.9f)  return (int64_t)1;
    if (p >= 0.8f)  return min(bw, (int64_t)3);
    if (p >= 0.6f)  return min(bw, (int64_t)6);
    return bw;
}

__device__ __forceinline__ int64_t apply_adaptive_all(
    int mode, float p, int64_t base_topk)
{
    if (mode == 0) return base_topk;
    if (mode == 1) {
        if (p >= 0.95f) return min(base_topk, (int64_t)2);
        if (p >= 0.85f) return min(base_topk, (int64_t)3);
        return base_topk;
    }
    // mode == 2 (aggressive)
    if (p >= 0.9f) return (int64_t)1;
    if (p >= 0.7f) return min(base_topk, (int64_t)2);
    if (p >= 0.5f) return min(base_topk, (int64_t)3);
    return base_topk;
}


// -----------------------------------------------------------------------
//   Block-1 kernel — single thread, K=4 serial.
// -----------------------------------------------------------------------

__global__ void build_tree_block1_kernel(
    // Inputs
    const int64_t* __restrict__ rank_preds,            // [K]
    const int64_t* __restrict__ greedy_target,         // [K]
    const float*   __restrict__ greedy_lps,            // [K]
    const int64_t* __restrict__ all_top_target,        // [K, MAX_TOPK]
    const float*   __restrict__ all_top_lps,           // [K, MAX_TOPK]
    const int64_t* __restrict__ rank_slot_topk_table,  // [RANK_CLASSES]
    // Tree outputs
    int64_t* __restrict__ tree_tokens,
    int64_t* __restrict__ tree_parents,
    float*   __restrict__ tree_lps,
    int64_t* __restrict__ tree_ranks,
    int64_t* __restrict__ tree_blocks,
    int64_t* __restrict__ tree_slots,
    // Pending outputs
    int64_t* __restrict__ pend_hidden_slots,
    int64_t* __restrict__ pend_input_ids,
    int64_t* __restrict__ pend_ttt_valid,
    int64_t* __restrict__ pend_node_indices,
    float*   __restrict__ pend_cum_lps,
    // Sizes [4]: {n_nodes_b1, n_active, total_alts1, N_pend}
    int64_t* __restrict__ sizes,
    // Scalar args
    int beam_width,
    int K,
    int MAX_TOPK,
    int GIVE_UP_CLASS,
    int ADAPTIVE_SLOT0_MODE,
    int ADAPTIVE_ALL_MODE)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Worst case K=8; we hold per-slot scratch in registers.
    const int KMAX = 8;
    int64_t ranks[KMAX], gts[KMAX];
    float   gls[KMAX];
    float   cum_glp[KMAX];
    int64_t slot_topks[KMAX];
    int64_t n_alt_slot[KMAX], cum_alt_excl[KMAX];
    int64_t active_rank[KMAX];

    // 1. Load + per-slot adaptive rule + cum_glp
    float running_sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        ranks[k] = rank_preds[k];
        gts[k]   = greedy_target[k];
        gls[k]   = greedy_lps[k];
        running_sum += gls[k];
        cum_glp[k] = running_sum;

        int64_t base = rank_slot_topk_table[ranks[k]];
        float p = __expf(gls[k]);
        slot_topks[k] = (k == 0)
            ? apply_adaptive_slot0(ADAPTIVE_SLOT0_MODE, p, (int64_t)beam_width)
            : apply_adaptive_all(ADAPTIVE_ALL_MODE,   p, base);
    }

    // 2. give_up: first non-zero rank equals GIVE_UP_CLASS
    bool give_up = false;
    for (int k = 0; k < K; ++k) {
        if (ranks[k] != 0) {
            give_up = (ranks[k] == GIVE_UP_CLASS);
            break;
        }
    }

    // 3. Alt + active prefix sums
    int64_t total_alts1 = 0;
    int64_t n_active = 0;
    for (int k = 0; k < K; ++k) {
        n_alt_slot[k] = (slot_topks[k] > 1) ? (slot_topks[k] - 1) : 0;
        cum_alt_excl[k] = total_alts1;
        total_alts1 += n_alt_slot[k];

        active_rank[k] = n_active;
        if (slot_topks[k] > 1) n_active += 1;
    }
    const bool last_active = (slot_topks[K - 1] > 1);
    const int64_t hitch_add = (!give_up && !last_active) ? 1 : 0;

    // 4. Write greedy chain
    for (int k = 0; k < K; ++k) {
        tree_tokens[k]  = gts[k];
        tree_parents[k] = (int64_t)(k - 1);
        tree_lps[k]     = cum_glp[k];
        tree_ranks[k]   = ranks[k];
        tree_blocks[k]  = 0;
        tree_slots[k]   = (int64_t)k;
    }

    // 5. Write alternatives + their pending entries
    for (int k = 0; k < K; ++k) {
        const float parent_lp = (k == 0) ? 0.0f : cum_glp[k - 1];
        const int64_t parent_node = (k == 0) ? (int64_t)-1 : (int64_t)(k - 1);
        const int64_t rank_k = ranks[k];
        const int64_t stop = slot_topks[k];

        for (int j = 1; j < stop; ++j) {
            const int64_t pos = (int64_t)K + cum_alt_excl[k] + (int64_t)(j - 1);
            const int64_t tok = all_top_target[k * MAX_TOPK + j];
            const float   lp  = parent_lp + all_top_lps[k * MAX_TOPK + j];

            tree_tokens[pos]  = tok;
            tree_parents[pos] = parent_node;
            tree_lps[pos]     = lp;
            tree_ranks[pos]   = rank_k;
            tree_blocks[pos]  = 0;
            tree_slots[pos]   = (int64_t)k;

            const int64_t p_pos = n_active + cum_alt_excl[k] + (int64_t)(j - 1);
            pend_hidden_slots[p_pos] = (int64_t)k;
            pend_input_ids[p_pos]    = tok;
            pend_ttt_valid[p_pos]    = (int64_t)(k + 1);
            pend_node_indices[p_pos] = pos;
            pend_cum_lps[p_pos]      = lp;
        }
    }

    // 6. Write active-slot pending entries
    for (int k = 0; k < K; ++k) {
        if (slot_topks[k] > 1) {
            const int64_t pos = active_rank[k];
            pend_hidden_slots[pos] = (int64_t)k;
            pend_input_ids[pos]    = gts[k];
            pend_ttt_valid[pos]    = (int64_t)(k + 1);
            pend_node_indices[pos] = (int64_t)k;
            pend_cum_lps[pos]      = cum_glp[k];
        }
    }

    // 7. Hitchhike entry (if applicable)
    if (hitch_add == 1) {
        const int64_t pos = n_active + total_alts1;
        pend_hidden_slots[pos] = (int64_t)(K - 1);
        pend_input_ids[pos]    = gts[K - 1];
        pend_ttt_valid[pos]    = (int64_t)K;
        pend_node_indices[pos] = (int64_t)(K - 1);
        pend_cum_lps[pos]      = cum_glp[K - 1];
    }

    // 8. Sizes
    sizes[0] = (int64_t)K + total_alts1;
    sizes[1] = n_active;
    sizes[2] = total_alts1;
    sizes[3] = n_active + total_alts1 + hitch_add;
}


// -----------------------------------------------------------------------
//   Fixed-N block-1 kernel: sort + select top-N pending by cum_lp.
// -----------------------------------------------------------------------
//
// Same math as build_tree_block1_kernel for node/pending generation, but
// the final pending-array output has exactly FIXED_N entries. Entries
// are written in cum_lp-desc order of the TOP FIXED_N variable pendings;
// remaining slots (if N_real < FIXED_N) are padded with dummy markers
// (ttt_valid = 0, cum_lp = -INF, etc.) so downstream BFS can mask them.
//
// This eliminates the host sync on sizes[3] (N_pend), enabling static
// shape for the subsequent BFS forward setup — prerequisite for CUDA
// graph capture.

__global__ void build_tree_block1_fixed_n_kernel(
    const int64_t* __restrict__ rank_preds,
    const int64_t* __restrict__ greedy_target,
    const float*   __restrict__ greedy_lps,
    const int64_t* __restrict__ all_top_target,
    const float*   __restrict__ all_top_lps,
    const int64_t* __restrict__ rank_slot_topk_table,
    int64_t* __restrict__ tree_tokens,
    int64_t* __restrict__ tree_parents,
    float*   __restrict__ tree_lps,
    int64_t* __restrict__ tree_ranks,
    int64_t* __restrict__ tree_blocks,
    int64_t* __restrict__ tree_slots,
    int64_t* __restrict__ pend_hidden_slots,
    int64_t* __restrict__ pend_input_ids,
    int64_t* __restrict__ pend_ttt_valid,
    int64_t* __restrict__ pend_node_indices,
    float*   __restrict__ pend_cum_lps,
    int64_t* __restrict__ sizes,          // [4]: {n_nodes_b1, n_active, total_alts1, N_real}
    int beam_width,
    int K,
    int MAX_TOPK,
    int GIVE_UP_CLASS,
    int ADAPTIVE_SLOT0_MODE,
    int ADAPTIVE_ALL_MODE,
    int FIXED_N)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const int KMAX = 8;
    const int NMAX = 64;  // worst case: K*(MAX_TOPK-1) + K + 1 = 36+4+1 = 41; pad to 64
    const float NEG_INF = -1e30f;

    int64_t ranks[KMAX], gts[KMAX];
    float   gls[KMAX];
    float   cum_glp[KMAX];
    int64_t slot_topks[KMAX];
    int64_t n_alt_slot[KMAX], cum_alt_excl[KMAX];

    // Scratch: up to NMAX pending candidates with their cum_lp + fields.
    float   scr_lp[NMAX];
    int64_t scr_hidden[NMAX];
    int64_t scr_inp[NMAX];
    int64_t scr_ttt[NMAX];
    int64_t scr_node[NMAX];
    int n_real = 0;

    // 1. Load + per-slot adaptive rule + cum_glp
    float running_sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        ranks[k] = rank_preds[k];
        gts[k]   = greedy_target[k];
        gls[k]   = greedy_lps[k];
        running_sum += gls[k];
        cum_glp[k] = running_sum;

        int64_t base = rank_slot_topk_table[ranks[k]];
        float p = __expf(gls[k]);
        slot_topks[k] = (k == 0)
            ? apply_adaptive_slot0(ADAPTIVE_SLOT0_MODE, p, (int64_t)beam_width)
            : apply_adaptive_all(ADAPTIVE_ALL_MODE, p, base);
    }

    // 2. give_up
    bool give_up = false;
    for (int k = 0; k < K; ++k) {
        if (ranks[k] != 0) {
            give_up = (ranks[k] == GIVE_UP_CLASS);
            break;
        }
    }

    // 3. Alt prefix sum
    int64_t total_alts1 = 0;
    int64_t n_active = 0;
    for (int k = 0; k < K; ++k) {
        n_alt_slot[k] = (slot_topks[k] > 1) ? (slot_topks[k] - 1) : 0;
        cum_alt_excl[k] = total_alts1;
        total_alts1 += n_alt_slot[k];
        if (slot_topks[k] > 1) n_active += 1;
    }
    const bool last_active = (slot_topks[K - 1] > 1);
    const int64_t hitch_add = (!give_up && !last_active) ? 1 : 0;

    // 4. Write greedy chain (K nodes, unchanged)
    for (int k = 0; k < K; ++k) {
        tree_tokens[k]  = gts[k];
        tree_parents[k] = (int64_t)(k - 1);
        tree_lps[k]     = cum_glp[k];
        tree_ranks[k]   = ranks[k];
        tree_blocks[k]  = 0;
        tree_slots[k]   = (int64_t)k;
    }

    // 5. Write alternatives (tree-buffer side, unchanged) + collect to scratch
    for (int k = 0; k < K; ++k) {
        const float parent_lp = (k == 0) ? 0.0f : cum_glp[k - 1];
        const int64_t parent_node = (k == 0) ? (int64_t)-1 : (int64_t)(k - 1);
        const int64_t rank_k = ranks[k];
        const int64_t stop = slot_topks[k];

        for (int j = 1; j < stop; ++j) {
            const int64_t pos = (int64_t)K + cum_alt_excl[k] + (int64_t)(j - 1);
            const int64_t tok = all_top_target[k * MAX_TOPK + j];
            const float   lp  = parent_lp + all_top_lps[k * MAX_TOPK + j];

            tree_tokens[pos]  = tok;
            tree_parents[pos] = parent_node;
            tree_lps[pos]     = lp;
            tree_ranks[pos]   = rank_k;
            tree_blocks[pos]  = 0;
            tree_slots[pos]   = (int64_t)k;

            // Scratch pending entry
            if (n_real < NMAX) {
                scr_lp[n_real]     = lp;
                scr_hidden[n_real] = (int64_t)k;
                scr_inp[n_real]    = tok;
                scr_ttt[n_real]    = (int64_t)(k + 1);
                scr_node[n_real]   = pos;
                n_real += 1;
            }
        }
    }

    // 6. Active-slot pending entries → scratch
    for (int k = 0; k < K; ++k) {
        if (slot_topks[k] > 1 && n_real < NMAX) {
            scr_lp[n_real]     = cum_glp[k];
            scr_hidden[n_real] = (int64_t)k;
            scr_inp[n_real]    = gts[k];
            scr_ttt[n_real]    = (int64_t)(k + 1);
            scr_node[n_real]   = (int64_t)k;
            n_real += 1;
        }
    }

    // 7. Hitchhike → scratch
    if (hitch_add == 1 && n_real < NMAX) {
        scr_lp[n_real]     = cum_glp[K - 1];
        scr_hidden[n_real] = (int64_t)(K - 1);
        scr_inp[n_real]    = gts[K - 1];
        scr_ttt[n_real]    = (int64_t)K;
        scr_node[n_real]   = (int64_t)(K - 1);
        n_real += 1;
    }

    // 8. Selection-sort + write top FIXED_N to pend_buf (padding after).
    //    Selection sort: for out slot i in [0, FIXED_N), scan scratch for
    //    the max remaining cum_lp and write it; mark as used (-INF).
    for (int i = 0; i < FIXED_N; ++i) {
        int   best_idx = -1;
        float best_lp  = NEG_INF;
        for (int j = 0; j < n_real; ++j) {
            if (scr_lp[j] > best_lp) {
                best_lp  = scr_lp[j];
                best_idx = j;
            }
        }
        if (best_idx >= 0) {
            pend_hidden_slots[i] = scr_hidden[best_idx];
            pend_input_ids[i]    = scr_inp[best_idx];
            pend_ttt_valid[i]    = scr_ttt[best_idx];
            pend_node_indices[i] = scr_node[best_idx];
            pend_cum_lps[i]      = scr_lp[best_idx];
            scr_lp[best_idx] = NEG_INF;  // mark used
        } else {
            // dummy: ttt_valid=0 signals BFS forward to mask this row.
            pend_hidden_slots[i] = 0;   // safe hidden index
            pend_input_ids[i]    = 0;
            pend_ttt_valid[i]    = 0;   // dummy marker
            pend_node_indices[i] = 0;
            pend_cum_lps[i]      = NEG_INF;
        }
    }

    // 9. Sizes: sizes[3] = n_real (informational only; N_pend is FIXED_N on caller side).
    sizes[0] = (int64_t)K + total_alts1;
    sizes[1] = n_active;
    sizes[2] = total_alts1;
    sizes[3] = (int64_t)n_real;
}


// -----------------------------------------------------------------------
//   BFS kernel — grid=N, each block handles one leaf.
// -----------------------------------------------------------------------
//
// Each block recomputes per-leaf slot_topks and the cross-leaf prefix sum
// to locate its own alt offset. This is redundant but cheap: O(N*K) reads
// per block, N ≤ 15 in production. One launch replaces triton's 2.
//
// Writes:
//   tree_start + leaf*K + slot         -> greedy chain node (K entries)
//   tree_start + N*K + leaf_alt_base   -> leaf's alt nodes
//
// total_alts_2 (global scalar) is written only by block 0 so a single
// copy back to host picks it up.

__global__ void build_tree_bfs_kernel(
    // Inputs
    const int64_t* __restrict__ all_rank_preds,        // [N, K]
    const int64_t* __restrict__ all_greedy_target,     // [N, K]
    const float*   __restrict__ all_greedy_lps,        // [N, K]
    const int64_t* __restrict__ top_target_all,        // [N, K, MAX_TOPK]
    const float*   __restrict__ top_lps_all,           // [N, K, MAX_TOPK]
    const float*   __restrict__ pend_cum_lps,          // [N]
    const int64_t* __restrict__ pend_node_indices,     // [N]
    const int64_t* __restrict__ rank_slot_topk_table,  // [RANK_CLASSES]
    // Tree outputs
    int64_t* __restrict__ tree_tokens,
    int64_t* __restrict__ tree_parents,
    float*   __restrict__ tree_lps,
    int64_t* __restrict__ tree_ranks,
    int64_t* __restrict__ tree_blocks,
    int64_t* __restrict__ tree_slots,
    // Scalar output
    int64_t* __restrict__ total_alts_2,  // [1]
    // Scalar args
    int N,
    int tree_start,
    int K,
    int MAX_TOPK,
    int GIVE_UP_CLASS,
    int ADAPTIVE_ALL_MODE,
    int PEND_DEPTH)
{
    const int leaf = blockIdx.x;
    if (leaf >= N || threadIdx.x != 0) return;

    const int KMAX = 8;
    int64_t my_slot_topks[KMAX];
    int64_t my_ranks[KMAX];
    int64_t my_gts[KMAX];
    float   my_gls[KMAX];

    // 1. Compute leaf_alt_base = sum_{l<leaf} n_alt_per_leaf[l]
    //    AND block-0 also records total for scalar output.
    int64_t leaf_alt_base = 0;
    int64_t total_alts = 0;
    for (int l = 0; l < N; ++l) {
        int64_t n_alt_l = 0;
        for (int k = 0; k < K; ++k) {
            int64_t r = all_rank_preds[l * K + k];
            int64_t base = rank_slot_topk_table[r];
            float p = __expf(all_greedy_lps[l * K + k]);
            int64_t stop = apply_adaptive_all(ADAPTIVE_ALL_MODE, p, base);
            if (stop > 1) n_alt_l += (stop - 1);

            // Cache our own leaf's per-slot data for later passes
            if (l == leaf) {
                my_slot_topks[k] = stop;
                my_ranks[k]      = r;
                my_gts[k]        = all_greedy_target[l * K + k];
                my_gls[k]        = all_greedy_lps[l * K + k];
            }
        }
        if (l < leaf) leaf_alt_base += n_alt_l;
        total_alts += n_alt_l;
    }

    // Block 0 exposes total
    if (leaf == 0) *total_alts_2 = total_alts;

    // 2. Greedy chain: K consecutive slots
    const int64_t chain_base = (int64_t)tree_start + (int64_t)leaf * K;
    const int64_t pend_node   = pend_node_indices[leaf];
    const float   pend_lp     = pend_cum_lps[leaf];

    float running = 0.0f;
    float cum_glp[KMAX];
    for (int k = 0; k < K; ++k) {
        running += my_gls[k];
        cum_glp[k] = running;

        const int64_t pos = chain_base + (int64_t)k;
        tree_tokens[pos]  = my_gts[k];
        tree_parents[pos] = (k == 0) ? pend_node : (pos - 1);
        tree_lps[pos]     = pend_lp + cum_glp[k];
        tree_ranks[pos]   = my_ranks[k];
        tree_blocks[pos]  = (int64_t)PEND_DEPTH;
        tree_slots[pos]   = (int64_t)k;
    }

    // 3. Alternatives
    int64_t cum_alt = 0;
    const int64_t alt_base = (int64_t)tree_start + (int64_t)N * K + leaf_alt_base;

    for (int k = 0; k < K; ++k) {
        const int64_t stop = my_slot_topks[k];
        if (stop <= 1) continue;

        const float parent_lp = (k == 0) ? 0.0f : cum_glp[k - 1];
        const int64_t parent_node = (k == 0) ? pend_node : (chain_base + (int64_t)(k - 1));
        const int64_t rank_k = my_ranks[k];

        for (int j = 1; j < stop; ++j) {
            const int64_t pos = alt_base + cum_alt;
            const int64_t tok = top_target_all[(leaf * K + k) * MAX_TOPK + j];
            const float   lp  = pend_lp + parent_lp + top_lps_all[(leaf * K + k) * MAX_TOPK + j];

            tree_tokens[pos]  = tok;
            tree_parents[pos] = parent_node;
            tree_lps[pos]     = lp;
            tree_ranks[pos]   = rank_k;
            tree_blocks[pos]  = (int64_t)PEND_DEPTH;
            tree_slots[pos]   = (int64_t)k;

            cum_alt += 1;
        }
    }
}


// -----------------------------------------------------------------------
//   aten::Tensor launchers
// -----------------------------------------------------------------------

void build_tree_block1_cuda(
    at::Tensor rank_preds,
    at::Tensor greedy_target,
    at::Tensor greedy_lps,
    at::Tensor all_top_target,
    at::Tensor all_top_lps,
    at::Tensor rank_slot_topk_table,
    at::Tensor tree_tokens,
    at::Tensor tree_parents,
    at::Tensor tree_lps,
    at::Tensor tree_ranks,
    at::Tensor tree_blocks,
    at::Tensor tree_slots,
    at::Tensor pend_hidden_slots,
    at::Tensor pend_input_ids,
    at::Tensor pend_ttt_valid,
    at::Tensor pend_node_indices,
    at::Tensor pend_cum_lps,
    at::Tensor sizes,
    int64_t beam_width,
    int64_t K,
    int64_t MAX_TOPK,
    int64_t GIVE_UP_CLASS,
    int64_t ADAPTIVE_SLOT0_MODE,
    int64_t ADAPTIVE_ALL_MODE)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    build_tree_block1_kernel<<<1, 1, 0, stream>>>(
        rank_preds.data_ptr<int64_t>(),
        greedy_target.data_ptr<int64_t>(),
        greedy_lps.data_ptr<float>(),
        all_top_target.data_ptr<int64_t>(),
        all_top_lps.data_ptr<float>(),
        rank_slot_topk_table.data_ptr<int64_t>(),
        tree_tokens.data_ptr<int64_t>(),
        tree_parents.data_ptr<int64_t>(),
        tree_lps.data_ptr<float>(),
        tree_ranks.data_ptr<int64_t>(),
        tree_blocks.data_ptr<int64_t>(),
        tree_slots.data_ptr<int64_t>(),
        pend_hidden_slots.data_ptr<int64_t>(),
        pend_input_ids.data_ptr<int64_t>(),
        pend_ttt_valid.data_ptr<int64_t>(),
        pend_node_indices.data_ptr<int64_t>(),
        pend_cum_lps.data_ptr<float>(),
        sizes.data_ptr<int64_t>(),
        (int)beam_width,
        (int)K,
        (int)MAX_TOPK,
        (int)GIVE_UP_CLASS,
        (int)ADAPTIVE_SLOT0_MODE,
        (int)ADAPTIVE_ALL_MODE);
}

void build_tree_block1_fixed_n_cuda(
    at::Tensor rank_preds,
    at::Tensor greedy_target,
    at::Tensor greedy_lps,
    at::Tensor all_top_target,
    at::Tensor all_top_lps,
    at::Tensor rank_slot_topk_table,
    at::Tensor tree_tokens,
    at::Tensor tree_parents,
    at::Tensor tree_lps,
    at::Tensor tree_ranks,
    at::Tensor tree_blocks,
    at::Tensor tree_slots,
    at::Tensor pend_hidden_slots,
    at::Tensor pend_input_ids,
    at::Tensor pend_ttt_valid,
    at::Tensor pend_node_indices,
    at::Tensor pend_cum_lps,
    at::Tensor sizes,
    int64_t beam_width,
    int64_t K,
    int64_t MAX_TOPK,
    int64_t GIVE_UP_CLASS,
    int64_t ADAPTIVE_SLOT0_MODE,
    int64_t ADAPTIVE_ALL_MODE,
    int64_t FIXED_N)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    build_tree_block1_fixed_n_kernel<<<1, 1, 0, stream>>>(
        rank_preds.data_ptr<int64_t>(),
        greedy_target.data_ptr<int64_t>(),
        greedy_lps.data_ptr<float>(),
        all_top_target.data_ptr<int64_t>(),
        all_top_lps.data_ptr<float>(),
        rank_slot_topk_table.data_ptr<int64_t>(),
        tree_tokens.data_ptr<int64_t>(),
        tree_parents.data_ptr<int64_t>(),
        tree_lps.data_ptr<float>(),
        tree_ranks.data_ptr<int64_t>(),
        tree_blocks.data_ptr<int64_t>(),
        tree_slots.data_ptr<int64_t>(),
        pend_hidden_slots.data_ptr<int64_t>(),
        pend_input_ids.data_ptr<int64_t>(),
        pend_ttt_valid.data_ptr<int64_t>(),
        pend_node_indices.data_ptr<int64_t>(),
        pend_cum_lps.data_ptr<float>(),
        sizes.data_ptr<int64_t>(),
        (int)beam_width,
        (int)K,
        (int)MAX_TOPK,
        (int)GIVE_UP_CLASS,
        (int)ADAPTIVE_SLOT0_MODE,
        (int)ADAPTIVE_ALL_MODE,
        (int)FIXED_N);
}

void build_tree_bfs_cuda(
    at::Tensor all_rank_preds,
    at::Tensor all_greedy_target,
    at::Tensor all_greedy_lps,
    at::Tensor top_target_all,
    at::Tensor top_lps_all,
    at::Tensor pend_cum_lps,
    at::Tensor pend_node_indices,
    at::Tensor rank_slot_topk_table,
    at::Tensor tree_tokens,
    at::Tensor tree_parents,
    at::Tensor tree_lps,
    at::Tensor tree_ranks,
    at::Tensor tree_blocks,
    at::Tensor tree_slots,
    at::Tensor total_alts_2,
    int64_t N,
    int64_t tree_start,
    int64_t K,
    int64_t MAX_TOPK,
    int64_t GIVE_UP_CLASS,
    int64_t ADAPTIVE_ALL_MODE,
    int64_t PEND_DEPTH)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    build_tree_bfs_kernel<<<(int)N, 1, 0, stream>>>(
        all_rank_preds.data_ptr<int64_t>(),
        all_greedy_target.data_ptr<int64_t>(),
        all_greedy_lps.data_ptr<float>(),
        top_target_all.data_ptr<int64_t>(),
        top_lps_all.data_ptr<float>(),
        pend_cum_lps.data_ptr<float>(),
        pend_node_indices.data_ptr<int64_t>(),
        rank_slot_topk_table.data_ptr<int64_t>(),
        tree_tokens.data_ptr<int64_t>(),
        tree_parents.data_ptr<int64_t>(),
        tree_lps.data_ptr<float>(),
        tree_ranks.data_ptr<int64_t>(),
        tree_blocks.data_ptr<int64_t>(),
        tree_slots.data_ptr<int64_t>(),
        total_alts_2.data_ptr<int64_t>(),
        (int)N,
        (int)tree_start,
        (int)K,
        (int)MAX_TOPK,
        (int)GIVE_UP_CLASS,
        (int)ADAPTIVE_ALL_MODE,
        (int)PEND_DEPTH);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_tree_block1_cuda", &build_tree_block1_cuda,
          "SpecBlock-specific block-1 tree build (CUDA).");
    m.def("build_tree_block1_fixed_n_cuda", &build_tree_block1_fixed_n_cuda,
          "Fixed-N variant: top-N pending selection, padded output.");
    m.def("build_tree_bfs_cuda", &build_tree_bfs_cuda,
          "SpecBlock-specific BFS tree build (CUDA).");
}
