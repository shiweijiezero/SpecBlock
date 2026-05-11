# Examples

End-to-end training launch scripts.

| Script | Method | Setup |
|---|---|---|
| `eagle_example/run_llama3_eagle3_online.sh` | EAGLE3 (baseline) | Single-node, Llama-3.1-8B |
| `specblock/run_llama3_specblock_online.sh` | SpecBlock (ours) | Single-node, Llama-3.1-8B |
| `specblock/run_specblock_3node.sh` | SpecBlock (ours) | Multi-node, Llama-3.1-8B |

Edit the paths and hyper-parameters at the top of each script before launching. The single-node scripts take `[NUM_GPUS] [TP_SIZE]` as positional arguments. Data preparation is done separately under `scripts/`.
