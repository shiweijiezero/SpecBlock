"""Release-integrity checks for optional SpecBlock tree-build backends.

These tests intentionally avoid importing torch or triton so they can run on a
CPU-only packaging/CI host.  They catch source-release omissions before users
hit a lazy import during evaluation.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS_DIR = REPO_ROOT / "benchmarks_hf" / "algorithms"


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class TreeBuildReleaseCompletenessTest(unittest.TestCase):
    def test_sdist_manifest_includes_benchmark_runtime_sources(self) -> None:
        manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("recursive-include benchmarks_hf *.py *.cu", manifest)

    def test_legacy_setup_version_file_is_present(self) -> None:
        version_file = REPO_ROOT / "version.txt"
        self.assertTrue(version_file.is_file())
        self.assertEqual("0.2.0", version_file.read_text(encoding="utf-8").strip())

    def test_tree_build_runtime_files_are_present(self) -> None:
        required = (
            "tree_build_triton.py",
            "tree_build_cuda_loader.py",
            "tree_build_cuda.cu",
        )
        missing = [name for name in required if not (ALGORITHMS_DIR / name).is_file()]
        self.assertEqual([], missing, f"missing tree-build release files: {missing}")

    def test_triton_builder_exports_expected_api(self) -> None:
        functions = _function_names(ALGORITHMS_DIR / "tree_build_triton.py")
        self.assertTrue(
            {"triton_build_block1", "triton_build_bfs"}.issubset(functions),
            functions,
        )

    def test_cuda_loader_exports_expected_api(self) -> None:
        functions = _function_names(ALGORITHMS_DIR / "tree_build_cuda_loader.py")
        self.assertTrue(
            {
                "cuda_build_block1",
                "cuda_build_block1_fixed_n",
                "cuda_build_bfs",
            }.issubset(functions),
            functions,
        )

    def test_cuda_extension_exports_loader_symbols(self) -> None:
        loader = (ALGORITHMS_DIR / "tree_build_cuda_loader.py").read_text(
            encoding="utf-8"
        )
        source = (ALGORITHMS_DIR / "tree_build_cuda.cu").read_text(
            encoding="utf-8"
        )
        symbols = (
            "build_tree_block1_cuda",
            "build_tree_block1_fixed_n_cuda",
            "build_tree_bfs_cuda",
        )
        for symbol in symbols:
            with self.subTest(symbol=symbol):
                self.assertIn(f"mod.{symbol}(", loader)
                self.assertIn(f'm.def("{symbol}"', source)

    def test_true_batch_runtime_files_are_present(self) -> None:
        required = (
            "eagle3_batch.py",
            "specblock_batch.py",
            "hf_batch_invariant.py",
            "target_kv_copy_triton.py",
        )
        missing = [name for name in required if not (ALGORITHMS_DIR / name).is_file()]
        self.assertEqual([], missing, f"missing true-batch runtime files: {missing}")

    def test_sglang_specblock_model_sources_are_present(self) -> None:
        models_dir = (
            REPO_ROOT / "sglang-main" / "python" / "sglang" / "srt" / "models"
        )
        required = (
            "registry.py",
            "_specblock_inference.py",
            "_specblock_shift_inference.py",
            "_tree_attention_triton.py",
            "llama_specblock_shift.py",
        )
        missing = [name for name in required if not (models_dir / name).is_file()]
        self.assertEqual([], missing, f"missing SGLang model sources: {missing}")


if __name__ == "__main__":
    unittest.main()
