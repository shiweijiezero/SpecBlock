#!/usr/bin/env python3
"""Upload a pre-processed training dataset to HuggingFace Datasets and ModelScope Datasets.

Reads credentials from environment variables — never hard-code tokens in the
repository. On a workstation, add the following to your shell rc file:

    export HF_TOKEN=<your-huggingface-write-token>
    export MODELSCOPE_API_TOKEN=<your-modelscope-access-token>

Usage:
    python scripts/upload_data.py \\
        ./data/sharegpt_llama31_distilled \\
        <org>/SpecBlock-train-data

Layout: pass any folder; everything under it is uploaded as-is. Typical layout:
    data_dir/
    ├── train.jsonl
    ├── eval.jsonl
    └── README.md           # short dataset card (optional)

Flags:
    --public      Upload as public (default: private)
    --skip-hf     Skip HuggingFace upload
    --skip-ms     Skip ModelScope upload
"""
import argparse
import os
import sys
from pathlib import Path


def upload_hf(local_dir: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi, create_repo

    token = os.environ["HF_TOKEN"]
    create_repo(repo_id, token=token, exist_ok=True, private=private, repo_type="dataset")
    HfApi(token=token).upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"upload {local_dir.name}",
    )


def upload_modelscope(local_dir: Path, repo_id: str, private: bool) -> None:
    from modelscope.hub.api import HubApi

    # Monkey-patch create_repo to be tolerant: ModelScope SDK's upload_folder
    # internally calls create_repo even when the repo already exists, and the
    # error response from the server is not recognized as "already exists" so
    # `exist_ok=True` does not help. Swallow create-permission / probe errors
    # and proceed to upload — the repo must already exist (manually created
    # via the web UI).
    _orig_create = HubApi.create_repo

    def _patched_create(self, repo_id, *args, **kwargs):
        try:
            return _orig_create(self, repo_id, *args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "10020101037" in msg or "已存在" in msg or "无权创建" in msg or "初始化项目失败" in msg:
                print(f"  (skipping create_repo for {repo_id}: assume already exists)")
                repo_type = kwargs.get("repo_type", "model")
                return f"https://www.modelscope.cn/{repo_type}s/{repo_id}"
            raise

    HubApi.create_repo = _patched_create

    api = HubApi()
    api.login(os.environ["MODELSCOPE_API_TOKEN"])
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        repo_type="dataset",
        commit_message=f"upload {local_dir.name}",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("local_dir", help="Local dataset directory")
    p.add_argument("repo_name", help="Repo name in <org>/<name> form")
    p.add_argument("--public", action="store_true", help="Upload as public (default: private)")
    p.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace upload")
    p.add_argument("--skip-ms", action="store_true", help="Skip ModelScope upload")
    args = p.parse_args()

    local_dir = Path(args.local_dir).resolve()
    if not local_dir.is_dir():
        sys.exit(f"Local dir not found: {local_dir}")

    private = not args.public

    if not args.skip_hf:
        if "HF_TOKEN" not in os.environ:
            sys.exit("HF_TOKEN env var not set")
        print(f"[HF]  Uploading {local_dir} -> {args.repo_name} (private={private})")
        upload_hf(local_dir, args.repo_name, private)
        print(f"[HF]  Done. https://huggingface.co/datasets/{args.repo_name}")

    if not args.skip_ms:
        if "MODELSCOPE_API_TOKEN" not in os.environ:
            sys.exit("MODELSCOPE_API_TOKEN env var not set")
        print(f"[MS]  Uploading {local_dir} -> {args.repo_name} (private={private})")
        upload_modelscope(local_dir, args.repo_name, private)
        print(f"[MS]  Done. https://modelscope.cn/datasets/{args.repo_name}")


if __name__ == "__main__":
    main()
