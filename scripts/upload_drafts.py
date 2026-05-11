#!/usr/bin/env python3
"""Upload a draft-model checkpoint to HuggingFace and ModelScope.

Reads credentials from environment variables — never hard-code tokens in the
repository. On a workstation, add the following to your shell rc file:

    export HF_TOKEN=<your-huggingface-write-token>
    export MODELSCOPE_API_TOKEN=<your-modelscope-access-token>

Usage:
    # Llama-3.1-8B SpecBlock draft
    python scripts/upload_drafts.py \\
        ./model/Llama-3.1-8B-Instruct/specblock-layer2/<your-checkpoint> \\
        <org>/SpecBlock-Llama-3.1-8B-Instruct

    # Qwen3-8B SpecBlock draft
    python scripts/upload_drafts.py \\
        ./model/Qwen3-8B/specblock-layer2/<your-checkpoint> \\
        <org>/SpecBlock-Qwen3-8B

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
    create_repo(repo_id, token=token, exist_ok=True, private=private, repo_type="model")
    HfApi(token=token).upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"upload {local_dir.name}",
    )


def upload_modelscope(local_dir: Path, model_id: str, private: bool) -> None:
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(os.environ["MODELSCOPE_API_TOKEN"])
    api.create_repo(
        repo_id=model_id,
        visibility="private" if private else "public",
        repo_type="model",
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=model_id,
        folder_path=str(local_dir),
        repo_type="model",
        commit_message=f"upload {local_dir.name}",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("local_dir", help="Local checkpoint directory")
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
        print(f"[HF]  Done. https://huggingface.co/{args.repo_name}")

    if not args.skip_ms:
        if "MODELSCOPE_API_TOKEN" not in os.environ:
            sys.exit("MODELSCOPE_API_TOKEN env var not set")
        print(f"[MS]  Uploading {local_dir} -> {args.repo_name} (private={private})")
        upload_modelscope(local_dir, args.repo_name, private)
        print(f"[MS]  Done. https://modelscope.cn/models/{args.repo_name}")


if __name__ == "__main__":
    main()
