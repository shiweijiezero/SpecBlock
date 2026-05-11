"""Common utility functions for dataset loading."""

import json
import os
from pathlib import Path
from typing import Iterator

import requests


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded files."""
    cache_dir = Path.home() / ".cache" / "benchmarks_hf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_and_cache_file(url: str, filename: str) -> Path:
    """Download a file from URL and cache it locally.

    Args:
        url: URL to download from
        filename: Name to save the file as in cache

    Returns:
        Path to the cached file
    """
    cache_dir = get_cache_dir()
    filepath = cache_dir / filename

    if not filepath.exists():
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        filepath.write_bytes(response.content)
        print(f"Downloaded to {filepath}")
    else:
        print(f"Using cached file: {filepath}")

    return filepath


def read_jsonl(filename: str) -> Iterator[dict]:
    """Read JSONL file and yield dictionaries.

    Args:
        filename: Name of the cached file to read

    Yields:
        Parsed JSON objects from each line
    """
    cache_dir = get_cache_dir()
    filepath = cache_dir / filename

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
