#!/usr/bin/env python3
"""Launch the speculative decoding visualization UI."""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import create_ui


def main():
    parser = argparse.ArgumentParser(
        description="Launch SpecForge visualization UI"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to bind to (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public shareable link"
    )
    args = parser.parse_args()

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
