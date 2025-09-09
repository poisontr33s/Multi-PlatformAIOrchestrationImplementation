#!/usr/bin/env python3
"""
Apply file blocks from agent output.
Parses markdown with code blocks and creates/updates files accordingly.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


def parse_file_blocks(content: str) -> List[Tuple[str, str, str]]:
    """
    Parse file blocks from markdown content.

    Args:
        content: Markdown content with file blocks

    Returns:
        List of (filename, language, content) tuples
    """
    # Pattern to match file blocks with name attribute
    pattern = r"```(\w+)?\s+name=([^\s\n]+)\n(.*?)```"

    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL | re.MULTILINE):
        language = match.group(1) or ""
        filename = match.group(2)
        file_content = match.group(3)

        # Clean up filename (remove quotes if present)
        filename = filename.strip("'\"")

        blocks.append((filename, language, file_content))

    return blocks


def apply_file_block(filename: str, content: str, dry_run: bool = False) -> bool:
    """
    Apply a file block to create or update a file.

    Args:
        filename: Target filename
        content: File content
        dry_run: If True, only print what would be done

    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(filename)

        if dry_run:
            print(f"Would {'update' if file_path.exists() else 'create'}: {filename}")
            print(f"Content length: {len(content)} characters")
            return True

        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        action = "Updated" if file_path.exists() else "Created"
        print(f"{action}: {filename}")
        return True

    except Exception as e:
        print(f"Error applying file block for {filename}: {e}", file=sys.stderr)
        return False


def main():
    """Main function to process file blocks from stdin or file."""
    parser = argparse.ArgumentParser(
        description="Apply file blocks from agent output",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Input file with file blocks (default: stdin)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--filter-lang",
        help="Only process blocks with specified language",
    )
    parser.add_argument(
        "--filter-path",
        help="Only process files matching path pattern",
    )

    args = parser.parse_args()

    # Read input
    if args.input_file:
        try:
            with open(args.input_file, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            return 1
    else:
        content = sys.stdin.read()

    # Parse file blocks
    blocks = parse_file_blocks(content)

    if not blocks:
        print("No file blocks found in input", file=sys.stderr)
        return 1

    print(f"Found {len(blocks)} file blocks")

    # Apply filters
    filtered_blocks = blocks

    if args.filter_lang:
        filtered_blocks = [
            (f, l, c) for f, l, c in filtered_blocks if l == args.filter_lang
        ]
        print(
            f"Filtered to {len(filtered_blocks)} blocks matching language: {args.filter_lang}"
        )

    if args.filter_path:
        filtered_blocks = [
            (f, l, c) for f, l, c in filtered_blocks if args.filter_path in f
        ]
        print(
            f"Filtered to {len(filtered_blocks)} blocks matching path: {args.filter_path}"
        )

    # Apply file blocks
    success_count = 0
    for filename, language, file_content in filtered_blocks:
        if apply_file_block(filename, file_content, args.dry_run):
            success_count += 1

    print(f"Successfully processed {success_count}/{len(filtered_blocks)} file blocks")

    if success_count < len(filtered_blocks):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
