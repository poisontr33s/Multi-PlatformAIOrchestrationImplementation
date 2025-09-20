#!/usr/bin/env python3
"""
Code Block Applier Utility

Parses and applies "file blocks" from AI-generated responses into real files.
Useful for AI assistants that generate code with file paths in their responses.

Usage:
    python scripts/apply_code_blocks.py <response_file>
    echo "AI response" | python scripts/apply_code_blocks.py
    ai-orchestrator chat --model gpt-4 --prompt "Create a hello world" | python scripts/apply_code_blocks.py
"""

import re
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import difflib
from datetime import datetime


class CodeBlock:
    """Represents a code block with file path and content."""
    
    def __init__(self, filepath: str, content: str, language: str = None):
        self.filepath = filepath
        self.content = content
        self.language = language
        
    def __repr__(self):
        return f"CodeBlock(filepath='{self.filepath}', language='{self.language}', lines={len(self.content.splitlines())})"


class CodeBlockParser:
    """Parses AI responses to extract code blocks with file paths."""
    
    # Patterns to match different code block formats
    PATTERNS = [
        # Standard markdown with filepath
        r'```(?:(\w+)\s+)?(.*?)\n(.*?)```',
        
        # File path before code block
        r'(?:File:|Filepath?:|Path:)\s*`?([^\n`]+)`?\s*\n```(?:(\w+))?\n(.*?)```',
        
        # Comments with file paths
        r'```(?:(\w+))?\n(?:#|//|/\*)\s*(?:File:|Path:)\s*([^\n*]+)(?:\*/\s*)?\n(.*?)```',
        
        # XML-style file blocks
        r'<file\s+(?:path|name)="([^"]+)"(?:\s+language="(\w+)")?>(?:\n)?(.*?)</file>',
        
        # YAML front matter style
        r'```(?:(\w+))?\n---\nfile:\s*([^\n]+)\n---\n(.*?)```',
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.DOTALL | re.MULTILINE) for pattern in self.PATTERNS]
    
    def parse(self, text: str) -> List[CodeBlock]:
        """Extract code blocks from text."""
        blocks = []
        
        # Try each pattern
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            
            for match in matches:
                if len(match) == 3:
                    language, filepath, content = match
                    # Handle different match group orders
                    if self._looks_like_filepath(language) and not self._looks_like_filepath(filepath):
                        language, filepath = filepath, language
                    elif not self._looks_like_filepath(filepath):
                        # Maybe it's (language, content, filepath) order
                        if self._looks_like_filepath(content):
                            language, filepath, content = language, content, filepath
                
                if self._looks_like_filepath(filepath):
                    blocks.append(CodeBlock(
                        filepath=filepath.strip(),
                        content=content.strip(),
                        language=language.strip() if language else None
                    ))
        
        # Also try to find files mentioned in comments or text
        blocks.extend(self._find_implicit_files(text))
        
        # Remove duplicates (same filepath)
        seen_files = set()
        unique_blocks = []
        for block in blocks:
            if block.filepath not in seen_files:
                seen_files.add(block.filepath)
                unique_blocks.append(block)
        
        return unique_blocks
    
    def _looks_like_filepath(self, text: str) -> bool:
        """Check if text looks like a file path."""
        if not text:
            return False
            
        # Clean up the text
        text = text.strip()
        
        # Check for common file indicators
        indicators = [
            '/' in text,  # Unix path
            '\\' in text,  # Windows path
            '.' in text and not text.startswith('.') and len(text.split('.')) >= 2,  # Has extension
            text.endswith(('.py', '.js', '.ts', '.html', '.css', '.md', '.txt', '.json', '.yaml', '.yml')),
            text.startswith(('./', '../', '/', '~/')),
        ]
        
        return any(indicators) and len(text.split()) == 1  # Single word/path
    
    def _find_implicit_files(self, text: str) -> List[CodeBlock]:
        """Find files mentioned in text but not in explicit code blocks."""
        # This is a basic implementation - could be enhanced
        blocks = []
        
        # Look for patterns like "Create file xyz.py:"
        pattern = r'(?:Create|Update|Write)\s+(?:file\s+|the\s+file\s+)?([^\s:]+\.[a-zA-Z]+):?\s*\n```(?:(\w+))?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        
        for filepath, language, content in matches:
            if self._looks_like_filepath(filepath):
                blocks.append(CodeBlock(
                    filepath=filepath.strip(),
                    content=content.strip(),
                    language=language.strip() if language else None
                ))
        
        return blocks


class CodeBlockApplier:
    """Applies code blocks to the filesystem."""
    
    def __init__(self, base_dir: str = ".", backup: bool = True, dry_run: bool = False):
        self.base_dir = Path(base_dir).resolve()
        self.backup = backup
        self.dry_run = dry_run
        self.applied_files = []
        
    def apply_blocks(self, blocks: List[CodeBlock]) -> Dict[str, str]:
        """Apply all code blocks to files."""
        results = {}
        
        for block in blocks:
            try:
                result = self.apply_block(block)
                results[block.filepath] = result
            except Exception as e:
                results[block.filepath] = f"ERROR: {str(e)}"
        
        return results
    
    def apply_block(self, block: CodeBlock) -> str:
        """Apply a single code block to a file."""
        filepath = self.base_dir / block.filepath
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists
        file_exists = filepath.exists()
        
        if self.dry_run:
            action = "UPDATE" if file_exists else "CREATE"
            return f"DRY RUN: Would {action} {filepath} ({len(block.content.splitlines())} lines)"
        
        # Create backup if file exists and backup is enabled
        if file_exists and self.backup:
            backup_path = self._create_backup(filepath)
            backup_info = f" (backup: {backup_path})"
        else:
            backup_info = ""
        
        # Write the content
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(block.content)
                
            self.applied_files.append(str(filepath))
            
            action = "UPDATED" if file_exists else "CREATED"
            return f"{action}: {filepath}{backup_info}"
            
        except Exception as e:
            return f"ERROR writing {filepath}: {str(e)}"
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create a backup of the existing file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f"{filepath.suffix}.backup_{timestamp}")
        
        import shutil
        shutil.copy2(filepath, backup_path)
        
        return backup_path
    
    def show_diff(self, block: CodeBlock) -> str:
        """Show diff between existing file and new content."""
        filepath = self.base_dir / block.filepath
        
        if not filepath.exists():
            return f"NEW FILE: {filepath}\n" + "\n".join(f"+{line}" for line in block.content.splitlines())
        
        with open(filepath, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        diff = difflib.unified_diff(
            existing_content.splitlines(keepends=True),
            block.content.splitlines(keepends=True),
            fromfile=str(filepath),
            tofile=f"{filepath} (new)",
            lineterm=""
        )
        
        return "".join(diff)


def main():
    parser = argparse.ArgumentParser(description="Apply code blocks from AI responses to files")
    parser.add_argument("input", nargs="?", help="Input file containing AI response (default: stdin)")
    parser.add_argument("--base-dir", "-d", default=".", help="Base directory for file operations")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups of existing files")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--show-diff", action="store_true", help="Show diffs instead of applying changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    # Read input
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    
    if not text.strip():
        print("No input provided", file=sys.stderr)
        return 1
    
    # Parse code blocks
    parser_obj = CodeBlockParser()
    blocks = parser_obj.parse(text)
    
    if not blocks:
        if args.verbose:
            print("No code blocks with file paths found in input")
        return 0
    
    if args.verbose:
        print(f"Found {len(blocks)} code blocks:")
        for block in blocks:
            print(f"  - {block}")
        print()
    
    # Apply or show diffs
    applier = CodeBlockApplier(
        base_dir=args.base_dir,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )
    
    if args.show_diff:
        # Show diffs
        for block in blocks:
            print(f"=== {block.filepath} ===")
            print(applier.show_diff(block))
            print()
    else:
        # Apply changes
        results = applier.apply_blocks(blocks)
        
        if args.format == "json":
            print(json.dumps({
                "applied_files": applier.applied_files,
                "results": results
            }, indent=2))
        else:
            for filepath, result in results.items():
                print(result)
            
            if args.verbose and applier.applied_files:
                print(f"\nApplied {len(applier.applied_files)} files successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())