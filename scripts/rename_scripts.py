#!/usr/bin/env python3
"""
Rename script files from chapter-based names to descriptive functional names.

Analyzes each Python file's docstring and content to generate appropriate descriptive names.
"""

import re
from pathlib import Path
from typing import Dict, Tuple
import json


def extract_description(file_path: Path) -> str:
    """Extract description from file's docstring."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Try to find docstring in first 50 lines
        lines = content.split('\n')[:50]
        in_docstring = False
        description_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if '"""' in stripped or "'''" in stripped:
                if in_docstring:
                    break
                in_docstring = True
                # Get text after opening quotes
                text = stripped.replace('"""', '').replace("'''", '').strip()
                if text and text != '#':
                    description_lines.append(text)
            elif in_docstring:
                if stripped and not stripped.startswith('#'):
                    description_lines.append(stripped)
        
        description = ' '.join(description_lines)
        
        # Clean up common prefixes
        description = re.sub(r'^Source:.*?Category:.*?$', '', description, flags=re.MULTILINE)
        description = description.replace('  ', ' ').strip()
        
        return description[:200]  # Limit length
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def generate_descriptive_name(description: str, category: str) -> str:
    """Generate a descriptive filename from description."""
    
    # Extract key action words and subjects
    description_lower = description.lower()
    
    # Common patterns to descriptive names
    patterns = {
        'inspect': ['inspect', 'inspecting', 'view', 'check'],
        'tokeniz': ['tokenizer', 'tokenization', 'token'],
        'extract': ['extract', 'extraction', 'exfiltrat'],
        'inject': ['inject', 'injection'],
        'bypass': ['bypass', 'evade', 'circumvent'],
        'jailbreak': ['jailbreak', 'dan', 'guardrail'],
        'test': ['test', 'testing'],
        'attack': ['attack', 'exploit'],
        'poison': ['poison', 'poisoning'],
        'steal': ['steal', 'theft'],
        'leak': ['leak', 'leakage'],
        'prompt': ['prompt'],
        'rag': ['rag', 'retrieval', 'vector'],
        'plugin': ['plugin', 'api', 'function'],
        'model': ['model loading', 'load model'],
        'fuzzer': ['fuzz', 'fuzzing'],
    }
    
    # Find matching patterns
    action = None
    subject = None
    
    for key, keywords in patterns.items():
        for keyword in keywords:
            if keyword in description_lower:
                action = key
                break
        if action:
            break
    
    # Extract subject
    if 'tokenizer' in description_lower or 'token' in description_lower:
        subject = 'tokenizer'
    elif 'rag' in description_lower or 'vector' in description_lower:
        subject = 'rag'
    elif 'plugin' in description_lower:
        subject = 'plugin'
    elif 'prompt' in description_lower:
        subject = 'prompt'
    elif 'model' in description_lower:
        subject = 'model'
    elif 'api' in description_lower:
        subject = 'api'
    
    # Build name
    parts = []
    if action:
        parts.append(action)
    if subject and subject not in (action or ''):
        parts.append(subject)
    
    if not parts:
        # Fallback: use first meaningful words
        words = re.findall(r'\b\w{4,}\b', description_lower)
        parts = words[:2] if len(words) >= 2 else words[:1]
    
    # Ensure we have something
    if not parts:
        parts = ['script']
    
    name = '_'.join(parts)
    
    # Clean up
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    
    return name


def rename_file(file_path: Path, dry_run: bool = True) -> Tuple[Path, Path]:
    """
    Rename a single file to a descriptive name.
    
    Returns (old_path, new_path)
    """
    # Extract category from parent directory
    category = file_path.parent.name
    
    # Get description from file
    description = extract_description(file_path)
    
    # Generate new name
    base_name = generate_descriptive_name(description, category)
    
    # Ensure uniqueness
    new_path = file_path.parent / f"{base_name}.py"
    counter = 2
    while new_path.exists() and new_path != file_path:
        new_path = file_path.parent / f"{base_name}_{counter}.py"
        counter += 1
    
    if not dry_run and new_path != file_path:
        file_path.rename(new_path)
    
    return (file_path, new_path)


def main():
    """Main renaming process."""
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Show renaming plan without executing')
    parser.add_argument('--execute', action='store_true', help='Execute the renaming')
    parser.add_argument('--category', help='Only process files in this category')
    args = parser.parse_args()
    
    base_dir = Path('/home/e/Desktop/ai-llm-red-team-handbook/scripts')
    
    # Find all chapter-based files
    chapter_files = list(base_dir.glob('**/chapter_*.py'))
    
    print(f"Found {len(chapter_files)} files to rename")
    
    if args.category:
        chapter_files = [f for f in chapter_files if f.parent.name == args.category]
        print(f"Filtered to {len(chapter_files)} files in category: {args.category}")
    
    # Build renaming plan
    rename_plan = []
    
    for file_path in sorted(chapter_files):
        old, new = rename_file(file_path, dry_run=True)
        if old != new:
            rename_plan.append((old, new))
    
    print(f"\nRenaming plan ({len(rename_plan)} files):")
    print("="*80)
    
    for old, new in rename_plan[:20]:  # Show first 20
        print(f"{old.parent.name}/{old.name:60s} -> {new.name}")
    
    if len(rename_plan) > 20:
        print(f"... and {len(rename_plan) - 20} more files")
    
    # Save plan to JSON
    plan_file = base_dir / 'rename_plan.json'
    plan_data = {
        'total_files': len(rename_plan),
        'renames': [
            {
                'old': str(old.relative_to(base_dir)),
                'new': str(new.relative_to(base_dir)),
                'category': old.parent.name
            }
            for old, new in rename_plan
        ]
    }
    plan_file.write_text(json.dumps(plan_data, indent=2))
    print(f"\nFull plan saved to: {plan_file}")
    
    # Execute if requested
    if args.execute and not args.dry_run:
        print("\n" + "="*80)
        print("EXECUTING RENAMES...")
        print("="*80 + "\n")
        
        success_count = 0
        error_count = 0
        
        for old_path, new_path in rename_plan:
            try:
                old_path.rename(new_path)
                success_count += 1
                print(f"✓ {old_path.name} -> {new_path.name}")
            except Exception as e:
                error_count += 1
                print(f"✗ {old_path.name}: {e}")
        
        print(f"\n{success_count} files renamed successfully")
        if error_count:
            print(f"{error_count} files failed")
    else:
        print("\nRun with --execute to perform the renaming")
        print("Example: python3 rename_scripts.py --execute")


if __name__ == '__main__':
    main()
