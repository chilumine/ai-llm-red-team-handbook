#!/usr/bin/env python3
"""
Extract all code blocks from handbook chapters.
Creates a catalog JSON file mapping chapters to code blocks.
"""

import re
import json
from pathlib import Path
from typing import List, Dict

def extract_code_blocks(markdown_file: Path) -> List[Dict]:
    """Extract all code blocks with metadata from a markdown file."""
    try:
        content = markdown_file.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {markdown_file}: {e}")
        return []
    
    # Pattern to match fenced code blocks
    pattern = r'```(\w+)\n(.*?)```'
    
    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL):
        language = match.group(1)
        code = match.group(2)
        
        if language in ['python', 'bash', 'sh']:
            # Get surrounding context (200 chars before and after)
            start = max(0, match.start() - 200)
            end = min(len(content), match.end() + 200)
            context = content[start:end]
            
            # Extract section header if available
            section_pattern = r'##\s+([^\n]+)'
            sections = re.findall(section_pattern, content[:match.start()])
            current_section = sections[-1] if sections else "Unknown"
            
            blocks.append({
                'language': language,
                'code': code.strip(),
                'context': context,
                'section': current_section,
                'line_number': content[:match.start()].count('\n') + 1,
                'length': len(code.strip().split('\n'))
            })
    
    return blocks

def catalog_all_code():
    """Scan all chapters and create catalog."""
    base_dir = Path('/home/e/Desktop/ai-llm-red-team-handbook')
    docs_dir = base_dir / 'docs'
    catalog = {}
    
    # Get all chapter files
    chapter_files = sorted(docs_dir.glob('Chapter_*.md'))
    
    print(f"Found {len(chapter_files)} chapter files")
    
    for chapter_file in chapter_files:
        chapter_name = chapter_file.stem
        print(f"Processing {chapter_name}...")
        
        blocks = extract_code_blocks(chapter_file)
        
        if blocks:
            catalog[chapter_name] = {
                'file': str(chapter_file),
                'python_blocks': len([b for b in blocks if b['language'] == 'python']),
                'bash_blocks': len([b for b in blocks if b['language'] in ['bash', 'sh']]),
                'total_blocks': len(blocks),
                'blocks': blocks
            }
            print(f"  Found {len(blocks)} code blocks ({catalog[chapter_name]['python_blocks']} Python, {catalog[chapter_name]['bash_blocks']} Bash)")
    
    # Save catalog
    catalog_file = base_dir / 'scripts' / 'code_catalog.json'
    catalog_file.parent.mkdir(exist_ok=True)
    catalog_file.write_text(json.dumps(catalog, indent=2))
    
    # Print summary
    total_python = sum(d['python_blocks'] for d in catalog.values())
    total_bash = sum(d['bash_blocks'] for d in catalog.values())
    total_blocks = sum(d['total_blocks'] for d in catalog.values())
    
    print(f"\n{'='*60}")
    print(f"Catalog saved to {catalog_file}")
    print(f"Total chapters with code: {len(catalog)}")
    print(f"Total code blocks: {total_blocks}")
    print(f"  - Python: {total_python}")
    print(f"  - Bash/sh: {total_bash}")
    print(f"{'='*60}")
    
    return catalog

if __name__ == '__main__':
    catalog_all_code()
