#!/usr/bin/env python3
"""
Generate organized, practical scripts from cataloged code blocks.
Creates proper directory structure and Python modules with CLI interfaces.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

# Category mappings based on keywords and chapter numbers
CATEGORY_MAPPINGS = {
    'utils': {
        'keywords': ['tokenizer', 'tokenization', 'encode', 'decode', 'tiktoken', 'import', 'helper', 'utility'],
        'chapters': ['Chapter_09', 'Chapter_10']
    },
    'reconnaissance': {
        'keywords': ['fingerprint', 'discover', 'enumerate', 'scan', 'reconnaissance', 'probe'],
        'chapters': ['Chapter_31']
    },
    'prompt_injection': {
        'keywords': ['injection', 'inject', 'ignore previous', 'system prompt leak', 'jailbreak injection'],
        'chapters': ['Chapter_14']
    },
    'data_extraction': {
        'keywords': ['extract', 'leak', 'exfiltration', 'pii', 'data leakage', 'memory dump'],
        'chapters': ['Chapter_15']
    },
    'jailbreak': {
        'keywords': ['jailbreak', 'bypass', 'character', 'roleplay', 'dan', 'bypass guardrail'],
        'chapters': ['Chapter_16']
    },
    'plugin_exploitation': {
        'keywords': ['plugin', 'function_call', 'tool use', 'api call', 'command injection'],
        'chapters': ['Chapter_11', 'Chapter_17']
    },
    'rag_attacks': {
        'keywords': ['rag', 'retrieval', 'vector', 'embedding', 'chromadb', 'faiss', 'poisoning'],
        'chapters': ['Chapter_12']
    },
    'evasion': {
        'keywords': ['obfuscate', 'evasion', 'encode', 'filter bypass', 'adversarial'],
        'chapters': ['Chapter_18', 'Chapter_34']
    },
    'model_attacks': {
        'keywords': ['model theft', 'extraction', 'membership inference', 'dos', 'resource exhaustion', 'adversarial example'],
        'chapters': ['Chapter_19', 'Chapter_20', 'Chapter_21', 'Chapter_25', 'Chapter_29', 'Chapter_30']
    },
    'automation': {
        'keywords': ['fuzzer', 'automation', 'orchestrat', 'framework', 'batch'],
        'chapters': ['Chapter_32', 'Chapter_33']
    },
    'labs': {
        'keywords': ['setup', 'install', 'configure', 'docker', 'environment'],
        'chapters': ['Chapter_07']
    },
    'supply_chain': {
        'keywords': ['supply chain', 'dependency', 'package', 'provenance'],
        'chapters': ['Chapter_13', 'Chapter_26']
    },
    'multimodal': {
        'keywords': ['image', 'audio', 'video', 'multimodal', 'cross-modal'],
        'chapters': ['Chapter_22']
    },
    'social_engineering': {
        'keywords': ['social engineering', 'phishing', 'manipulation', 'persuasion'],
        'chapters': ['Chapter_24']
    },
    'post_exploitation': {
        'keywords': ['persistence', 'post-exploitation', 'backdoor', 'maintain access'],
        'chapters': ['Chapter_23', 'Chapter_30', 'Chapter_35']
    },
    'compliance': {
        'keywords': ['compliance', 'standard', 'regulation', 'audit'],
        'chapters': ['Chapter_39', 'Chapter_40', 'Chapter_41']
    }
}

def classify_code_block(chapter_name: str, code: str, section: str) -> str:
    """Classify a code block into a category."""
    combined = (chapter_name + ' ' + code + ' ' + section).lower()
    
    # First try chapter-based classification  
    for category, mapping in CATEGORY_MAPPINGS.items():
        for chapter_prefix in mapping['chapters']:
            if chapter_name.startswith(chapter_prefix):
                return category
    
    # Then try keyword-based classification
    for category, mapping in CATEGORY_MAPPINGS.items():
        for keyword in mapping['keywords']:
            if keyword.lower() in combined:
                return category
    
    # Default to utils for generic code
    return 'utils'

def extract_imports(code: str) -> List[str]:
    """Extract import statements from Python code."""
    imports = []
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
    return imports

def extract_functions(code: str) -> List[str]:
    """Extract function definitions from Python code."""
    functions = []
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped.startswith('def '):
            functions.append(stripped)
    return functions

def clean_code(code: str) -> str:
    """Clean and format code."""
    # Remove excessive blank lines
    lines = code.split('\n')
    cleaned = []
    prev_blank = False
    
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank
    
    return '\n'.join(cleaned)

def create_directory_structure(base_path: Path):
    """Create the organized directory structure."""
    categories = list(CATEGORY_MAPPINGS.keys())
    categories.append('workflows')  # Add workflows category
    
    for category in categories:
        category_dir = base_path / category
        category_dir.mkdir(exist_ok=True, parents=True)
        
        # Create __init__.py for Python modules
        init_file = category_dir / '__init__.py'
        if not init_file.exists():
            init_file.write_text(f'"""{category.replace("_", " ").title()} module for AI LLM Red Teaming."""\n')
        
    print(f"Created directory structure with {len(categories)} categories")

def generate_script(block_data: Dict, category: str, chapter: str, index: int) -> tuple:
    """Generate a complete, runnable script from a code block."""
    code = block_data['code']
    section = block_data.get('section', 'Unknown')
    language = block_data['language']
    
    if language == 'bash' or language == 'sh':
        # Generate bash script
        script_content = f'''#!/bin/bash
# AI LLM Red Team - {section}
# Source: {chapter}
# Category: {category}

{code}
'''
        filename = f"{chapter.lower()}_{index:02d}.sh"
        return filename, script_content
    
    # Python script
    imports = extract_imports(code)
    functions = extract_functions(code)
    
    # Remove imports from main code
    code_lines = []
    for line in code.split('\n'):
        stripped = line.strip()
        if not (stripped.startswith('import ') or stripped.startswith('from ')):
            code_lines.append(line)
    
    main_code = '\n'.join(code_lines).strip()
    
    # Build script
    script_parts = [
        '#!/usr/bin/env python3',
        '"""',
        f'{section}',
        '',
        f'Source: {chapter}',
        f'Category: {category}',
        '"""',
        '',
    ]
    
    # Add imports
    if imports:
        script_parts.extend(imports)
        script_parts.append('')
    
    # Add standard imports for CLI
    if 'argparse' not in ' '.join(imports):
        script_parts.extend([
            'import argparse',
            'import sys',
            ''
        ])
    
    # Add main code
    script_parts.append(clean_code(main_code))
    script_parts.append('')
    
    # Add main function if not present
    if 'if __name__' not in main_code:
        script_parts.extend([
            '',
            'def main():',
            '    """Command-line interface."""',
            '    parser = argparse.ArgumentParser(description=__doc__)',
            '    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")',
            '    args = parser.parse_args()',
            '    ',
            '    # TODO: Add main execution logic',
            '    pass',
            '',
            'if __name__ == "__main__":',
            '    main()',
        ])
    
    script_content = '\n'.join(script_parts)
    filename = f"{chapter.lower()}_{index:02d}_{category}.py"
    
    return filename, script_content

def main():
    """Generate all scripts from catalog."""
    base_path = Path('/home/e/Desktop/ai-llm-red-team-handbook/scripts')
    catalog_file = base_path / 'code_catalog.json'
    
    # Load catalog
    catalog = json.loads(catalog_file.read_text())
    
    print(f"Loaded catalog with {len(catalog)} chapters")
    
    # Create directory structure
    create_directory_structure(base_path)
    
    # Track statistics
    stats = defaultdict(int)
    generated_files = []
    
    # Process each chapter
    for chapter_name, chapter_data in sorted(catalog.items()):
        print(f"\nProcessing {chapter_name}...")
        
        blocks = chapter_data['blocks']
        
        for idx, block in enumerate(blocks, 1):
            # Classify the block
            category = classify_code_block(chapter_name, block['code'], block.get('section', ''))
            
            # Generate script
            filename, script_content = generate_script(block, category, chapter_name, idx)
            
            # Write to appropriate category folder
            category_dir = base_path / category
            output_file = category_dir / filename
            
            # Only write if file doesn't exist (avoid overwriting custom edits)
            if not output_file.exists():
                output_file.write_text(script_content)
                print(f"  Created: {category}/{filename}")
                stats[category] += 1
                generated_files.append(str(output_file.relative_to(base_path)))
            else:
                print(f"  Skipped (exists): {category}/{filename}")
    
    # Generate summary
    print(f"\n{'='*60}")
    print(f"Script Generation Complete")
    print(f"{'='*60}")
    print(f"Total scripts generated: {sum(stats.values())}")
    print(f"\nScripts by category:")
    for category in sorted(stats.keys()):
        print(f"  {category:25s}: {stats[category]:3d} scripts")
    
    # Save file list
    filelist = base_path / 'generated_scripts.txt'
    filelist.write_text('\n'.join(sorted(generated_files)))
    print(f"\nFile list saved to: generated_scripts.txt")
    
    return stats

if __name__ == '__main__':
    main()
