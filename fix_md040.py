#!/usr/bin/env python3
"""
Fix MD040 linting errors by adding language identifiers to bare code blocks.
Specifically designed for Chapter 14: Prompt Injection
"""

import re
import sys

def detect_code_block_language(content):
    """
    Heuristically determine the language of a code block based on its content.
    """
    lines = content.strip().split('\n')
    
    # Check for Python indicators
    if any(keyword in content for keyword in ['def ', 'import ', 'class ', 'print(', 'return ', '    #', 'if __name__']):
        return 'python'
    
    # Check for bash/shell indicators  
    if any(keyword in content for keyword in ['#!/bin/', 'echo ', 'curl ', 'grep ', 'sed ', 'awk ', '$ ', 'export ']):
        return 'bash'
    
    # Check for JSON indicators
    if content.strip().startswith('{') and content.strip().endswith('}'):
        if '"' in content and ':' in content:
            return 'json'
    
    # Check for YAML indicators
    if re.search(r'^\w+:\s*$', content, re.MULTILINE) and not any(c in content for c in ['{', '(', 'def ']):
        return 'yaml'
    
    # Check for markdown content
    if '## ' in content or '### ' in content or '- [ ]' in content:
        return 'markdown'
    
    # Check for prompts/examples (conversation format)
    if any(indicator in content.lower() for indicator in ['user:', 'assistant:', 'system:', 'prompt:', 'response:']):
        return 'text'
    
    # Check for bullet points or list-like content
    if content.count('\n- ') > 2 or content.count('\n* ') > 2:
        return 'text'
    
    # Check for HTTP/API content  
    if any(keyword in content for keyword in ['HTTP/', 'GET ', 'POST ', 'Content-Type:', 'Authorization:']):
        return 'http'
        
    # Default to text for plain examples
    return 'text'

def fix_code_blocks(filepath):
    """
    Read a markdown file and add language identifiers to bare code blocks.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match bare code blocks (``` without language identifier)
    # This matches ``` at start of line, optionally followed by newline
    pattern = r'^```\s*$'
    
    lines = content.split('\n')
    in_code_block = False
    code_block_start = -1
    code_block_lines = []
    fixed_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a code fence
        if re.match(r'^```+', line):
            # Check if it's a bare fence (no language specified)
            match = re.match(r'^(```+)(\s*)$', line)
            
            if match and not in_code_block:
                # Start of a bare code block
                in_code_block = True
                code_block_start = i
                code_block_lines = []
                i += 1
                continue
            elif match and in_code_block:
                # End of code block - analyze and fix
                code_content = '\n'.join(code_block_lines)
                language = detect_code_block_language(code_content)
                
                # Replace the opening fence with language identifier
                lines[code_block_start] = f'```{language}'
                fixed_count += 1
                
                in_code_block = False
                code_block_start = -1
                code_block_lines = []
        elif in_code_block:
            code_block_lines.append(line)
        
        i += 1
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return fixed_count

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_md040.py <markdown_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    count = fix_code_blocks(filepath)
    print(f"Fixed {count} bare code blocks in {filepath}")
