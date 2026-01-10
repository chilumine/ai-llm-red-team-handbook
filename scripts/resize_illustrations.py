
import os
import re

DOCS_DIR = 'docs'
TARGET_WIDTH = "512"

def should_process_file(filename):
    if not filename.endswith('.md'):
        return False
    match = re.search(r'Chapter_(\d+)', filename)
    if match:
        num = int(match.group(1))
        return 25 <= num <= 46
    return False

def process_content(content):
    lines = content.split('\n')
    new_lines = []
    i = 0
    modified = False
    
    while i < len(lines):
        line = lines[i]
        
        # Pattern 1: Markdown images ![alt](src)
        # Regex to capture ![alt](src)
        # Note: This is a simple regex and might fail on nested brackets, but standard markdown images usually work.
        md_img_match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
        
        if md_img_match:
            alt_text = md_img_match.group(1)
            src = md_img_match.group(2)
            
            # Special handling for page headers
            if 'page_header.svg' in src:
                alt_text = ""
                
            # Construct replacement HTML
            # Check if line is just the image or has valid surrounding text.
            # Usually images are on their own paragraph in this handbook.
            
            replacement = f'<p align="center">\n  <img src="{src}" alt="{alt_text}" width="{TARGET_WIDTH}">\n</p>'
            new_lines.append(replacement)
            modified = True
            i += 1
            continue

        # Pattern 2: Existing HTML <img> tags
        # We need to ensure they have width=768 and are centered.
        # This is harder to parse line-by-line if spanning multiple lines, but let's look for <img ...>
        
        if '<img' in line:
            # Check if it's already in a centered p-tag in the previous line?
            # Or just handle the img tag itself and wrap it if not wrapped?
            # Current file structure seems to use:
            # <p align="center">
            #   <img ...>
            # </p>
            
            # Let's simple regex replace the <img ...> tag attributes
            def replace_img_tag(match):
                img_tag = match.group(0)
                
                # Extract src and alt
                src_match = re.search(r'src=["\'](.*?)["\']', img_tag)
                alt_match = re.search(r'alt=["\'](.*?)["\']', img_tag)
                
                src = src_match.group(1) if src_match else ""
                alt = alt_match.group(1) if alt_match else ""
                
                if 'page_header.svg' in src:
                    alt = ""
                    
                # Reconstruct
                return f'<img src="{src}" alt="{alt}" width="{TARGET_WIDTH}">'

            new_line = re.sub(r'<img[^>]+>', replace_img_tag, line)
            
            if new_line != line:
                modified = True
                line = new_line
            
            # Now checking for wrapping is tricky without context buffers.
            # Assuming for now that if we modifying specific chapters, we generally want to enforce the p-align wrapper
            # But changing structure might be risky if already wrapped.
            # Let's stick to modifying the <img> tag width for now, as that's the primary request.
            # The Markdown conversion (Pattern 1) handles the wrapping for those.
            # Existing HTML tags in these chapters (like Ch 25) seem to be already wrapped.
            
            new_lines.append(line)
            i += 1
            continue

        new_lines.append(line)
        i += 1

    return '\n'.join(new_lines) if modified else None

def main():
    count = 0
    for filename in sorted(os.listdir(DOCS_DIR)):
        if should_process_file(filename):
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, 'r') as f:
                content = f.read()
            
            new_content = process_content(content)
            
            if new_content:
                with open(filepath, 'w') as f:
                    f.write(new_content)
                print(f"Updated {filename}")
                count += 1
    
    print(f"Modified {count} files.")

if __name__ == '__main__':
    main()
