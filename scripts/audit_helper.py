
import re
import sys
import json
import time
import argparse
import requests
import concurrent.futures
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict, Any

def extract_links(markdown_content: str) -> List[str]:
    """Extract all markdown links [text](url) and <url>."""
    # Match [text](url)
    md_links = re.findall(r'\[.*?\]\((https?://[^)]+)\)', markdown_content)
    # Match <url>
    angle_links = re.findall(r'<(https?://[^>]+)>', markdown_content)
    # Match bare URLs (ignoring those already captured in () or <>)
    # This regex looks for http/https, but tries to avoid trailing punctuation often found in text
    plain_links = re.findall(r'https?://[^\s<>")]+', markdown_content)
    
    all_links = sorted(list(set(md_links + angle_links + plain_links)))
    
    # Remove any trailing punctuation that might have slipped in (like periods or commas)
    clean_links = []
    for link in all_links:
        link = link.rstrip('.,;:')
        clean_links.append(link)
        
    return sorted(list(set(clean_links)))

def check_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Check a single URL for reachability."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if response.status_code == 405: # Method Not Allowed, try GET
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.close()
        
        return {
            "url": url,
            "status": response.status_code,
            "alive": 200 <= response.status_code < 400
        }
    except requests.RequestException as e:
        return {
            "url": url,
            "status": -1,
            "alive": False,
            "error": str(e)
        }

def audit_chapter(file_path: str, output_file: str = None) -> Dict[str, Any]:
    """Audit a markdown chapter file."""
    start_time = time.time()
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    content = path.read_text(encoding='utf-8')
    
    # 1. Verification: Link Checking
    links = extract_links(content)
    print(f"Found {len(links)} unique remote URLs. Verifying...")
    
    url_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(check_url, url): url for url in links}
        for future in concurrent.futures.as_completed(future_to_url):
            url_results.append(future.result())
            
    broken_links = [r for r in url_results if not r['alive']]
    
    # 2. Fact/Formatting Check (Basic Regex)
    issues = []
    
    # Check for empty links
    if re.search(r'\[.*?\]\(\s*\)', content):
        issues.append({"type": "Formatting", "issue": "Found empty markdown link ()"})
        
    # Check for duplicate headers
    headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
    if len(headers) != len(set(headers)):
         issues.append({"type": "Structure", "issue": "Duplicate headers found"})

    total_duration = time.time() - start_time
    
    report = {
        "metadata": {
            "chapter": path.name,
            "audit_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(total_duration, 2),
            "version": "2.0"
        },
        "summary": {
            "total_links": len(links),
            "broken_links": len(broken_links)
        },
        "url_verification": convert_results_to_json_friendly(url_results),
        "structure_issues": issues
    }

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")
    else:
        print(json.dumps(report, indent=2))
        
    return report

def convert_results_to_json_friendly(results):
    # Ensure all data in results is JSON serializable if needed
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit MD Chapter")
    parser.add_argument("filename", help="Markdown file to audit")
    parser.add_argument("--output", help="JSON output file", default="audit_report.json")
    args = parser.parse_args()
    
    audit_chapter(args.filename, args.output)
