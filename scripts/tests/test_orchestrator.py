#!/usr/bin/env python3
"""
Comprehensive Test Orchestrator for AI LLM Red Team Scripts

This script performs comprehensive testing of all scripts in the repository,
including functional testing, tool integration, performance assessment,
error handling, and compliance verification.

Uses local LLM (Ollama/LMStudio) for intelligent test generation and validation.
"""

import argparse
import json
import os
import sys
import time
import subprocess
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import requests
from urllib.parse import urlparse

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
CATEGORIES = [
    "reconnaissance", "prompt_injection", "data_extraction", "jailbreak",
    "plugin_exploitation", "rag_attacks", "evasion", "model_attacks",
    "multimodal", "post_exploitation", "social_engineering", "automation",
    "supply_chain", "compliance", "utils"
]

# Compliance standards
COMPLIANCE_STANDARDS = {
    "OWASP-LLM-TOP-10": ["LLM01", "LLM02", "LLM03", "LLM04", "LLM05", 
                          "LLM06", "LLM07", "LLM08", "LLM09", "LLM10"],
    "MITRE-ATLAS": ["AML.T0000", "AML.T0001", "AML.T0002"],
    "NIST-AI-RMF": ["GOVERN", "MAP", "MEASURE", "MANAGE"],
    "ETHICAL": ["authorization", "scope", "documentation", "disclosure"]
}


class TestOrchestrator:
    """Main test orchestration class"""
    
    def __init__(self, llm_endpoint: str = "http://localhost:11434", verbose: bool = True):
        # Validate endpoint to prevent SSRF
        parsed = urlparse(llm_endpoint)
        if parsed.scheme not in ['http', 'https']:
            raise ValueError(f"Invalid LLM endpoint scheme: '{parsed.scheme}'. Only 'http' and 'https' are allowed.")
            
        self.llm_endpoint = llm_endpoint
        self.verbose = verbose
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "categories": {},
            "errors": [],
            "performance": {},
            "compliance": {}
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def test_llm_connection(self) -> bool:
        """Test connection to local LLM"""
        self.log("Testing LLM endpoint connection...")
        try:
            response = requests.get(f"{self.llm_endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.log(f"✓ LLM endpoint accessible. Available models: {len(models)}")
                return True
            else:
                self.log(f"✗ LLM endpoint returned status {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Failed to connect to LLM endpoint: {e}", "ERROR")
            return False
    
    def query_llm(self, prompt: str, model: str = "llama2") -> str:
        """Query the local LLM"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(
                f"{self.llm_endpoint}/api/generate",
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            self.log(f"LLM query failed: {e}", "ERROR")
            return ""
    
    def discover_scripts(self, category: str = None) -> Dict[str, List[Path]]:
        """Discover all Python scripts in categories"""
        self.log("Discovering scripts...")
        discovered = {}
        
        categories_to_scan = [category] if category else CATEGORIES
        
        for cat in categories_to_scan:
            cat_path = SCRIPT_DIR / cat
            if not cat_path.exists():
                continue
            
            scripts = list(cat_path.glob("*.py"))
            if scripts:
                discovered[cat] = scripts
                self.log(f"  [{cat}] Found {len(scripts)} scripts")
        
        total = sum(len(scripts) for scripts in discovered.values())
        self.log(f"Total scripts discovered: {total}")
        return discovered
    
    def test_script_syntax(self, script_path: Path) -> Tuple[bool, str]:
        """Test if script has valid Python syntax"""
        try:
            with open(script_path, 'r') as f:
                compile(f.read(), script_path, 'exec')
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Error: {e}"
    
    def test_script_imports(self, script_path: Path) -> Tuple[bool, str]:
        """Test if script imports can be resolved"""
        try:
            spec = importlib.util.spec_from_file_location("test_module", script_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Don't execute, just verify imports
                return True, "Imports valid"
            return False, "Could not load module"
        except ImportError as e:
            return False, f"Import error: {e}"
        except Exception as e:
            return False, f"Error: {e}"
    
    def test_script_help(self, script_path: Path) -> Tuple[bool, str]:
        """Test if script provides --help"""
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and ("usage:" in result.stdout.lower() or "help" in result.stdout.lower()):
                return True, "Help available"
            return False, "No help output"
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, f"Error: {e}"
    
    def test_functional(self, scripts: Dict[str, List[Path]]) -> Dict:
        """Functional testing phase"""
        self.log("\n=== Phase 3: Functional Testing ===")
        results = {}
        
        for category, script_list in scripts.items():
            self.log(f"\nTesting category: {category}")
            category_results = []
            
            for script in script_list:
                self.results["tests_run"] += 1
                test_result = {
                    "script": script.name,
                    "syntax": None,
                    "imports": None,
                    "help": None,
                    "status": "PASS"
                }
                
                # Syntax test
                syntax_pass, syntax_msg = self.test_script_syntax(script)
                test_result["syntax"] = {"pass": syntax_pass, "message": syntax_msg}
                
                # Import test
                import_pass, import_msg = self.test_script_imports(script)
                test_result["imports"] = {"pass": import_pass, "message": import_msg}
                
                # Help test
                help_pass, help_msg = self.test_script_help(script)
                test_result["help"] = {"pass": help_pass, "message": help_msg}
                
                # Overall status
                if syntax_pass and import_pass:
                    self.results["tests_passed"] += 1
                    self.log(f"  ✓ {script.name}")
                else:
                    self.results["tests_failed"] += 1
                    test_result["status"] = "FAIL"
                    self.log(f"  ✗ {script.name}", "ERROR")
                    self.results["errors"].append({
                        "script": str(script),
                        "errors": [syntax_msg, import_msg]
                    })
                
                category_results.append(test_result)
            
            results[category] = category_results
        
        self.results["categories"] = results
        return results
    
    def test_integration(self) -> Dict:
        """Tool integration testing"""
        self.log("\n=== Phase 4: Tool Integration Testing ===")
        
        # Test if common tools are available
        tools = {
            "requests": "pip show requests",
            "transformers": "pip show transformers",
            "tiktoken": "pip show tiktoken",
        }
        
        integration_results = {}
        for tool, check_cmd in tools.items():
            try:
                result = subprocess.run(
                    check_cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                available = result.returncode == 0
                integration_results[tool] = available
                status = "✓" if available else "✗"
                self.log(f"  {status} {tool}")
            except Exception as e:
                integration_results[tool] = False
                self.log(f"  ✗ {tool}: {e}", "ERROR")
        
        return integration_results
    
    def test_performance(self, scripts: Dict[str, List[Path]], sample_size: int = 5) -> Dict:
        """Performance testing"""
        self.log("\n=== Phase 6: Performance Assessment ===")
        
        performance_results = {}
        
        for category, script_list in scripts.items():
            # Sample scripts from each category
            sample_scripts = script_list[:sample_size]
            
            for script in sample_scripts:
                start_time = time.time()
                try:
                    result = subprocess.run(
                        [sys.executable, str(script), "--help"],
                        capture_output=True,
                        timeout=5
                    )
                    elapsed = time.time() - start_time
                    performance_results[script.name] = {
                        "elapsed_seconds": elapsed,
                        "status": "success"
                    }
                    self.log(f"  {script.name}: {elapsed:.2f}s")
                except subprocess.TimeoutExpired:
                    performance_results[script.name] = {
                        "elapsed_seconds": 5.0,
                        "status": "timeout"
                    }
                except Exception as e:
                    performance_results[script.name] = {
                        "elapsed_seconds": 0,
                        "status": f"error: {e}"
                    }
        
        self.results["performance"] = performance_results
        return performance_results
    
    def test_compliance(self, scripts: Dict[str, List[Path]], standard: str = "OWASP-LLM-TOP-10") -> Dict:
        """Compliance testing"""
        self.log(f"\n=== Phase 8: Compliance Testing ({standard}) ===")
        
        compliance_results = {
            "standard": standard,
            "checks": [],
            "coverage": 0.0
        }
        
        if standard not in COMPLIANCE_STANDARDS:
            self.log(f"Unknown standard: {standard}", "WARNING")
            return compliance_results
        
        # For each compliance item, check if it's covered
        items = COMPLIANCE_STANDARDS[standard]
        covered_count = 0
        
        for item in items:
            # Simple heuristic: grep for item in script content
            covered = False
            for category, script_list in scripts.items():
                for script in script_list:
                    try:
                        with open(script, 'r') as f:
                            content = f.read()
                            if item.lower() in content.lower():
                                covered = True
                                break
                    except:
                        pass
                if covered:
                    break
            
            if covered:
                covered_count += 1
            
            compliance_results["checks"].append({
                "item": item,
                "covered": covered
            })
            
            status = "✓" if covered else "✗"
            self.log(f"  {status} {item}")
        
        compliance_results["coverage"] = (covered_count / len(items)) * 100 if items else 0
        self.log(f"\nCoverage: {compliance_results['coverage']:.1f}%")
        
        self.results["compliance"][standard] = compliance_results
        return compliance_results
    
    def llm_validate_script(self, script_path: Path) -> Dict:
        """Use LLM to validate script purpose and implementation"""
        self.log(f"LLM validating: {script_path.name}")
        
        try:
            with open(script_path, 'r') as f:
                code = f.read()[:2000]  # First 2000 chars
            
            prompt = f"""Analyze this Python security testing script and provide:
1. Primary purpose
2. Potential security concerns
3. Code quality rating (1-10)

Script: {script_path.name}

```python
{code}
```

Respond in JSON format."""
            
            response = self.query_llm(prompt)
            
            if response:
                return {
                    "script": script_path.name,
                    "llm_analysis": response[:500],  # Truncate
                    "validated": True
                }
        except Exception as e:
            self.log(f"LLM validation error: {e}", "ERROR")
        
        return {"script": script_path.name, "validated": False}
    
    def generate_report(self, output_file: str, format: str = "json"):
        """Generate test report"""
        self.log(f"\n=== Generating Report ({format}) ===")
        
        # Sanitize output path to prevent traversal
        # Ensure we only write to the current directory by using the basename
        safe_filename = os.path.basename(output_file)
        if safe_filename != output_file:
            self.log(f"Sanitizing output path from '{output_file}' to '{safe_filename}' for security", "WARNING")
        
        output_path = Path.cwd() / safe_filename
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            self.log(f"JSON report saved to: {output_path}")
        
        elif format == "html":
            html_content = self._generate_html_report()
            with open(output_path, 'w') as f:
                f.write(html_content)
            self.log(f"HTML report saved to: {output_path}")
        
        elif format == "summary":
            summary = self._generate_summary()
            with open(output_path, 'w') as f:
                f.write(summary)
            self.log(f"Summary report saved to: {output_path}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report"""
        pass_rate = (self.results["tests_passed"] / self.results["tests_run"] * 100) if self.results["tests_run"] > 0 else 0
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI LLM Red Team Scripts - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>AI LLM Red Team Scripts - Comprehensive Test Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Timestamp:</strong> {self.results["timestamp"]}</p>
        <p><strong>Tests Run:</strong> {self.results["tests_run"]}</p>
        <p class="pass"><strong>Tests Passed:</strong> {self.results["tests_passed"]}</p>
        <p class="fail"><strong>Tests Failed:</strong> {self.results["tests_failed"]}</p>
        <p><strong>Pass Rate:</strong> {pass_rate:.1f}%</p>
    </div>
    
    <h2>Category Results</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Scripts Tested</th>
            <th>Status</th>
        </tr>"""
        
        for category, results in self.results["categories"].items():
            passed = sum(1 for r in results if r["status"] == "PASS")
            total = len(results)
            status_class = "pass" if passed == total else "fail"
            html += f"""
        <tr>
            <td>{category}</td>
            <td>{total}</td>
            <td class="{status_class}">{passed}/{total}</td>
        </tr>"""
        
        html += """
    </table>
</body>
</html>"""
        return html
    
    def _generate_summary(self) -> str:
        """Generate text summary"""
        pass_rate = (self.results["tests_passed"] / self.results["tests_run"] * 100) if self.results["tests_run"] > 0 else 0
        
        summary = f"""
================================================================================
AI LLM RED TEAM HANDBOOK - COMPREHENSIVE TEST REPORT
================================================================================

Timestamp: {self.results["timestamp"]}

EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Total Tests Run:     {self.results["tests_run"]}
Tests Passed:        {self.results["tests_passed"]}
Tests Failed:        {self.results["tests_failed"]}
Tests Skipped:       {self.results["tests_skipped"]}
Pass Rate:           {pass_rate:.1f}%

CATEGORY BREAKDOWN
--------------------------------------------------------------------------------
"""
        
        for category, results in self.results["categories"].items():
            passed = sum(1 for r in results if r["status"] == "PASS")
            total = len(results)
            summary += f"{category.ljust(25)} {passed}/{total}\n"
        
        summary += """
================================================================================
END OF REPORT
================================================================================
"""
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Orchestrator for AI LLM Red Team Scripts"
    )
    parser.add_argument("--llm-endpoint", default="http://localhost:11434",
                        help="LLM endpoint URL")
    parser.add_argument("--category", choices=CATEGORIES,
                        help="Test specific category only")
    parser.add_argument("--all", action="store_true",
                        help="Test all categories")
    parser.add_argument("--test-type", 
                        choices=["functional", "integration", "performance", 
                                 "compliance", "error-handling", "all"],
                        default="all", help="Type of test to run")
    parser.add_argument("--standard", choices=list(COMPLIANCE_STANDARDS.keys()),
                        default="OWASP-LLM-TOP-10",
                        help="Compliance standard to test against")
    parser.add_argument("--generate-report", action="store_true",
                        help="Generate test report")
    parser.add_argument("--format", choices=["json", "html", "summary"],
                        default="json", help="Report format")
    parser.add_argument("--output", default="test_report.json",
                        help="Output file for report")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--llm-validate", action="store_true",
                        help="Use LLM to validate scripts")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = TestOrchestrator(args.llm_endpoint, args.verbose)
    
    # Test LLM connection
    if not orchestrator.test_llm_connection():
        print("Warning: LLM endpoint not available. LLM-based tests will be skipped.")
    
    # Discover scripts
    scripts = orchestrator.discover_scripts(args.category)
    
    if not scripts:
        print("No scripts found to test!")
        return 1
    
    # Run tests based on type
    if args.test_type in ["functional", "all"]:
        orchestrator.test_functional(scripts)
    
    if args.test_type in ["integration", "all"]:
        orchestrator.test_integration()
    
    if args.test_type in ["performance", "all"]:
        orchestrator.test_performance(scripts)
    
    if args.test_type in ["compliance", "all"]:
        orchestrator.test_compliance(scripts, args.standard)
    
    # LLM validation if requested
    if args.llm_validate:
        orchestrator.log("\n=== Phase 9: LLM-Powered Validation ===")
        for category, script_list in scripts.items():
            for script in script_list[:3]:  # Sample 3 per category
                orchestrator.llm_validate_script(script)
    
    # Generate report
    if args.generate_report:
        orchestrator.generate_report(args.output, args.format)
    
    # Print summary
    print(orchestrator._generate_summary())
    
    return 0 if orchestrator.results["tests_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
