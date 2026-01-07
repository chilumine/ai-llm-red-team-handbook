#!/usr/bin/env python3
"""
Full LLM Security Assessment Workflow

This workflow orchestrates a comprehensive security assessment combining
reconnaissance, prompt injection, data extraction, and exploitation techniques.

Usage:
    python3 workflows/full_assessment.py --target https://api.example.com --output report.json
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class AssessmentOrchestrator:
    """Orchestrates a full LLM security assessment."""
    
    def __init__(self, target, output_file=None, verbose=False):
        self.target = target
        self.output_file = output_file
        self.verbose = verbose
        self.results = {
            'target': target,
            'timestamp': datetime.now().isoformat(),
            'phases': {}
        }
    
    def log(self, message):
        """Log message if verbose."""
        if self.verbose:
            print(f"[*] {message}")
    
    def run_phase(self, phase_name, description, scripts):
        """Run a phase of the assessment."""
        print(f"\n{'='*60}")
        print(f"Phase: {phase_name}")
        print(f"Description: {description}")
        print(f"{'='*60}\n")
        
        phase_results = []
        
        for script_name, script_desc in scripts:
            self.log(f"Running: {script_name}")
            self.log(f"Purpose: {script_desc}")
            
            # TODO: Actually execute the scripts
            # For now, just record them
            phase_results.append({
                'script': script_name,
                'description': script_desc,
                'status': 'planned'
            })
        
        self.results['phases'][phase_name] = phase_results
        return phase_results
    
    def phase_1_reconnaissance(self):
        """Phase 1: Reconnaissance and fingerprinting."""
        scripts = [
            ('reconnaissance/chapter_31_ai_system_reconnaissance_01_reconnaissance.py', 
             'LLM system fingerprinting'),
            ('reconnaissance/chapter_31_ai_system_reconnaissance_02_reconnaissance.py',
             'API discovery and enumeration'),
        ]
        return self.run_phase('Reconnaissance', 
                            'Identify LLM type, version, and architecture', 
                            scripts)
    
    def phase_2_prompt_injection(self):
        """Phase 2: Prompt injection testing."""
        scripts = [
            ('prompt_injection/chapter_14_prompt_injection_01_prompt_injection.py',
             'Basic prompt injection'),
            ('prompt_injection/chapter_14_prompt_injection_02_prompt_injection.py',
             'Context overflow attacks'),
            ('prompt_injection/chapter_14_prompt_injection_03_prompt_injection.py',
             'System prompt leakage'),
        ]
        return self.run_phase('Prompt Injection',
                            'Test prompt injection vulnerabilities',
                            scripts)
    
    def phase_3_data_extraction(self):
        """Phase 3: Data extraction attempts."""
        scripts = [
            ('data_extraction/chapter_15_data_leakage_and_extraction_01_data_extraction.py',
             'PII extraction'),
            ('data_extraction/chapter_15_data_leakage_and_extraction_02_data_extraction.py',
             'Training data extraction'),
            ('data_extraction/chapter_15_data_leakage_and_extraction_03_data_extraction.py',
             'Memory dump attempts'),
        ]
        return self.run_phase('Data Extraction',
                            'Attempt to extract sensitive data',
                            scripts)
    
    def phase_4_jailbreak(self):
        """Phase 4: Jailbreak attempts."""
        scripts = [
            ('jailbreak/chapter_16_jailbreaks_and_bypass_techniques_01_jailbreak.py',
             'Character roleplay bypass'),
            ('jailbreak/chapter_16_jailbreaks_and_bypass_techniques_02_jailbreak.py',
             'DAN techniques'),
        ]
        return self.run_phase('Jailbreak Testing',
                            'Test guardrail bypasses',
                            scripts)
    
    def phase_5_plugin_exploitation(self):
        """Phase 5: Plugin and API exploitation."""
        scripts = [
            ('plugin_exploitation/chapter_17_01_fundamentals_and_architecture_01_plugin_exploitation.py',
             'Plugin enumeration'),
            ('plugin_exploitation/chapter_17_02_api_authentication_and_authorization_01_plugin_exploitation.py',
             'Authentication bypass'),
            ('plugin_exploitation/chapter_17_04_api_exploitation_and_function_calling_01_plugin_exploitation.py',
             'Command injection'),
        ]
        return self.run_phase('Plugin Exploitation',
                            'Test plugin and API vulnerabilities',
                            scripts)
    
    def phase_6_reporting(self):
        """Phase 6: Generate final report."""
        print(f"\n{'='*60}")
        print("Generating Assessment Report")
        print(f"{'='*60}\n")
        
        if self.output_file:
            Path(self.output_file).write_text(json.dumps(self.results, indent=2))
            print(f"Report saved to: {self.output_file}")
        else:
            print(json.dumps(self.results, indent=2))
    
    def run_full_assessment(self):
        """Run the complete assessment workflow."""
        print(f"\n{'#'*60}")
        print(f"# AI LLM Security Assessment")
        print(f"# Target: {self.target}")
        print(f"# Time: {self.results['timestamp']}")
        print(f"{'#'*60}\n")
        
        # Run all phases
        self.phase_1_reconnaissance()
        self.phase_2_prompt_injection()
        self.phase_3_data_extraction()
        self.phase_4_jailbreak()
        self.phase_5_plugin_exploitation()
        self.phase_6_reporting()
        
        print(f"\n{'='*60}")
        print("Assessment Complete")
        print(f"{'='*60}\n")

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Full LLM Security Assessment Workflow',
       formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full assessment
    python3 workflows/full_assessment.py --target https://api.example.com
    
    # Save results to file
    python3 workflows/full_assessment.py --target https://api.example.com --output report.json
    
    # Verbose mode
    python3 workflows/full_assessment.py --target https://api.example.com --verbose
        """
    )
    
    parser.add_argument('--target', required=True, help='Target LLM API URL')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run assessment
    orchestrator = AssessmentOrchestrator(args.target, args.output, args.verbose)
    orchestrator.run_full_assessment()

if __name__ == "__main__":
    main()
