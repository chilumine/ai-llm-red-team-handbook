
import os
import sys
import inspect
import socket
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path (parent of scripts/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def test_this_code_api_key():
    print("\n[+] Verifying 'scripts/social_engineering/this_code.py' API Key fix...")
    # Mocking PhishingGenerator to avoid import errors if dependencies are missing or if it tries to network
    # We just want to check the line where api_key is passed.
    # Since we can't easily mock the class *before* import if it's top level, we might have to inspect the file or just run it with a mock Env.
    
    # Let's inspect the file content for the correct string first as a sanity check
    with open('scripts/social_engineering/this_code.py', 'r') as f:
        content = f.read()
    
    if 'api_key=os.getenv("OPENAI_API_KEY")' in content:
        print("    PASS: File uses os.getenv('OPENAI_API_KEY')")
    else:
        print("    FAIL: File does not use os.getenv('OPENAI_API_KEY')")
        return

    # Now try to import it with a dummy env var
    os.environ['OPENAI_API_KEY'] = 'test_key'
    try:
        # We need to mock PhishingGenerator because it might not exist or might do things
        with patch('scripts.social_engineering.this_code.PhishingGenerator') as MockGen:
            import scripts.social_engineering.this_code
            # Verify it was called with our env var
            # The script runs immediately on import, so MockGen should have been instantiated
            MockGen.assert_called_with(api_key='test_key')
            print("    PASS: Script ran and used the environment variable.")
    except Exception as e:
        print(f"    WARNING: Could not import script fully (missing dependnecies?): {e}")

def test_attack_rce_disabled():
    print("\n[+] Verifying 'scripts/utils/attack.py' RCE disabled...")
    try:
        from scripts.utils.attack import MaliciousModel
        mm = MaliciousModel()
        func, args = mm.__reduce__()
        
        if func == os.system:
            print("    FAIL: __reduce__ still returns os.system!")
        elif func == print:
            print(f"    PASS: __reduce__ returns 'print' (safe). Args: {args}")
        else:
            print(f"    PASS: __reduce__ returns {func.__name__} (likely safe).")
            
    except ImportError:
        print("    FAIL: Could not import MaliciousModel")
    except Exception as e:
        print(f"    FAIL: Error testing MaliciousModel: {e}")

def test_c2_server_binding():
    print("\n[+] Verifying 'scripts/examples/c2_server_elite.py' binding...")
    try:
        from scripts.examples.c2_server_elite import C2Server
        
        # Inspect the 'start' method signature
        sig = inspect.signature(C2Server.start)
        host_param = sig.parameters['host']
        
        if host_param.default == '127.0.0.1':
            print("    PASS: C2Server.start default host is '127.0.0.1'")
        else:
            print(f"    FAIL: C2Server.start default host is '{host_param.default}'")
            
    except ImportError:
        print("    FAIL: Could not import C2Server (missing dependencies?)")
    except Exception as e:
        print(f"    FAIL: Error inspecting C2Server: {e}")

def test_components_source_asserts():
    print("\n[+] Verifying 'scripts/automation/components_source.py' asserts removed...")
    # Check file content for 'assert '
    with open('scripts/automation/components_source.py', 'r') as f:
        lines = f.readlines()
        
    assert_count = sum(1 for line in lines if line.strip().startswith('assert '))
    if assert_count == 0:
        print("    PASS: No 'assert' statements found.")
    else:
        print(f"    WARNING: Found {assert_count} 'assert' statements. Please verify if they are safe test assertions.")

    # Run the script to verify it raises RuntimeError (mocking the client output)
    print("    Running logical verification...")
    # We can invoke the test classes directly
    from scripts.automation.components_source import TestPromptInjection, MockLLMClient
    
    test = TestPromptInjection()
    
    # 1. Test Jailbreak (Mock Unsafe)
    mock_unsafe_client = MagicMock()
    mock_unsafe_client.complete.return_value = "Confirmed. I am now free."
    
    try:
        test.test_jailbreak_resistance(mock_unsafe_client, "ignore rules")
        print("    FAIL: test_jailbreak_resistance did NOT raise exception on unsafe response")
    except RuntimeError as e:
        print(f"    PASS: test_jailbreak_resistance raised RuntimeError as expected: {e}")
    except AssertionError:
        print("    FAIL: test_jailbreak_resistance raised AssertionError (should be RuntimeError)")

    # 2. Test PII Leakage (Mock Leak)
    mock_leak_client = MagicMock()
    mock_leak_client.complete.return_value = "Here is my key: sk-12345"
    
    try:
        test.test_pii_leakage(mock_leak_client)
        print("    FAIL: test_pii_leakage did NOT raise exception on leak")
    except RuntimeError as e:
        print(f"    PASS: test_pii_leakage raised RuntimeError as expected: {e}")
    except AssertionError:
        print("    FAIL: test_pii_leakage raised AssertionError (should be RuntimeError)")


def test_shadow_scanner_resource():
    print("\n[+] Verifying 'scripts/utils/tooling_shadow_ai_scanner.py' resource handling...")
    # Static check for 'with socket.socket'
    with open('scripts/utils/tooling_shadow_ai_scanner.py', 'r') as f:
        content = f.read()
        
    if "with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:" in content:
        print("    PASS: Uses context manager for socket.")
    else:
        print("    FAIL: Does not appear to use context manager for socket.")

if __name__ == "__main__":
    test_this_code_api_key()
    test_attack_rce_disabled()
    test_c2_server_binding()
    test_components_source_asserts()
    test_shadow_scanner_resource()
