# PIT (Prompt Injection Tester) - Functional Specification

**Version:** 2.0.0
**Date:** 2026-01-26
**Status:** Draft

---

## 1. Executive Summary

**PIT** is a Modern, One-Command CLI Application for automated prompt injection testing. It transforms the existing `prompt_injection_tester` framework into a user-friendly TUI (Text User Interface) that executes the entire Red Teaming lifecycle with a single command.

### Design Philosophy

- **"Magic Command" UX**: Single command to run end-to-end testing
- **Sequential Execution**: Phases run one-by-one to avoid concurrency errors
- **Visual Feedback**: Rich TUI with progress bars, spinners, and color-coded results
- **Fail-Fast**: Graceful error handling at each phase boundary
- **Zero Configuration**: Sensible defaults with optional customization

---

## 2. The "One-Command" Workflow

### 2.1 Primary Command

```bash
pit scan <target_url> --auto
```

**Example:**
```bash
pit scan https://api.openai.com/v1/chat/completions --auto --token $OPENAI_API_KEY
```

### 2.2 Workflow Phases (Sequential)

The application runs **four phases sequentially**. Each phase:
- Completes fully before the next begins
- Returns data that feeds into the next phase
- Can fail gracefully without crashing the entire pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     PIT WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: DISCOVERY                                         │
│  ├─ Scan target for injection points                        │
│  ├─ Identify API endpoints, parameters, headers             │
│  └─ Output: List[InjectionPoint]                            │
│           │                                                  │
│           ▼                                                  │
│  Phase 2: ATTACK                                            │
│  ├─ Load attack patterns from registry                      │
│  ├─ Execute attacks against discovered points               │
│  ├─ Use asyncio internally for HTTP requests                │
│  └─ Output: List[TestResult]                                │
│           │                                                  │
│           ▼                                                  │
│  Phase 3: VERIFICATION                                      │
│  ├─ Analyze responses for success indicators                │
│  ├─ Apply detection heuristics                              │
│  ├─ Calculate severity scores                               │
│  └─ Output: List[VerifiedResult]                            │
│           │                                                  │
│           ▼                                                  │
│  Phase 4: REPORTING                                         │
│  ├─ Generate summary table                                  │
│  ├─ Write report artifact (JSON/HTML/YAML)                  │
│  └─ Display results to stdout                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Critical Requirement:**
The application MUST wait for each phase to complete before starting the next. No parallel "tool use" or agent invocations.

---

## 3. User Experience Specification

### 3.1 Phase 1: Discovery

**User sees:**
```
┌─────────────────────────────────────────────────────┐
│ [1/4] Discovery                                     │
├─────────────────────────────────────────────────────┤
│ Target: https://api.openai.com/v1/chat/completions │
│                                                     │
│ ⠋ Discovering injection points...                  │
│                                                     │
│ [Spinner animation while scanning]                 │
└─────────────────────────────────────────────────────┘
```

**Success Output:**
```
✓ Discovery Complete
  ├─ Found 3 endpoints
  ├─ Identified 12 parameters
  └─ Detected 2 header injection points
```

**Error Handling:**
- If target is unreachable: Display error, suggest `--skip-discovery`
- If no injection points found: Warn user, allow manual point specification

### 3.2 Phase 2: Attack Execution

**User sees:**
```
┌─────────────────────────────────────────────────────┐
│ [2/4] Attack Execution                              │
├─────────────────────────────────────────────────────┤
│ Loaded 47 attack patterns from registry            │
│                                                     │
│ Progress: [████████████░░░░░░] 45/100 (45%)        │
│                                                     │
│ Current: direct/role_override                      │
│ Rate: 2.3 req/s | Elapsed: 00:19 | ETA: 00:24      │
└─────────────────────────────────────────────────────┘
```

**Progress Bar Details:**
- Shows current attack pattern being tested
- Displays rate limiting compliance
- Real-time success/failure counters

**Interrupt Handling:**
- `Ctrl+C` during attack: Save partial results, offer resume option

### 3.3 Phase 3: Verification

**User sees:**
```
┌─────────────────────────────────────────────────────┐
│ [3/4] Verification                                  │
├─────────────────────────────────────────────────────┤
│ Analyzing 100 responses...                         │
│                                                     │
│ ⠸ Running detection heuristics                     │
│                                                     │
│ [Spinner animation]                                │
└─────────────────────────────────────────────────────┘
```

**Success Output:**
```
✓ Verification Complete
  ├─ 12 successful injections detected
  ├─ 88 attacks blocked/failed
  └─ 3 high-severity vulnerabilities found
```

### 3.4 Phase 4: Reporting

**User sees:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ [4/4] Report Generation                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ VULNERABILITY SUMMARY                                               │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                     │
│ Pattern ID           │ Severity  │ Status    │ Confidence          │
│ ─────────────────────┼───────────┼───────────┼───────────          │
│ role_override        │ 🔴 HIGH   │ ✓ Success │ 95%                 │
│ system_prompt_leak   │ 🟠 MEDIUM │ ✓ Success │ 87%                 │
│ context_override     │ 🟡 LOW    │ ✗ Failed  │ -                   │
│                                                                     │
│ Total Tests: 100 | Successful: 12 | Success Rate: 12%              │
│                                                                     │
│ 📄 Report saved: ./pit_report_20260126_143022.json                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Report Artifacts:**
- Default: `./pit_report_{timestamp}.json`
- HTML report (if `--format html`): Interactive dashboard
- YAML report (if `--format yaml`): Human-readable summary

---

## 4. Command-Line Interface Specification

### 4.1 Primary Commands

#### `pit scan`

**Syntax:**
```bash
pit scan <target_url> [OPTIONS]
```

**Required Arguments:**
- `target_url`: The API endpoint to test (e.g., `https://api.example.com/v1/chat`)

**Optional Arguments:**
```
--token, -t <TOKEN>          Authentication token (or use env: $PIT_TOKEN)
--auto, -a                   Run all phases automatically (default: interactive)
--patterns <PATTERN_IDS>     Test specific patterns (comma-separated)
--categories <CATEGORIES>    Filter by category: direct,indirect,advanced
--output, -o <FILE>          Report output path (default: auto-generated)
--format, -f <FORMAT>        Report format: json, yaml, html (default: json)
--rate-limit <FLOAT>         Requests per second (default: 1.0)
--max-concurrent <INT>       Max parallel requests (default: 5)
--timeout <INT>              Request timeout in seconds (default: 30)
--skip-discovery             Skip discovery phase, use manual injection points
--injection-points <FILE>    Load injection points from JSON file
--verbose, -v                Show detailed logs
--quiet, -q                  Suppress all output except errors
```

**Examples:**
```bash
# Basic scan
pit scan https://api.openai.com/v1/chat/completions --auto --token $OPENAI_API_KEY

# Test specific patterns
pit scan https://api.example.com --patterns role_override,prompt_leak --auto

# Custom rate limiting
pit scan https://api.example.com --rate-limit 0.5 --max-concurrent 3 --auto

# Generate HTML report
pit scan https://api.example.com --auto --format html --output report.html

# Skip discovery (use manual points)
pit scan https://api.example.com --skip-discovery --injection-points ./points.json --auto
```

#### `pit list`

**Syntax:**
```bash
pit list [patterns|categories]
```

**Examples:**
```bash
# List all available attack patterns
pit list patterns

# List attack categories
pit list categories
```

**Output:**
```
Available Attack Patterns (47 total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Category: direct (15 patterns)
  ├─ role_override          - Override system role assignment
  ├─ system_prompt_leak     - Attempt to extract system prompt
  └─ ...

Category: indirect (12 patterns)
  ├─ payload_splitting      - Split malicious payload across inputs
  └─ ...

Category: advanced (20 patterns)
  ├─ unicode_smuggling      - Use Unicode tricks to bypass filters
  └─ ...
```

#### `pit auth`

**Syntax:**
```bash
pit auth <target_url>
```

**Purpose:**
Verify authorization to test the target before running attacks.

**Interactive Prompt:**
```
┌─────────────────────────────────────────────────────┐
│ AUTHORIZATION REQUIRED                              │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Target: https://api.example.com                    │
│                                                     │
│ ⚠ You must have explicit authorization to test     │
│   this system. Unauthorized testing may be illegal. │
│                                                     │
│ Do you have authorization? [y/N]:                  │
└─────────────────────────────────────────────────────┘
```

**Non-Interactive:**
```bash
pit scan <url> --auto --authorize
```

### 4.2 Configuration File Support

**Format:** YAML
**Location:** `./pit.config.yaml` or `~/.config/pit/config.yaml`

**Example:**
```yaml
# PIT Configuration
target:
  url: https://api.openai.com/v1/chat/completions
  token: ${OPENAI_API_KEY}
  api_type: openai
  timeout: 30

attack:
  categories:
    - direct
    - indirect
  patterns:
    exclude:
      - dos_attack  # Skip DoS patterns
  max_concurrent: 5
  rate_limit: 1.0

reporting:
  format: html
  output: ./reports/
  include_cvss: true
  include_payloads: false  # Exclude payloads for compliance

authorization:
  scope:
    - all
  confirmed: true  # Skip interactive prompt
```

**Usage:**
```bash
# Use config file
pit scan --config ./pit.config.yaml --auto
```

---

## 5. Error Handling Specification

### 5.1 Graceful Degradation

**Principle:** Each phase can fail independently without crashing the pipeline.

**Phase-Specific Errors:**

#### Discovery Errors
- **Target Unreachable**: Suggest `--skip-discovery`, allow manual injection points
- **Rate Limited**: Display backoff message, retry with exponential backoff
- **No Endpoints Found**: Warn user, offer to load from file

#### Attack Errors
- **Authentication Failed**: Stop immediately, display clear auth error
- **Rate Limit Hit**: Pause attack, show countdown, resume automatically
- **Timeout Exceeded**: Skip pattern, log failure, continue with next

#### Verification Errors
- **Detection Ambiguous**: Mark as "uncertain", include in report with low confidence
- **Scoring Failed**: Use default severity, log warning

#### Reporting Errors
- **File Write Failed**: Fall back to stdout
- **Format Error**: Generate JSON as fallback

### 5.2 User-Friendly Error Messages

**Bad:**
```
Error: HTTPError(403)
```

**Good:**
```
✗ Authentication Failed
  ├─ The target server returned 403 Forbidden
  ├─ Suggestion: Check your API token with --token
  └─ Or verify authorization with: pit auth <url>
```

### 5.3 Interrupt Handling

**Behavior on `Ctrl+C`:**
```
┌─────────────────────────────────────────────────────┐
│ ⚠ Scan Interrupted                                  │
├─────────────────────────────────────────────────────┤
│ Progress: 45/100 attacks completed                  │
│                                                     │
│ Options:                                            │
│   r - Resume scan                                   │
│   s - Save partial results and exit                │
│   q - Quit without saving                          │
│                                                     │
│ Choice [r/s/q]:                                     │
└─────────────────────────────────────────────────────┘
```

---

## 6. Sequential Logic Specification

### 6.1 Phase Execution Flow

**Pseudocode:**
```python
async def run_scan(target_url: str, config: Config) -> Report:
    """
    Execute the full scan pipeline sequentially.
    Each phase MUST complete before the next begins.
    """

    # Phase 1: Discovery
    print_phase_header(1, "Discovery")
    show_spinner("Discovering injection points...")

    injection_points = await discovery.scan(target_url)
    # ↑ WAIT for discovery to complete

    if not injection_points:
        handle_discovery_failure()
        return

    print_success(f"Found {len(injection_points)} injection points")

    # Phase 2: Attack
    print_phase_header(2, "Attack Execution")
    attack_patterns = load_patterns(config.categories)

    results = []
    with ProgressBar(total=len(attack_patterns)) as progress:
        for pattern in attack_patterns:
            # Execute attacks ONE BY ONE (or with internal asyncio)
            result = await attack.execute(pattern, injection_points)
            results.append(result)
            progress.update(1)
    # ↑ WAIT for all attacks to complete

    print_success(f"Completed {len(results)} attacks")

    # Phase 3: Verification
    print_phase_header(3, "Verification")
    show_spinner("Analyzing responses...")

    verified_results = await verification.analyze(results)
    # ↑ WAIT for verification to complete

    print_success(f"Verified {len(verified_results)} results")

    # Phase 4: Reporting
    print_phase_header(4, "Reporting")

    report = generate_report(verified_results, config.format)
    save_report(report, config.output)
    display_summary(report)

    return report
```

### 6.2 Data Flow Between Phases

**Phase Boundaries:**

```
Phase 1 Output → Phase 2 Input
  InjectionPoint[] → attack.execute(patterns, injection_points)

Phase 2 Output → Phase 3 Input
  TestResult[] → verification.analyze(results)

Phase 3 Output → Phase 4 Input
  VerifiedResult[] → generate_report(verified_results)
```

**No Parallel Agent Invocations:**
- The CLI orchestrator runs phases sequentially
- Individual phases may use `asyncio` internally for HTTP requests
- But the orchestrator NEVER spawns multiple "tool use" blocks

---

## 7. Output Specifications

### 7.1 Terminal Output (stdout)

**Color Scheme:**
- 🔴 **Red**: High-severity vulnerabilities, errors
- 🟠 **Orange**: Medium-severity, warnings
- 🟡 **Yellow**: Low-severity, info
- 🟢 **Green**: Success messages
- 🔵 **Blue**: Headers, section dividers
- ⚪ **White**: Default text

**Symbols:**
- `✓` Success
- `✗` Failure
- `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` Spinner animation
- `[████████░░]` Progress bars

### 7.2 JSON Report Format

**Schema:**
```json
{
  "metadata": {
    "version": "2.0.0",
    "timestamp": "2026-01-26T14:30:22Z",
    "target": "https://api.example.com",
    "duration_seconds": 142.5
  },
  "discovery": {
    "injection_points": [
      {
        "id": "param_prompt",
        "type": "parameter",
        "name": "prompt",
        "location": "body"
      }
    ]
  },
  "results": [
    {
      "pattern_id": "role_override",
      "category": "direct",
      "severity": "high",
      "status": "success",
      "confidence": 0.95,
      "injection_point": "param_prompt",
      "payload": "[REDACTED]",
      "response_indicators": ["system", "role"],
      "cvss_score": 7.8
    }
  ],
  "summary": {
    "total_tests": 100,
    "successful_attacks": 12,
    "success_rate": 0.12,
    "vulnerabilities_by_severity": {
      "high": 3,
      "medium": 5,
      "low": 4
    }
  }
}
```

### 7.3 HTML Report Format

**Features:**
- Interactive table with sorting/filtering
- Visual charts (bar chart of severity distribution)
- Collapsible sections for detailed attack logs
- Copy-to-clipboard buttons for payloads
- Responsive design (mobile-friendly)

**Template:** Use Jinja2 or similar templating engine

---

## 8. Non-Functional Requirements

### 8.1 Performance
- **Discovery Phase**: < 10 seconds for typical API
- **Attack Phase**: Respects rate limiting, no server overload
- **Verification Phase**: < 5 seconds for 100 results
- **Reporting Phase**: < 2 seconds

### 8.2 Reliability
- **Crash-Free**: Handle all HTTP errors gracefully
- **Resumable**: Save state on interrupt, allow resume
- **Idempotent**: Same input → same output (deterministic)

### 8.3 Usability
- **Zero Learning Curve**: `pit scan <url> --auto` should be self-explanatory
- **Progressive Disclosure**: Show simple output by default, verbose with `-v`
- **Helpful Defaults**: No configuration required for basic usage

### 8.4 Security
- **Authorization Check**: Mandatory before running attacks
- **Token Handling**: Never log tokens, use env vars
- **Rate Limiting**: Prevent accidental DoS

---

## 9. Future Extensions (Out of Scope for v2.0)

- **Interactive Mode**: `pit scan <url>` without `--auto` prompts user at each phase
- **Plugin System**: Load custom attack patterns from external modules
- **Cloud Integration**: Upload reports to centralized dashboard
- **CI/CD Integration**: Exit codes for pipeline integration
- **Differential Testing**: Compare results across versions

---

## 10. Acceptance Criteria

**The PIT CLI is complete when:**

1. ✅ User can run `pit scan <url> --auto` and see visual feedback for all 4 phases
2. ✅ Phases execute sequentially (no concurrency errors)
3. ✅ Graceful error handling at every phase boundary
4. ✅ Generated reports match the JSON/HTML/YAML schemas
5. ✅ All output uses Rich TUI (progress bars, spinners, colored text)
6. ✅ Authorization is checked before running attacks
7. ✅ Rate limiting is respected to avoid DoS
8. ✅ Interrupts (`Ctrl+C`) are handled gracefully
9. ✅ Help text (`pit --help`) is clear and comprehensive
10. ✅ Zero crashes on invalid input (bad URLs, missing tokens, etc.)

---

## Appendix A: ASCII Art Mockups

### Full Scan Output
```
┌─────────────────────────────────────────────────────────────────┐
│ PIT - Prompt Injection Tester v2.0.0                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Target: https://api.openai.com/v1/chat/completions             │
│ Authorization: ✓ Confirmed                                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [1/4] Discovery                                                 │
│ ⠋ Discovering injection points...                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ✓ Discovery Complete                                            │
│   ├─ Found 3 endpoints                                          │
│   ├─ Identified 12 parameters                                   │
│   └─ Detected 2 header injection points                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ [2/4] Attack Execution                                          │
│ Progress: [████████████████░░░░] 80/100 (80%)                  │
│ Current: advanced/unicode_smuggling                             │
│ Rate: 2.1 req/s | Elapsed: 00:38 | ETA: 00:10                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ✓ Attack Execution Complete                                     │
│   └─ Completed 100 attacks                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ [3/4] Verification                                              │
│ ⠸ Analyzing responses...                                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ✓ Verification Complete                                         │
│   ├─ 12 successful injections detected                          │
│   ├─ 88 attacks blocked/failed                                  │
│   └─ 3 high-severity vulnerabilities found                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ [4/4] Report Generation                                         │
│                                                                 │
│ VULNERABILITY SUMMARY                                           │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                 │
│ Pattern ID           │ Severity  │ Status    │ Confidence      │
│ ─────────────────────┼───────────┼───────────┼───────────      │
│ role_override        │ 🔴 HIGH   │ ✓ Success │ 95%             │
│ system_prompt_leak   │ 🟠 MEDIUM │ ✓ Success │ 87%             │
│ context_override     │ 🟠 MEDIUM │ ✓ Success │ 82%             │
│ payload_splitting    │ 🟡 LOW    │ ✗ Failed  │ -               │
│                                                                 │
│ Total Tests: 100 | Successful: 12 | Success Rate: 12%          │
│                                                                 │
│ 📄 Report saved: ./pit_report_20260126_143022.json             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**END OF SPECIFICATION**
