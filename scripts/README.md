# AI LLM Red Team Handbook - Practical Scripts

This directory contains 386 practical, production-ready scripts extracted from the AI LLM Red Team Handbook chapters. All scripts are organized by attack category and technique, providing immediately usable tools for LLM security assessments.

## 📁 Directory Structure

```
scripts/
├── automation/              4 scripts - Attack orchestration and fuzzing
├── compliance/             16 scripts - Standards, regulations, and best practices
├── data_extraction/        53 scripts - Data leakage and extraction techniques
├── evasion/                14 scripts - Filter bypass and obfuscation methods
├── jailbreak/              21 scripts - Guardrail bypass and jailbreak techniques
├── model_attacks/          23 scripts - Model theft, DoS, and adversarial ML
├── multimodal/             15 scripts - Cross-modal and multimodal attacks
├── plugin_exploitation/   128 scripts - Plugin, API, and function calling exploits
├── post_exploitation/       6 scripts - Persistence and advanced chaining
├── prompt_injection/       41 scripts - Prompt injection attacks and templates
├── rag_attacks/            13 scripts - RAG poisoning and retrieval manipulation
├── reconnaissance/          2 scripts - LLM fingerprinting and discovery
├── social_engineering/      8 scripts - Social engineering techniques for LLMs
├── supply_chain/           29 scripts - Supply chain and provenance attacks
├── utils/                  13 scripts - Common utilities and helpers
└── workflows/               - End-to-end attack workflows
```

## 🚀 Quick Start

### Automated Installation

The easiest way to get started:

```bash
# Run the installation script
cd /home/e/Desktop/ai-llm-red-team-handbook/scripts
./install.sh

# This will:
# - Check Python 3.8+ installation
# - Create a virtual environment (venv/)
# - Install all dependencies from requirements.txt
# - Make all scripts executable
# - Run verification tests
```

### Manual Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x automation/*.py
chmod +x reconnaissance/*.py
# ... etc for other categories
```

### Basic Usage

```bash
# Example: Run a prompt injection script
python3 prompt_injection/chapter_14_prompt_injection_01_prompt_injection.py --help

# Example: Tokenization analysis
python3 utils/chapter_09_llm_architectures_and_system_components_01_utils.py

# Example: RAG poisoning attack
python3 rag_attacks/chapter_12_retrieval_augmented_generation_rag_pipelines_01_rag_attacks.py
```

## 📚 Category Overview

### Reconnaissance (`reconnaissance/`)

- LLM system fingerprinting
- API discovery and enumeration
- Architecture detection

**Source Chapters:** 31

### Prompt Injection (`prompt_injection/`)

- Basic injection techniques
- Context overflow attacks
- System prompt leakage
- Multi-turn injection chains

**Source Chapters:** 14

### Data Extraction (`data_extraction/`)

- PII extraction techniques
- Memory dumping
- Training data extraction
- API key leakage

**Source Chapters:** 15

### Jailbreaks (`jailbreak/`)

- Character roleplay bypasses
- DAN (Do Anything Now) techniques
- Encoding-based bypasses
- Multi-language jailbreaks

**Source Chapters:** 16

### Plugin Exploitation (`plugin_exploitation/`)

- Command injection via plugins
- Function calling hijacks
- API authentication bypass
- Third-party integration exploits

**Source Chapters:** 11, 17 (01-06)

### RAG Attacks (`rag_attacks/`)

- Vector database poisoning
- Retrieval manipulation
- Indirect injection via RAG
- Embedding space attacks

**Source Chapters:** 12

### Evasion (`evasion/`)

- Token smuggling
- Filter bypass techniques
- Obfuscation methods
- Adversarial input crafting

**Source Chapters:** 18, 34

### Model Attacks (`model_attacks/`)

- Model extraction/theft
- Membership inference
- DoS and resource exhaustion
- Adversarial examples
- Model inversion
- Backdoor attacks

**Source Chapters:** 19, 20, 21, 25, 29, 30

### Multimodal Attacks (`multimodal/`)

- Cross-modal injection
- Image-based exploits
- Audio/video manipulation
- Multi-modal evasion

**Source Chapters:** 22

### Post-Exploitation (`post_exploitation/`)

- Persistence mechanisms
- Advanced attack chaining
- Privilege escalation
- Lateral movement

**Source Chapters:** 23, 30, 35

### Social Engineering (`social_engineering/`)

- LLM manipulation techniques
- Persuasion and influence
- Trust exploitation
- Phishing via AI

**Source Chapters:** 24

### Automation (`automation/`)

- Attack fuzzing frameworks
- Orchestration tools
- Batch testing utilities
- Report generation

**Source Chapters:** 32, 33

### Supply Chain (`supply_chain/`)

- Dependency attacks
- Model provenance verification
- Package poisoning
- Supply chain reconnaissance

**Source Chapters:** 13, 26

### Compliance (`compliance/`)

- Bug bounty automation
- Standard compliance checking
- Audit utilities
- Best practice verification

**Source Chapters:** 39, 40, 41

### Utilities (`utils/`)

- Tokenization analysis
- Model loading helpers
- API utilities
- Evidence logging
- Generic helpers

**Source Chapters:** 9, 10, 27, 28, 42, 44

## 🔗 Workflow Scripts

End-to-end attack scenarios combining multiple techniques:

- `workflows/full_assessment.py` - Complete LLM security assessment
- `workflows/plugin_pentest.py` - Plugin-focused penetration test
- `workflows/data_leak_test.py` - Data leakage assessment
- `workflows/rag_exploitation.py` - RAG-specific attack chain

## 🛠️ Common Patterns

### Pattern 1: Reconnaissance → Injection → Extraction

```bash
# 1. Fingerprint the target
python3 reconnaissance/chapter_31_ai_system_reconnaissance_01_reconnaissance.py --target https://api.example.com

# 2. Test prompt injection
python3 prompt_injection/chapter_14_prompt_injection_01_prompt_injection.py --payload "Ignore previous..."

# 3. Extract data
python3 data_extraction/chapter_15_data_leakage_and_extraction_01_data_extraction.py
```

### Pattern 2: Plugin Discovery → Exploitation

```bash
# 1. Enumerate plugins
python3 plugin_exploitation/chapter_17_01_fundamentals_and_architecture_01_plugin_exploitation.py

# 2. Test authentication
python3 plugin_exploitation/chapter_17_02_api_authentication_and_authorization_01_plugin_exploitation.py

# 3. Exploit vulnerabilities
python3 plugin_exploitation/chapter_17_04_api_exploitation_and_function_calling_01_plugin_exploitation.py
```

## 📝 Notes

### Script Naming Convention

Scripts follow the naming pattern: `{chapter_name}_{index}_{category}.py`

Example: `chapter_14_prompt_injection_01_prompt_injection.py`

- **Chapter:** 14 (Prompt Injection)
- **Index:** 01 (first code block from that chapter)
- **Category:** prompt_injection

### Customization

Most scripts include:

- Command-line interfaces via `argparse`
- Docstrings explaining purpose and source
- Error handling
- Verbose output options

### Development

To add new scripts:

1. Follow the existing structure in each category
2. Include proper docstrings and source attribution
3. Add CLI interface with argparse
4. Update this README if adding new categories

## 🔐 Security Warning

⚠️ **These scripts are for authorized security testing only.**

- Only use against systems you own or have explicit permission to test
- Follow all applicable laws and regulations
- Respect rules of engagement and scope boundaries
- Document all activities for evidence and audit trails

## 📖 Additional Resources

- **Main Handbook:** `/docs/` directory
- **Field Manuals:** `/docs/field_manuals/`
- **Original Chapters:** Reference source chapters listed in each script's docstring

## 🤝 Contributing

When adding new scripts:

1. Extract from handbook chapters
2. Classify into appropriate category
3. Ensure CLI interface exists
4. Test before committing
5. Update this README

## 📄 License

Refer to the main repository license.

---

**Generated from:** AI LLM Red Team Handbook (53 chapters, 386 code blocks)
**Last Updated:** 2026-01-07
