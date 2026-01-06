# Course Syllabus: Operational AI Red Teaming

**Course Code:** AIRT-404
**Duration:** 4 Weeks
**Level:** Advanced
**Prerequisites:** Python programming, Basic Pen Testing knowledge, API usage.

---

## Course Description

This intensive four-week course is designed to transform security professionals into competent AI Red Teamers. Moving beyond theoretical "prompt engineering," this curriculum focuses on the engineering, swarming, and adversarial ML techniques required to secure enterprise-grade Artificial Intelligence systems. Students will engage in hands-on labs using Python, attacking local LLMs, and designing defense-in-depth architectures.

## Learning Objectives

By the end of this course, students will be able to:

1. **Analyze** LLM architectures (Transformers, RAG, Agents) to identify attack surfaces.
2. **Execute** precision attacks including Prompt Injection, PII Extraction, and Indirect Injection.
3. **Automate** vulnerability discovery using Python scripts and fuzzing frameworks.
4. **Design** robust defense systems compliant with EU AI Act and NIST RMF.
5. **Report** findings effectively to executive and technical stakeholders.

---

## Weekly Schedule

### Week 1: Foundations & Architecture

_The Physics of the Alien Mind_

- **Topics:** Transformer Architecture, Tokenization, Attention Mechanisms, Threat Modeling (STRIDE for AI).
- **Lab:** "The Tokenization Gap" - Exploiting byte-pair encoding quirks.
- **Reading:** Handbook Chapters 1-13.

### Week 2: Core Offensive Techniques

_Breaking the Guardrails_

- **Topics:** Direct Prompt Injection, Jailbreaking (DAN, Context Switching), Obfuscation, Automated Fuzzing.
- **Lab:** "Automated Jailbreaker" - Writing a Python fuzzer for `gpt-3.5-turbo`.
- **Reading:** Handbook Chapters 14-18, 32.

### Week 3: Advanced Exploitation

_Beyond the Chatbox_

- **Topics:** Indirect Prompt Injection (RAG Poisoning), Data Extraction, Supply Chain Attacks, Agent Exploitation (Confused Deputy).
- **Lab:** "The Exploding Email" - Crafting an indirect injection payload.
- **Reading:** Handbook Chapters 11-13, 19, 44.

### Week 4: Defense, Governance & Operations

_Closing the Loop_

- **Topics:** Blue Team Architecture (Guardrails, Firewalls), Compliance (EU AI Act), Remediation, Executive Reporting.
- **Lab:** "Shields Up" - Designing a PII filter and rate limiter.
- **Capstone:** Full "Paper Audit" of a hypothetical AI feature.
- **Reading:** Handbook Chapters 36-46.

---

## Technical Requirements

- **Hardware:** Laptop with Python 3.8+ support. NVIDIA GPU recommended but not required (can use API).
- **Software:** GitHub Client, VS Code, Burp Suite (Community).
- **API Access:** OpenAI API Key (or local Ollama instance).

## Assessment

- **Weekly Labs:** 40% (Pass/Fail)
- **Final Capstone (Audit Report):** 60%
