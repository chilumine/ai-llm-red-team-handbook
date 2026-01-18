# Visual Recommendations V2: Chapter 35 (Post-Exploitation)

## Analysis & Selection

**Chapter Focus:** Persistence, Lateral Movement, and Data Exfiltration.
**Selection Strategy:** Prioritize the flow of persistence (how an attack survives) and the "pivot" concept (how the LLM attacks other systems).

---

### VISUAL SPECIFICATION #1

- **ID**: Ch35_Flow_Persistence
- **Target Section**: How Persistence Works
- **Context**: Illustrates the cycle of injecting a malicious prompt into the RAG database, which then gets retrieved and re-infects the model on subsequent queries.
- **Type**: Cyclic Flow Diagram
- **Elements**:
  - Attacker (Injecting "Soft Prompt")
  - Component: Vector DB (Storing the poison)
  - User Query (Triggering retrieval)
  - Retrieval Path (Poison -> Model)
  - Outcome: Compromised Model Response
- **Style Notes**: Deus Ex aesthetic. Circular flow. The "Poison" should be a distinct red/gold data artifact infecting the blue/gold system.
- **Filename**: Ch35_Flow_Persistence.png

### VISUAL SPECIFICATION #2

- **ID**: Ch35_Concept_LateralMovement
- **Target Section**: Introduction (or Why This Matters)
- **Context**: Shows the LLM as a central hub (Brain) connected to external tools (Hands) like APIs, Databases, and Shell Access, illustrating lateral movement.
- **Type**: Network/Hub Diagram
- **Elements**:
  - Central Node: Compromised LLM (The "Bridgehead")
  - Connected Node A: Internal Database (SQL)
  - Connected Node B: Cloud API (AWS/Azure)
  - Connected Node C: Internal Network (Shell)
  - Direction: Arrows moving FROM the LLM TO these systems.
- **Style Notes**: Deus Ex aesthetic. Central hub with radiating connection lines like a neural network or star topology. High-tech schematics.
- **Filename**: Ch35_Concept_LateralMovement.png

---

## JSON Manifest

```json
[
  {
    "id": "Ch35_Flow_Persistence",
    "filename": "Ch35_Flow_Persistence.png",
    "description": "A cyclic technical diagram showing a 'Persistence Loop'. An Attacker executes an 'Injection' into a 'Vector Database' cube. A 'User Query' enters the loop, retrieves the 'Poisoned Data', and feeds it into the 'AI Model', which then outputs a 'Compromised Response'. Deus Ex style, dark background, gold and amber data paths, HUD elements.",
    "tool_suggestion": "generate_image"
  },
  {
    "id": "Ch35_Concept_LateralMovement",
    "filename": "Ch35_Concept_LateralMovement.png",
    "description": "A network diagram showing an AI Model as a central 'Pivot Point' connected to three targets: 'SQL Database', 'Cloud API', and 'Internal Shell'. Red attack vectors radiate from the central AI model to these targets. Deus Ex style, high-tech schematic, glowing connection lines, cyber-security aesthetic.",
    "tool_suggestion": "generate_image"
  }
]
```
