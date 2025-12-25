# Agent.md — Codex Agent Setup (Local Dev)

This repo implements an **NPC roleplay/chat agent** that:
- Builds RP prompts from a `Character` definition (pl_list, example dialogues, situation)
- Retrieves **context** from a Chroma-backed knowledge store (mini RAG)
- Generates a final NPC response via a text generation backend (`ChromaDBHelper.generate_text`)

The core flow lives in `Character.prompt()` and `Character.initiate_conversion()`.

---

## 1) Architecture Snapshot

### File / folder structure (architectural context)
Use the repository layout as the primary mental model:

- `/classes`
  - Contains the core classes used for the **roleplaying aspect**.
  - Examples: `Character`, `ChromaDBHelper`, and other roleplay/prompt/RAG utilities.
  - In general: orchestration and “how to behave” live here.

- `/data`
  - Contains raw data (primarily **CSV**) for:
    - lore
    - character information
    - faction information
  - This data is used to build and operate the **mini RAG** as the knowledge retrieval strategy.

- `/test`
  - Test files that attempt to verify the **quality of NPC roleplaying**.
  - Includes both:
    - automated verification via an AI-judge model
    - classic unit tests

### Top-level scripts (entrypoints and data pipelines)
- `add_character_embeddings.py`
  - Loads character data and adds embeddings to the mini RAG.

- `add_faction_embeddings.py`
  - Loads faction data and adds embeddings to the mini RAG.

- `query_check_character.py`
  - Entryway to start a conversation with a character via a selection menu.

- `models.py`
  - Definition of all relevant data structures, including enums and data object definitions. Here are also the enums relevant for knowledge retrieval mechanics

**Design principle (high-level):**  
Prefer keeping “RP logic and orchestration” inside `/classes`, and keep “data ingestion + embedding creation” in top-level scripts.

---

## 2) Local Setup (why Codex benefits from this)

This section is not primarily a “getting started guide” for humans — it’s here because Codex tends to make better changes when it understands:

- **What environment assumptions are safe** (e.g., Python version range, venv usage)
- **What dependencies are expected** (so changes don’t introduce mismatched tooling)
- **What services might exist locally** (e.g., Chroma or a local model server), so Codex avoids implementing code that assumes cloud-only or unavailable infrastructure
- **How the code is actually executed** (entrypoint scripts), so it edits the right files and doesn’t invent new run paths

In short: it reduces “helpful but wrong” refactors, and prevents Codex from proposing changes that won’t run locally.

---

## 3) General NPC conversation loop

The rough process looks like this:

1. **User selects a character** to talk to (selection menu / entry script).
2. **NPC starts with a greeting**.
3. **User prompts the NPC freely** (no strict format required).
4. **NPC computes an answer**, typically involving:
   4.1 **Internal response formation**: emotional reaction, possible intentions, behavioral effects, etc.  
   4.2 **Knowledge recall / retrieval**: the character may need information (knowledge, lore, memories).  
   4.3 **Final response generation**: given the full context from 4.1 and 4.2, the LLM generates the response text.
5. **NPC decides whether to continue** the conversation.
   - If yes: repeat from step 3.
   - If no: the script ends.

---

## 4) Knowledge Retrieval Rules (Chroma Metadata Filters)

The agent uses metadata filters with `$and` / `$or` to retrieve context.

(POC note: the specific enum mappings and exact filter composition are still in flux. At a high level, “fetching knowledge” means creating the appropriate metadata filters/tags so Chroma returns the right slices of lore/memories/relations/etc.)

---

## 5) Prompting Conventions (RP)

### Greeting prompt
The greeting prompt should:
- Force first-person RP as the character
- Include situation + character definition + example dialogues (if available)
- Instruct the model to initiate the interaction in-character

### Answer prompt
The answer prompt should:
- Include situation and a “General context” section (retrieved from Chroma)
- Enforce:
  - first-person RP
  - include non-verbal actions as `*...*`
  - do not reveal NPC thoughts
  - output only dialogue-perceivable content

When editing prompt text:
- Keep instructions explicit and near the bottom
- Avoid ambiguous constraints like “be concise” unless required
- Prefer stable headings because retrieval quality tends to benefit from consistent structure

---

## 6) Developing / Extending Tools

If/when you add new “agent behaviors” (even if not formalized as tools yet), keep these principles:

- Validate model outputs defensively (LLM-driven inputs can be malformed)
- Prefer small, reversible changes (agent behavior is prompt-sensitive)
- Keep behavior-related orchestration inside `/classes`, not in entry scripts

---

## 7) Debugging Checklist

If the NPC responds with irrelevant text:
- Inspect the retrieved context output from the mini RAG
- Confirm metadata tags / filters match how documents are stored
- Check the prompt for missing constraints (RP format, “no internal thoughts”, etc.)

If retrieval seems empty or noisy:
- Verify the embedding scripts populated the correct collections
- Ensure the CSV data you expect is being loaded and embedded
- Verify your filter/tag strategy matches your stored metadata schema

If the loop never ends / ends too early:
- Check the “continue/stop conversation” decision logic
- Ensure the entry script correctly exits when the character stops

---

## 8) What Codex Should Do in This Repo

When making changes, Codex should:
1. Preserve the existing high-level flow (selection → greeting → user prompt → context/reasoning → response → continue/stop)
2. Keep RP constraints intact (first-person, `*...*`, no hidden thoughts)
3. Avoid large refactors unless explicitly requested (the system is prompt-sensitive)
4. Prefer improving clarity/robustness of retrieval filters/tags rather than inventing parallel retrieval systems
5. Add minimal tests where feasible, especially around retrieval filter building and prompt formatting

If asked to implement new agent behaviors:
- Prefer extending behavior in `/classes` close to where prompts and retrieval are orchestrated
- Add/adjust tests in `/test` to prevent regressions in RP quality

**Do**
- Do preserve prompt section headings and ordering unless explicitly asked.
- Do keep RP rules explicit and near the bottom of prompt templates.
- Do edit behavior in `/classes` and keep entry scripts thin.
- Do add or update tests in `/test` when prompt or retrieval logic changes.
- Do prefer small, reversible changes; describe intent in code comments only when needed.
- Do always ask for additional approval in cases an action would be advisable but contradict the rules stated in the file.

**Don’t**
- Don’t change the prompt workflow without explicit approval.
- Don’t change RP output format (first‑person, `*...*`, no internal thoughts).
- Don’t “simplify” prompts by removing constraints or examples.
- Don’t refactor entry scripts into orchestration logic.
- Don’t add external dependencies or network calls without approval.
- Don't change system prompt snippets without approval or explicitly instructed to do so.