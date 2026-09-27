This repo implements an **NPC roleplay/chat agent** that:
- Builds RP prompts from a `Character` definition (pl_list, example dialogues, situation)
- Retrieves **context** from a Chroma-backed knowledge store (mini RAG)
- Generates a final NPC response via a text generation backend (`ChromaDBHelper.generate_text`)

The core flow lives in `Character.prompt()` and `Character.initiate_conversation()`.

---

# Architecture Snapshot

## File / folder structure (architectural context)
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

- `/workflow`
  - Contains all script related to the internal thinking workflow of the NPC for the answer computation
    - `/pipeline.py`: The orchestrator defining the sequential process of all relevant stages
    - `/stages/*`: All stages as self contained python classes

- `/test`
  - Test files that attempt to verify the **quality of NPC roleplaying**.
  - Includes automated verification via an AI-judge model

- `/ai`
  - Contains operability code for LLM Provider and adapters

## Top-level scripts (entrypoints and data pipelines)
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

# General NPC conversation loop

The rough process looks like this:

1. **User selects a character** to talk to (selection menu / entry script).
2. **NPC starts with a greeting**.
3. **User prompts the NPC freely** (no strict format required).
4. **NPC computes an answer**
5. **NPC decides whether to continue** the conversation.

---

# Answer computation in detail
The NPC runs a specific routine to simulate thinking and decision process to generate an authentic answer (#4 in General NPC conversation loop).

1. Initial Context (`/workflow/stages/initial_context_stage.py`): Loads all relevant context data in preparation of the NPC simulation. This includes static data (e.g. pl_list, ali_chat examples) as well as dynamic data (relationship data, recent conversation history). This stage operates unrelated to the incoming player input.

2. Perception interprets the player input subjectively. It may identify a material gap,
   retrieve once and revise once through `workflow/enrichment.py`.
3. Appraisal evaluates personal significance and NPC emotion, with the same optional enrichment.
   Finalized perception stays fixed, including plausible mistakes.
4. Strategy selects conversational goals and immediate actions, with the same optional enrichment.
   Finalized perception, appraisal and emotion stay fixed.
5. Response realizes the strategy in character; it never retrieves.
6. Terminal Update persists relevant state as before.

`ContextEnrichment`, `GapAnalysis` (shared prompt/validation), and `ContextRetriever` are
reusable helpers, not pipeline stages. A stage emits its normal cognitive JSON plus
`context_request` (one concise string, empty for no gap) and `gap_reason` (a brief explanation).
The helper separates these fields from the cognitive result. Each eligible stage gets
at most one backend request and one revision. No retries or query splitting occur.
Initial context loading is separate from stage enrichment.
Token consumption is guided by prompts, not new hard token caps.

`RetrievedContext` carries initial context, shared evidence with record IDs/metadata,
and stage/request/reason history. Failed retrieval adds no evidence; prior requests remain
in history to prevent repeats. All later stages reuse it. Request matching is
conservative and deterministic; the prompts additionally require semantic reuse.
Every record is accessible except explicit character records belonging to another NPC.
Failed enrichment preserves the valid draft. Conflicting later evidence may cause confusion
without reopening upstream reasoning. Trace events belong to the requesting stage.

## Perception contract

Finalized Perception returns exactly `raw_prompt` and `core`, validated by
`workflow/perception_contract.py` for model results and fixed inputs. The shared helper
extracts the enrichment decision before cognitive validation.
`core` contains one-sentence situation, intent and player-attitude interpretations;
one interaction type and topic; perceived player emotion (primary, low/medium/high intensity,
and summary); and threat/manipulation levels (none/low/medium/high).
Uncertainty belongs in the summaries. Player emotion is distinct from appraisal's NPC emotion.
All cognitive fields are required; unknown fields, invalid types/values, and mismatched player
text fail explicitly. Gap errors preserve a valid cognitive draft. Downstream stages consume
the complete core and original player input without retrieval hints.

# Knowledge Retrieval Rules (Chroma Metadata Filters)

The agent uses metadata filters with `$and` / `$or` to retrieve context.

(POC note: the specific enum mappings and exact filter composition are still in flux. At a high level, “fetching knowledge” means creating the appropriate metadata filters/tags so Chroma returns the right slices of lore/memories/relations/etc.)

---

# Prompting Conventions (RP)

## Greeting prompt
The greeting prompt should:
- Force first-person RP as the character
- Include situation + character definition + example dialogues (if available)
- Instruct the model to initiate the interaction in-character

## Answer prompt
The answer prompt should enforce:
- first-person RP
- include non-verbal actions as `*...*`
- do not reveal NPC thoughts
- output only dialogue-perceivable content

When editing prompt text:
- Keep instructions explicit and near the bottom
- Avoid ambiguous constraints like “be concise” unless required
- Prefer stable headings because retrieval quality tends to benefit from consistent structure
- Be concise, prefer short and pointed instructions over descriptive and detailed instructions.

---

# Developing / Extending Tools

If/when you add new “agent behaviors” (even if not formalized as tools yet), keep these principles:

- Validate model outputs defensively (LLM-driven inputs can be malformed)
- Prefer small, reversible changes (agent behavior is prompt-sensitive)

---

# What Codex Should Do in This Repo

When making changes, Codex should:
1. Preserve the existing high-level flow (See Answer computation in detail)
2. Keep RP constraints intact (first-person, `*...*`, no hidden thoughts)
3. Avoid large refactors unless explicitly requested
4. Prefer improving clarity/robustness of retrieval filters/tags rather than inventing parallel retrieval systems

If asked to implement new agent behaviors:
- Prefer extending behavior in `/classes` close to where prompts and retrieval are orchestrated
- Add/adjust tests in `/test` to prevent regressions in RP quality

**Do**
- Do preserve prompt section headings and ordering unless explicitly asked.
- Do keep RP rules explicit and near the bottom of prompt templates.
- Do edit behavior in `/classes` and keep entry scripts thin.
- Do prefer small, reversible changes; describe intent in code comments only when needed.
- Do always ask for additional approval in cases an action would be advisable but contradict the rules stated in the file.

**Don’t**
- Don’t change the prompt workflow without explicit approval.
- Don’t change RP output format (first‑person, `*...*`, no internal thoughts).
- Don’t “simplify” prompts by removing constraints or examples.
- Don’t refactor entry scripts into orchestration logic.
- Don’t add external dependencies or network calls without approval.
- Don't change system prompt snippets without approval or explicitly instructed to do so.
- Don't add unit testing.
