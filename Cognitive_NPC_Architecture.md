# Cognitive NPC Architecture

```text
┌─────────────────┐
│ Player Message  │
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│ Initial Context          │
│ ------------------------ │
│ • Character Sheet        │
│ • Current Scene          │
│ • Mood                   │
│ • Relationship Summary   │
│ • Active Goals           │
│ • Recent Turns           │
└────────┬─────────────────┘
         │
         ▼

┌──────────────────────────────────────────────────────────────┐
│ Context-Aware Interpretation                                │
│ (Max 1 optional retrieval iteration per turn)               │
│                                                              │
│  ┌─────────────────────┐                                    │
│  │ Mandatory Perception│                                    │
│  ├─────────────────────┤                                    │
│  │ Interpret Intent    │                                    │
│  │ Detect Emotion      │                                    │
│  │ Detect Request Type │                                    │
│  │ Topic / Ambiguity   │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ Context Gap Analyzer│                                    │
│  ├─────────────────────┤                                    │
│  │ Need Memory?        │                                    │
│  │ Need Relationship?  │                                    │
│  │ Need Knowledge?     │                                    │
│  │ Need Social Context?│                                    │
│  │ Need Threat?        │                                    │
│  │ Need Manipulation?  │                                    │
│  │ Need Sensitivity?   │                                    │
│  └──────┬───────┬──────┘                                    │
│         │       │                                           │
│         │       └──────────────► Continue                   │
│         │                         (No gaps)                │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────┐                                    │
│  │ Optional Agent Tools│                                    │
│  ├─────────────────────┤                                    │
│  │ Recall Memory       │                                    │
│  │ Relationship Recall │                                    │
│  │ Knowledge Recall    │                                    │
│  │ Social Context Eval │                                    │
│  │ Threat Analysis     │                                    │
│  │ Manipulation Eval   │                                    │
│  │ Topic Sensitivity   │                                    │
│  │ Trust/Respect Impact│                                    │
│  │ Fear/Suspicion      │                                    │
│  │ Internal Conflict   │                                    │
│  │ Belief Updates      │                                    │
│  │ Goal Updates        │                                    │
│  │ Planning            │                                    │
│  └─────────┬───────────┘                                    │
│            │                                                │
│            └───── reinterpret once ─────► Perception        │
│                                                              │
│  NOTE: After one retrieval pass, continue.                  │
│        No repeated memory requests.                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼

┌──────────────────────────┐
│ Motivation & Strategy    │
├──────────────────────────┤
│ Evaluate Emotion         │
│ Select Conversation Goal │
│ Assess Risk              │
│ Decide Disclosure        │
│ Select Social Strategy   │
│ Select Tone              │
│ Select Verbosity         │
│ Select Conversation Move │
└──────────┬───────────────┘
           │
           ▼

┌──────────────────────────┐
│ Response                 │
├──────────────────────────┤
│ Generate NPC Reply       │
└──────────┬───────────────┘
           │
           ▼

┌──────────────────────────┐
│ Terminal State Updates   │
├──────────────────────────┤
│ Update Emotional State   │
│ Update Relationship      │
│ Update Beliefs           │
│ Update Goals             │
│ Store Memory             │
│ External Actions         │
└──────────┬───────────────┘
           │
           ▼

┌──────────────────────────┐
│ Final State              │
├──────────────────────────┤
│ Persisted NPC State      │
│ World State              │
│ Future Context           │
└──────────────────────────┘


Circular Dependency Cluster
---------------------------

Intent              ↔ Memory
Intent              ↔ Social Context
Beliefs             ↔ Motive Inference
Emotion             ↔ Goal Selection
Risk Assessment     ↔ Disclosure
Memory Storage      ↔ Future Recall

Feedback Across Turns
---------------------

Final State ─────► Initial Context (next interaction)
```
