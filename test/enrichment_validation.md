# Stage-local enrichment migration validation

Offline verification completed on 2026-09-24. No live LLM quality scores were produced.
Scripted decision-provider outputs exercised the actual pipeline, stage parsers, shared
enrichment helper, action selection and evaluation runner. No unit test suite was added.

Verified behaviors:

- Three eligible stages can each retrieve and revise once; a multi-question string causes
  one backend call. Requests emitted during revision cannot trigger another retrieval.
- No-gap drafts require no retrieval or revision.
- Empty results and timeouts add no evidence; request history suppresses equivalent failed requests.
- Malformed gap fields preserve valid cognition. Invalid initial cognition fails normally.
- Invalid revisions preserve the original perception, coupled appraisal/emotion, or strategy
  and action set. Successfully fetched evidence remains available downstream.
- A repeated stage invocation cannot enrich twice against the same turn context.
- Duplicate evidence IDs are neither appended again nor used to trigger another revision.
- Downstream prompts receive earlier evidence; upstream results remain fixed. Only the
  finalized strategy's actions reach Response and Terminal Update.
- All 10 tracked roleplay scenarios execute through the evaluator with scripted outputs;
  controlled evidence and error fixtures exercise revision/fallback without a live backend.
- All 67 migrated local CSV scenarios validate; cognitive fixtures execute offline.
- The deterministic greeting smoke check can address an indexed enrichment-history entry.

A separate in-memory Chroma check used real filtering and vector queries with fixed local
embeddings. It returned Tom's character records, another faction's lore and untyped shared
information, while excluding another character's private records. No persistent knowledge
store was changed by this check.

Syntax/import checks and whitespace checks also passed. Runtime verification establishes
mechanical bounds and contracts, not whether a live model consistently recognizes material
gaps, avoids paraphrased duplicates, or preserves character subjectivity.

For live quality evaluation with configured providers:

```shell
python -m test.TestRunner --character Tom --suite test/enrichment_scenarios.json --stage "All stages"
```

The scenario evidence is controlled by default. The decision model and judge are live in
that command; `live_enrichment: true` additionally opts a scenario into real retrieval.
The local CSV under `data/test_data` remains ignored by Git according to repository policy;
the new JSON scenario suite is tracked independently of it.

After simplifying `EnrichmentDecision` to stage/request/reason, offline scripted checks
were rerun on 2026-09-25 for no-fetch decisions, successful retrieval/revision, stage limits,
duplicate requests, empty/failed retrieval, evidence deduplication,
and revision failure fallback. Schema and JSON fixture validation passed. Request correlation
IDs were removed; database record IDs remain for evidence deduplication. The shared call
counter and configurable limit were subsequently removed; the fixed flow permits at most
one retrieval and revision per eligible stage. No live model evaluation was run.
