# Tom strategy test concept

Ten StrategyStage rows live alongside appraisal cases in `data/test_data/tom_stage_testsuite.csv`.
The semicolon CSV keeps the existing columns. `stage_inputs` is a JSON object serialized
into one CSV cell; use a CSV writer to escape quotes, rather than hand-escaping JSON.
Additional enrichment scenarios are in `test/enrichment_scenarios.json`.

Each row contains `scenario_id` and five objects: `initial_context_payload`,
`perception_payload`, `retrieved_context_payload`, `appraisal_payload`, and
`emotion_payload`. Their fields match the dataclasses in `workflow/models.py`;
appraisal uses prose `summary` and dynamic `aspects`; emotion uses `components` and an authoritative prose `summary`.
The perception fixture includes `raw_prompt` and `core`; its raw prompt must exactly match the CSV's `user_query`. All nested fields are validated strictly by the shared perception validator. Strategy sees the complete core and raw prompt, with shared evidence.
An empty retrieved-context object explicitly means no recalled information.
Missing or incorrectly typed payloads fail before strategy generation. No context
builder, perception or appraisal stage runs for these rows. Strategy owns its optional enrichment;
controlled `enrichment_evidence` is empty by default, and `live_enrichment: true` opts into the backend.

The cases cover small talk, confrontation with possible violence, an insult and
possible exit, swindling, boasting, a food bargain, withholding private information,
backing down, seeking help, and relationship repair. Inputs describe the scene and
prior mental state; only `notes` specifies expected strategy outcomes. Notes are
visible to the judge, never to the strategy model. A proposed deal is not a completed
transaction, a threat is not an attack, and seeking help does not guarantee assistance.

The existing semantic metrics plus context-enrichment evaluation use the existing LLM judge and 0–1 scoring (pass >= 0.5).
Results remain separate per metric, with explanations; there is no aggregate score
that hides an author-note failure behind a good character-fit score. Author-note
restrictions apply only to author-note agreement. Missing notes fail that metric.
The accompanying `strategy_metric_review.csv` records the metric audit and decisions.
`strategy_validation.csv` records implementation validation separately from live model scores.
Fictional deception or aggression can be authentic; evaluate context and proportionality
instead of rewarding unconditional friendliness. These ten cases are a qualitative
baseline, not a statistical estimate of model reliability.

Roleplay evaluation covers action selection, fixture isolation, enrichment necessity,
evidence reuse and the quality of the finalized strategy. Contextual action validity
belongs to the judge. Unknown action calls remain ignored by runtime; if no recognized
action is selected, the existing keep-talking fallback applies. During enrichment,
draft actions remain unexecuted and the final revision selects the complete action set.

New attack_player, flee and call_for_help tools expose selectable actions and return
action metadata. TODOs mark unimplemented combat, movement and other-NPC reactions.
The strategy prompt includes fixture character definition, situation and beliefs so
the model can see the same scene the judge evaluates.

Run the configured live suite with `python -m test.TestRunner` in an interactive terminal
and select Tom, then StrategyStage. The entrypoint also accepts --character, --stage and --suite.
Like appraisal testing, the runner exports actual outputs and inputs to
`test/results/tom_stage_testsuite_prompts.csv` and per-metric judgments to
`test/results/tom_stage_testsuite_results.csv`. These character-wide exports are
overwritten by each run, including a stage-filtered run. Offline tests mock the exporter
and judge; they do not manufacture a live quality-results CSV.
