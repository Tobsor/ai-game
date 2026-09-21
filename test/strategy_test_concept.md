# Tom strategy test concept

Ten StrategyStage rows live alongside appraisal cases in `data/test_data/tom_stage_testsuite.csv`.
The semicolon CSV keeps the existing columns. `stage_inputs` is a JSON object serialized
into one CSV cell; use a CSV writer to escape quotes, rather than hand-escaping JSON.
`test/build_strategy_fixtures.py` documents and reproducibly builds these rows.

Each row contains `scenario_id` and five objects: `initial_context_payload`,
`perception_payload`, `retrieved_context_payload`, `appraisal_payload`, and
`emotion_payload`. Their fields match the dataclasses in `workflow/models.py`;
appraisal uses flat `attribution_source` and `attribution_responsibility` fields.
The perception's raw prompt is always taken from the CSV's `user_query`.
An empty retrieved-context object explicitly means no recalled information.
Missing or incorrectly typed payloads fail before strategy generation. No context
builder, perception, gap analysis, retrieval or appraisal stage runs for these rows.
Other stage runners retain their existing behavior.

The cases cover small talk, confrontation with possible violence, an insult and
possible exit, swindling, boasting, a food bargain, withholding private information,
backing down, seeking help, and relationship repair. Inputs describe the scene and
prior mental state; only `notes` specifies expected strategy outcomes. Notes are
visible to the judge, never to the strategy model. A proposed deal is not a completed
transaction, a threat is not an attack, and seeking help does not guarantee assistance.

All six semantic metrics use the existing LLM judge and 0–1 scoring (pass >= 0.5).
Results remain separate per metric, with explanations; there is no aggregate score
that hides an author-note failure behind a good character-fit score. Author-note
restrictions apply only to author-note agreement. Missing notes fail that metric.
The accompanying `strategy_metric_review.csv` records the metric audit and decisions.
`strategy_validation.csv` records implementation validation separately from live model scores.
Fictional deception or aggression can be authentic; evaluate context and proportionality
instead of rewarding unconditional friendliness. These ten cases are a qualitative
baseline, not a statistical estimate of model reliability.

Deterministic unit tests cover tool availability, invocation-to-action mapping,
multiple calls, the existing no-call/unknown-call keep-talking fallback, fixture
isolation, missing-input failure and results export. They do not decide whether Tom
ought to attack or leave. All six live quality metrics remain judge-based; no hybrid
evaluation mode or duplicated model generations are needed. Unknown calls are still
ignored by runtime and duplicates are preserved; these tests do not imply stronger
runtime validation. Contextual action validity belongs to the judge.

New attack_player, flee and call_for_help tools expose selectable actions and return
action metadata. TODOs mark unimplemented combat, movement and other-NPC reactions.
The strategy prompt includes fixture character definition, situation and beliefs so
the model can see the same scene the judge evaluates.

Run offline checks with `python -m unittest test.test_strategy_stage_test test.test_agent_test`.
Run the configured live suite with `python -m test.TestRunner` in an interactive terminal
and select Tom, then StrategyStage. The current entrypoint does not parse CLI flags.
Like appraisal testing, the runner exports actual outputs and inputs to
`test/results/tom_stage_testsuite_prompts.csv` and per-metric judgments to
`test/results/tom_stage_testsuite_results.csv`. These character-wide exports are
overwritten by each run, including a stage-filtered run. Offline tests mock the exporter
and judge; they do not manufacture a live quality-results CSV.
