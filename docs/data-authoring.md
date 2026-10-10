# Maintaining Word sources

Edit files under `data/sources`, then run from the repository:

```powershell
python sync_data.py --check
python sync_data.py
# Stop the NPC service before synchronizing the database:
python sync_data.py --sync-embeddings
```

The first command validates and previews without writing or contacting an AI provider.
The second only exports CSV. The third also updates the configured local Chroma collection
using the existing embedding provider. Restart the NPC service to load updated character fields.
Source/output paths default to this repository regardless of the working directory; Chroma uses
the existing AI settings. `--data-root` selects an alternative source/output directory.

## Layout

```text
data/sources/
  characters/tom/
    entity.json
    pl_list/personality.docx
    pl_list/conversational.docx
    ali_chat/examples.docx
    knowledge/fishing.docx
    knowledge/player-rumors.docx
    past/early-life.docx
    relations/mentor.docx
    goal/short-term.docx
    sentiment/initial-attitude.docx
  factions/raccoon/
    entity.json
    lore/history.docx
  world/arim/
    entity.json
    lore/seasons.docx
  locations/rack/
    entity.json
    lore/village.docx
```

Create an entity folder by copying a manifest from `docs/templates` as `entity.json`.
Copy `docs/templates/information.docx` into a category folder to start a new information piece.
Keep entity IDs stable. The maintainer is responsible for consistent names, identities,
and faction assignments across records; the importer does not enforce cross-row consistency.
Character and faction entities require one of the existing faction values: `raccoon`,
`jellyfish`, `ants`, `apes`, `chameleon`, `fennek`, or `1` (world).
World entities default to faction `1`. Other named entities can use a descriptive type,
such as `location` or `item`, and an optional valid faction. They are shared lore; they do
not create new playable characters or new enum factions.

Character categories: `pl_list`, `ali_chat`, `knowledge`, `past`, `relations`, `goal`,
`belief`, `sentiment`, `memory`, and `lore`. Use `pl_list` for the character definition;
missing direct-context fields default to empty strings. All shared entities use `lore`.
Subfolders beneath a category are allowed. Top-level grouping folders are organizational;
the manifest defines entity type and identity.

The first folder beneath a category supplies an optional, free-text `subcategory`:
`goal/short-term/find-food.docx` produces category `goal` and subcategory `short-term`.
Likewise, use `goal/mid-term/business.docx`, `goal/long-term/fame.docx`, or
`past/childhood/school.docx`. Deeper folders are organizational and do not change the
subcategory. Files directly inside the category (such as `goal/goals.docx`) have an empty
subcategory; filenames do not infer one. Existing documents are not automatically split.

One Word file is one information piece / CSV row. Write ordinary prose, headings, or lists;
no prescribed headings or AI extraction are required. Split long topics into separate files
when you want distinct pieces. Renaming a file replaces its record identity. Multiple files
per category are supported. Direct character fields are joined in source-path order;
prefix filenames with numbers if order matters.

Only `.docx` is supported. Body paragraphs and table rows stay in document order. Table cells
are separated with tabs and paragraphs/rows with newlines. List text is preserved; automatic
bullets/numbers and other visual formatting are not stored. Text boxes, images, embedded
objects, fields, footnotes/endnotes, and tracked changes are rejected.
Headers and footers are ignored; maintained information is expected in plain body text.
Resolve revisions and move information into ordinary body text before importing. Word's
`~$` lock files are ignored. Unreadable/unsupported documents are skipped with a warning.

## CSV and embeddings

`data/character_data.csv` and `data/lore_data.csv` share this header:

```text
record_id;entity_id;entity_type;name;faction;category;subcategory;value;source_path
```

Files use UTF-8, semicolons, and CSV quoting for multiline text. Treat them as generated output;
edits will be replaced. Older CSVs without `subcategory` remain readable and default to empty.
Each unusable row produces a warning and is skipped; valid rows
continue to CSV export and embedding. Only ownership information, known faction/category
values, and nonempty information values are checked. Repeated entity/category entries
(such as several Tom/past records) and even identical rows are allowed. Repeated record IDs
receive separate embedding IDs, so different values cannot overwrite each other.
An empty Word document is skipped, not treated as a fatal error.
An empty source tree is refused to guard against an accidentally wrong path.
Skipped records are excluded from the desired embedding set; a sync removes their previous
importer-owned chunks just as it removes deleted documents. If all documents are skipped,
the generated CSVs contain only headers and authored embeddings are removed; runtime records stay.

`pl_list` and `ali_chat` load directly into NPC context; all other character categories and
shared lore are embedded. Chunks remain 100 words with 50-word overlap, independently per
file. Metadata retains `name`, `type`, `faction`, and `category`, plus provenance and fingerprints.
Other characters' explicit records remain private. Shared entities use `type=lore`, factions
use `type=faction`, and world lore retains faction `1`. Retrieval thresholds are unchanged.

Nonempty subcategories are copied to Chroma metadata on every chunk; empty ones are omitted.
Category-only retrieval still returns all subcategories. To filter character records, use
`get_character_documents(MetadataCategory.GOAL, subcategory="short-term")` or pass the optional
subcategory to `get_character_category_filter`. Shared lore can use the same `subcategory`
key in Chroma filters. No cognitive stage automatically prioritizes a particular subcategory.

Goals and sentiment in Word now become available to existing initial-context retrieval;
the old importer omitted them. Different document boundaries can change retrieval rankings.
The loader supplies the existing character model; cognitive prompts are unchanged.

Synchronization skips unchanged vectors, upserts changes, and removes stale importer-owned
chunks only after preparation/upserts succeed. Runtime state and unrelated records are retained.
No snapshots or automatic rollback are created. Write failures propagate to the caller;
completed writes remain, and rerunning synchronization finishes the authored updates.
Chroma has no multi-record transaction: stop the app and avoid concurrent importer runs.
A process crash can require rerunning synchronization or rebuilding from the maintained sources.

To discard the configured collection and freshly embed only current Word sources, run:

```powershell
python sync_data.py --db-reset
```

`--db-reset` implies `--sync-embeddings` and cannot be combined with `--check`. It replaces
`--rebuild-embeddings`. Stop the NPC service first. The reset affects only the configured
collection, not other collections in the Chroma directory.

Without reset, incompatible stored embedding configuration or vector dimensions are fatal.
With reset, old metadata, vectors, and documents are not read or reused. Current model and
metric settings determine the replacement collection. All previous records, including runtime
memories and unrelated records in that collection, are discarded. Source documents remain.

Fresh vectors are generated and validated before deleting the collection. Invalid vector
values or inconsistent dimensions remain fatal even in reset mode. Source-row warnings and
skipping still work as documented above. Provider, filesystem, and database failures also
remain fatal; reset bypasses old-state compatibility checks, not operational errors.
If deletion/writing fails, rerun with `--db-reset`; there is no rollback or backup.
Even an input with no embeddable rows clears the selected collection on reset.

The result reports `reset=true`; `deleted` is null because reset does not inspect old records
to count them. An incremental run reports the number of stale records deleted as before.

Both old embedding entrypoints and `setup_infrastructure.py` synchronize the complete authored
set through this implementation. Do not use them with an incomplete source subset.

## Completed migration and active sources

The one-time migration has been completed and its script retired. Original Word files and
CSV data were retained, with backups under `data/backups`. CSV-only information was converted
into Word documents during that migration; normal imports only read Word files.

Inspect `data/migration_report.json` for differences, fallbacks, and unused legacy files.
Synchronization no longer recognizes or cleans up old migration seeds. It only removes
obsolete records marked as belonging to the current importer.
Source documents, CSVs, reports, and backups remain local under the existing Git
ignore rules. Code, documentation, and reusable templates are tracked.

The active authoring tree is **only `data/sources`**. The old `data/character_data` and
`data/faction_data` directories were intentionally retained as pre-migration originals.
Editing those originals does not update the generated CSVs or embeddings. Manifests live
inside each entity folder, for example `data/sources/characters/tom/entity.json`.
For a new entity, copy the supplied manifest and Word templates into this active tree.

## Descriptive metadata in embedded text

Every authored chunk and runtime document is embedded and stored with a readable prefix:

```text
Name: Tom
Type: character
Faction: raccoon
Category: goal
Subcategory: long-term

Tom wants to become the world's most famous fisherman.
```

Nonempty fields appear in this order: name, entity type (or type), faction, category,
subcategory, and tags. Existing JSON-encoded runtime tags become a comma-separated list.
IDs, paths, importer markers, fingerprints, and Boolean tag copies stay in structured metadata
only. A document without descriptive metadata retains its original text.

The importer chunks raw text first and prefixes each resulting chunk. Word files and CSV values
remain unchanged. Retrieval, including direct goal lookups, returns the prefixed text. Queries
are embedded without a document prefix. Callers of document-ingestion methods must supply raw
text; formatting occurs once at the storage boundary. Technical-only metadata changes can
reuse a vector, but descriptive changes regenerate it. Tags inside Word prose are not parsed.

The prefix format is versioned in the embedding configuration fingerprint. Before using this
format with an existing old-format collection, stop the service and run:

```powershell
python sync_data.py --db-reset
```

This is an explicit migration: it regenerates authored records and discards runtime records.
No automatic reset is performed. Retrieval thresholds, access rules, and cognitive prompts
are unchanged.

## Verification

Run `python -B -m test.verify_embedding_prefix` for prefix integration scenarios using a real
isolated Chroma collection and deterministic local embeddings. It verifies authored/runtime/
fixture formatting, chunk prefixes, raw query handling, retrieval, idempotence, vector reuse,
and the explicit reset requirement. It does not contact an LLM or change the maintained database.
