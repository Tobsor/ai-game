"""Explicit Word -> CSV -> optional Chroma synchronization command."""
import argparse
import json
from pathlib import Path

from data_ingestion import DATA_ROOT, csv_text, discover_sources, export_csvs


def run(*, data_root=DATA_ROOT, check=False, sync_embeddings=False, db_reset=False):
    if check and db_reset:
        raise ValueError("--check cannot be combined with --db-reset")
    data_root = Path(data_root)
    rows = discover_sources(data_root / "sources")
    preview = {}
    for name, character in (("character_data.csv", True), ("lore_data.csv", False)):
        records = [r for r in rows if (r["entity_type"] == "character") == character]
        path = data_root / name
        content = csv_text(records).encode("utf-8")
        preview[name] = {"rows": len(records), "changed": not path.exists() or path.read_bytes() != content}
    print(json.dumps(preview, indent=2))
    if check:
        return preview
    if sync_embeddings or db_reset:
        from logger import configure_logging
        configure_logging()
        from embedding_sync import synchronize
        result = synchronize(rows, db_reset=db_reset)
        print(json.dumps(result, indent=2))
    export_csvs(rows, data_root)
    return preview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate and preview CSV changes without writing or calling AI")
    mode.add_argument("--sync-embeddings", action="store_true", help="Also synchronize authored Chroma records")
    parser.add_argument("--db-reset", action="store_true", help="Clear the configured collection and freshly embed only current Word sources; implies --sync-embeddings")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args(argv)
    if args.db_reset and args.check:
        parser.error("--db-reset cannot be combined with --check")
    try:
        run(data_root=args.data_root, check=args.check, sync_embeddings=args.sync_embeddings, db_reset=args.db_reset)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Import failed: {exc}\n")


if __name__ == "__main__":
    main()
