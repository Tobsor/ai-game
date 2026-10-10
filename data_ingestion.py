"""Deterministic Word source discovery and the shared long-form CSV contract."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET

from models import Character, Faction, MetadataCategory

APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APP_ROOT / "data"
FIELDS = ("record_id", "entity_id", "entity_type", "name", "faction", "category", "subcategory", "value", "source_path")
CHARACTER_FIELDS = ("pl_list", "ali_chat", "knowledge", "past", "relations", "sentiment")
CATEGORIES = {category.value for category in MetadataCategory} | {"pl_list", "ali_chat"}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_docx(path: Path) -> str:
    """Read visible body text; refuse constructs whose contents would be lost."""
    try:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
            body = root.find(W + "body")
            if body is None:
                raise ValueError("missing document body")
            unsupported = {"txbxContent", "ins", "del", "moveFrom", "moveTo", "altChunk",
                           "drawing", "pict", "object", "footnoteReference", "endnoteReference",
                           "fldChar", "fldSimple"}
            for node in root.iter():
                local = node.tag.split("}")[-1]
                if local in unsupported or local.endswith("PrChange"):
                    raise ValueError(f"unsupported Word content: {local}; convert it to plain body text")
            def paragraph(node):
                parts = []
                for child in node.iter():
                    if child.tag == W + "t":
                        parts.append(child.text or "")
                    elif child.tag == W + "tab":
                        parts.append("\t")
                    elif child.tag in {W + "br", W + "cr"}:
                        parts.append("\n")
                return "".join(parts)

            def blocks(parent):
                result = []
                for child in parent:
                    if child.tag == W + "p":
                        result.append(paragraph(child))
                    elif child.tag == W + "tbl":
                        for row in child.findall(W + "tr"):
                            result.append("\t".join("\n".join(blocks(cell)) for cell in row.findall(W + "tc")))
                    elif child.tag == W + "sdt":
                        content = child.find(W + "sdtContent")
                        if content is not None:
                            result.extend(blocks(content))
                    elif child.tag not in {W + "sectPr", W + "tcPr", W + "bookmarkStart", W + "bookmarkEnd"}:
                        if any((n.text or "").strip() for n in child.iter(W + "t")):
                            raise ValueError(f"unsupported body container: {child.tag}")
                return result

            value = "\n".join(blocks(body)).strip("\n")
            return value
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def warn_skipped(location: str, reason: str) -> None:
    print(f"WARNING: Skipping {location}: {reason}", file=sys.stderr)


def validate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter unusable rows; content and cross-row consistency belong to the maintainer."""
    valid = []
    factions = {faction.value for faction in Faction}
    for index, row in enumerate(rows, 1):
        location = row.get("source_path") or f"CSV row {index}"
        reason = ""
        if any(not isinstance(row.get(field), str) or not row[field].strip()
               for field in ("entity_id", "entity_type", "name")):
            reason = "missing ownership information (entity_id, entity_type, or name)"
        elif row["entity_type"] in {"character", "faction"} and not row.get("faction"):
            reason = "missing faction for character/faction ownership"
        elif row.get("faction") and (not isinstance(row["faction"], str) or row["faction"] not in factions):
            reason = f"unknown faction {row['faction']!r}"
        elif not isinstance(row.get("category"), str) or row["category"] not in (CATEGORIES if row["entity_type"] == "character" else {"lore"}):
            reason = f"unknown or unsupported category {row.get('category')!r}"
        elif not isinstance(row.get("value"), str) or not row["value"].strip():
            reason = "empty information value"
        if reason:
            warn_skipped(location, reason)
            continue
        valid.append({**row, "subcategory": row.get("subcategory") or ""})
    return valid


def discover_sources(root: Path) -> list[dict[str, str]]:
    """Collect one row per Word file using its nearest entity.json and category folder."""
    if not root.is_dir():
        raise ValueError(f"Source directory not found: {root}; see docs/data-authoring.md")
    rows = []
    manifests = {}
    documents = [path for path in sorted(root.rglob("*"))
                 if path.is_file() and path.suffix.lower() in {".docx", ".doc", ".docm"}
                 and not path.name.startswith("~$")]
    if not documents:
        raise ValueError(f"{root}: no source documents; refusing to erase existing data")
    for path in documents:
        if path.suffix.lower() != ".docx":
            warn_skipped(str(path), "save as .docx before importing")
            continue
        manifest = next((parent / "entity.json" for parent in path.parents
                         if (parent == root or root in parent.parents)
                         and (parent / "entity.json").is_file()), None)
        if manifest is None:
            warn_skipped(str(path), "no owning entity.json")
            continue
        if manifest not in manifests:
            try:
                entity = json.loads(manifest.read_text(encoding="utf-8-sig"))
                if not isinstance(entity, dict):
                    raise ValueError("expected a JSON object")
                manifests[manifest] = (entity, None)
            except (OSError, ValueError) as exc:
                manifests[manifest] = ({}, str(exc))
        entity, error = manifests[manifest]
        if error:
            warn_skipped(str(path), f"cannot read ownership from {manifest}: {error}")
            continue
        relative = path.relative_to(manifest.parent)
        try:
            value = read_docx(path)
        except ValueError as exc:
            warn_skipped(str(path), str(exc))
            continue
        entity_id = entity.get("entity_id", "")
        rows.append(dict(record_id=digest(str(entity_id) + ":" + relative.as_posix()),
                         entity_id=entity_id, entity_type=entity.get("type", ""), name=entity.get("name", ""),
                         faction=entity.get("faction", "1" if entity.get("type") == "world" else ""),
                         category=relative.parts[0] if len(relative.parts) > 1 else "",
                         subcategory=relative.parts[1] if len(relative.parts) > 2 else "",
                         value=value, source_path=path.relative_to(root).as_posix()))
    return validate_rows(rows)


def csv_text(rows: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter=";", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def export_csvs(rows: list[dict[str, str]], data_root: Path) -> dict[str, bool]:
    rows = validate_rows(rows)
    outputs = {"character_data.csv": [r for r in rows if r["entity_type"] == "character"],
               "lore_data.csv": [r for r in rows if r["entity_type"] != "character"]}
    data_root.mkdir(parents=True, exist_ok=True)
    staged = []
    changed = {}
    previous = {}
    replaced = []
    try:
        for name, records in outputs.items():
            path = data_root / name
            content = csv_text(records).encode("utf-8")
            previous[path] = path.read_bytes() if path.exists() else None
            changed[name] = previous[path] != content
            if not changed[name]:
                continue
            with tempfile.NamedTemporaryFile(dir=data_root, delete=False) as file:
                file.write(content)
                staged.append((Path(file.name), path))
        for temporary, path in staged:
            os.replace(temporary, path)
            replaced.append(path)
    except Exception:
        for path in replaced:
            if previous[path] is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous[path])
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return changed


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        if reader.fieldnames not in (list(FIELDS), [field for field in FIELDS if field != "subcategory"]):
            raise ValueError(f"{path}: expected long-form CSV; run sync_data.py")
        rows = list(reader)
    return validate_rows(rows)


def load_characters(path: Path = DATA_ROOT / "character_data.csv") -> list[dict[str, str]]:
    grouped = {}
    for row in sorted(read_rows(path), key=lambda r: r["source_path"]):
        if row["entity_type"] != "character":
            raise ValueError(f"{path}: expected character records only")
        character = grouped.setdefault(row["entity_id"], {"name": row["name"], "faction": row["faction"], **dict.fromkeys(CHARACTER_FIELDS, "")})
        if row["category"] in CHARACTER_FIELDS:
            field = row["category"]
            character[field] += ("\n\n" if character[field] else "") + row["value"]
    for character in grouped.values():
        Character(**character)
    return list(grouped.values())
