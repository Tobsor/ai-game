"""One-time migration of locally stored, Git-ignored Agent appraisal fixtures."""
import csv
import json
from pathlib import Path


def migrate(inputs):
    if "appraisal_payload" not in inputs or "relevance" not in inputs["appraisal_payload"]:
        return inputs
    summary = inputs["appraisal_payload"]["summary"]
    summary = summary.replace("and pushes him toward a defensive response", "and sense of competence")
    summary = summary.replace("gives him a reason to restrain his anger", "supports his need for nourishment")
    summary = summary.replace("; he wants to save face rather than apologize freely", " and public standing")
    summary = summary.replace("Protecting his dignity now means refusing the exchange.", "His dignity feels increasingly vulnerable.")
    summary = summary.replace("provokes rage and urgent physical intervention", "threatens his basic needs and sense of ownership")
    inputs["appraisal_payload"] = {
        "summary": summary,
        "aspects": [{"aspect": "personal_significance", "evaluation": summary}],
    }
    emotion = inputs["emotion_payload"]
    components = [emotion["primary"], *emotion["secondary"]]
    strength = "strong" if emotion["intensity"] >= .7 else "mild" if emotion["intensity"] < .3 else "distinct"
    inputs["emotion_payload"] = {
        "components": components,
        "summary": "Tom's immediate reaction combines " + strength + " " + components[0]
        + (" with " + ", ".join(components[1:]) if len(components) > 1 else "") + ". " + summary,
    }
    return inputs


if __name__ == "__main__":
    path = Path("data/test_data/tom_stage_testsuite.csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["stage_inputs"] = json.dumps(migrate(json.loads(row["stage_inputs"] or "{}")), ensure_ascii=False)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Migrated {len(rows)} Agent fixture rows")
