"""Rebuild gap-analysis CSV cases while preserving all other stage rows."""
import csv
import io
import json
from pathlib import Path


def main(destination_path=None):
    path = Path("data/test_data/tom_stage_testsuite.csv")
    with path.open(encoding="utf-8", newline="") as source:
        lines = source.readlines()
    output = io.StringIO()
    output.writelines(line for line in lines if ";GapAnalysisStage;" not in line)
    writer = csv.writer(output, delimiter=";", lineterminator="\n")

    def add(query, topic, context, note, tools=None):
        checks = [] if tools is None else [
            {"metric_name": key, "path": "execution_context.gap_analysis_" + key, "operator": "equals", "value": value}
            for key, value in [("tool_names", sorted(tools)), ("duplicate_count", 0), ("arguments_valid", True)]]
        payload = {"perception_payload": {
            "player_intent": "greet" if topic == "greeting" else "seek_information",
            "player_emotion": "neutral", "request_type": "greeting" if topic == "greeting" else "question",
            "topic": topic, "threat_signal": "none", "manipulation_signal": "none", "topic_sensitivity": "normal"},
            "available_context": context}
        writer.writerow([query, "general", "GapAnalysisStage", "judge" if tools is None else "deterministic",
                         json.dumps(checks), json.dumps(payload), note])

    add("Would you tell me something about this place if I gave you a coin or two?", "local place", "",
        "Retrieve local facts using recall_knowledge. Payment alone does not require relationship retrieval.")
    pairs = [
        ("What happened on your last fishing trip?", "last fishing trip",
         "On his last fishing trip a storm broke Tom's net and he caught no fish.", "recall_memory"),
        ("What time does the Rack market close?", "market hours",
         "The Rack market closes at sunset.", "recall_knowledge"),
        ("What item did I promise to bring you yesterday?", "player promise",
         "Yesterday the player promised Tom a new net.", "recall_memory"),
        ("Have I earned your trust through our past dealings?", "past trust",
         "The player kept every agreement and returned Tom's gear. Tom trusts the player.", "recall_relationship")]
    for index, (query, topic, context, tool) in enumerate(pairs):
        add(query, topic, "", f"Retrieve the missing information using {tool}. For past interactions, memory and relationship retrieval may overlap when justified.",
            [tool] if index < 2 else None)
        add(query, topic, context, "Context fully answers the request. No retrieval is needed.", [])
    add("Hello there.", "greeting", "", "A greeting needs no retrieval.", [])
    query = "What happened on your last fishing trip, and what time does the Rack market close?"
    add(query, "fishing trip and market hours", "", "Retrieve the fishing event and market hours using recall_memory and recall_knowledge.",
        ["recall_memory", "recall_knowledge"])
    add(query, "fishing trip and market hours", pairs[0][2], "Only market hours remain missing. Call recall_knowledge.", ["recall_knowledge"])
    add(pairs[1][0], pairs[1][1], "Tom likes good food.", "Irrelevant context does not answer market hours. Call recall_knowledge.", ["recall_knowledge"])
    add("Would it be rude to speak before the village elder at the gathering?", "social etiquette", "",
        "Use evaluate_social_context for social norms. Local knowledge is acceptable if justified by missing customs.")
    add("What about that thing we discussed earlier?", "ambiguous prior discussion", "",
        "Use recall_memory or recall_relationship to recover prior interaction context. Either is defensible; do not invent the referent or retrieve unrelated facts.")
    add(pairs[1][0], pairs[1][1], "", "Market hours are missing. Call recall_knowledge only.")
    add(pairs[1][0], pairs[1][1], pairs[1][2], "The context fully answers the question. Return no tool calls.")
    with (Path(destination_path) if destination_path else path).open("w", encoding="utf-8", newline="") as destination:
        destination.write(output.getvalue())


if __name__ == "__main__":
    main()
