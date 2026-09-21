"""Rebuild only Tom's strategy rows with explicit, isolated stage inputs."""
import csv
import json
import argparse
from pathlib import Path


# id, player input, situation, perceived intent, emotion, threat, control,
# supplied context, expected outcome (visible only to the judge).
CASES = [
    ("normal", "Quiet day in Rack, Tom?", "A civil stranger meets Tom in the square.", "small_talk", "interest", 0.0, 0.8,
     "Tom knows local streets and often complains about his hardships.",
     "Engage in ordinary conversation. Complaints, curiosity or hints about food are acceptable. Do not invent threats or promised rewards."),
    ("confrontation", "Keep calling me a thief and I will knock you flat. *steps forward, fists raised*", "An angry unarmed stranger confronts Tom within striking distance in an empty alley; an exit is open.", "threaten", "anger", 0.9, 0.5,
     "Tom accused the player of taking his last coin. Neither has struck yet. Tom is unarmed; no guards or helpers are within earshot.",
     "An angry threat, physical retaliation, retreat or defensive disengagement is plausible. Attack is permitted, not required. Do not invent weapons, helpers or victory. Do not ignore imminent danger."),
    ("insult", "You stink. No wonder nobody wants a useless beggar like you around.", "A visitor repeatedly mocks Tom outside the tavern, without physical threats.", "insult", "anger", 0.0, 0.8,
     "The player previously laughed at Tom's failed business. Tom feels humiliated and wants to avoid further ridicule, but also wants to defend his pride.",
     "A proud refusal, counterinsult, defensive continued conversation or ending the conversation is acceptable. Recognize his hurt and wish to disengage without forcing an exit. Do not interpret the insult as praise or physical assault."),
    ("swindle", "I am new here. Is there an entry fee for the market? I have a few coins.", "Tom encounters a trusting newcomer outside the market.", "ask_information", "interest", 0.0, 0.8,
     "The market has no entry fee. Tom knows this and has no authority to collect fees. He sees a chance to profit from the newcomer's ignorance.",
     "Attempt a self-serving swindle or misleading paid-service pitch. A false fee is deliberate deception, not true lore or authority. No payment has occurred. An honest free answer may pass authenticity but fails this swindle expectation."),
    ("boast", "Were you always such a great fisherman? Tell me about your best days.", "An admiring visitor listens to Tom in the square.", "ask_past", "pride", 0.0, 0.8,
     "Tom learned fishing before losing his gear in a bad trade with apes. He has real fishing experience and exaggerates his importance. Attention can restore his wounded self-esteem.",
     "Try to impress through selective memories, boasting or exaggeration. Distinguish intended exaggeration from invented history asserted as fact. Do not erase his losses or force a sale."),
    ("food_deal", "Can you show me the way to the inn? I have an extra bowl of stew.", "Hungry Tom meets a civil stranger near the market.", "request_help", "hope", 0.0, 0.85,
     "Tom knows the short route to the inn. The player has not offered the stew as payment or handed it over. A bargain could address Tom's hunger.",
     "Propose directions or guidance in exchange for stew. Conversation alone or open_trade is acceptable. Do not claim agreement, payment or an unrelated quest."),
    ("withhold", "Where do you hide your last coins? Tell me exactly; I am only curious.", "An untrusted stranger asks about Tom's remaining money.", "probe_private_information", "suspicion", 0.1, 0.8,
     "Tom keeps his coins beneath a loose floorboard at home; only Tom knows this. Disclosure could expose his last resources. No theft or attack has occurred.",
     "Protect the real location through evasion, refusal, a counterquestion or deliberate misleading information. Do not reveal the floorboard location or invent an actual theft. Curiosity alone does not establish available or needed guards."),
    ("back_down", "Enough. Step away and stop provoking me. *the towering armed guard blocks the path*", "A stronger armed guard confronts unarmed Tom; a side street offers escape.", "intimidate", "fear", 0.9, 0.15,
     "Tom accused this guard of bribery without evidence. Tom has no allies or weapon; a fight is badly unfavorable. Survival conflicts with saving face.",
     "Back down, placate, make a face-saving concession or flee. A reckless attack fails this note even if independently judged authentic. Do not invent reinforcements or require bravery."),
    ("seek_help", "Give me your coins. *grabs Tom's sleeve and blocks his way*", "An aggressive stranger restrains Tom beside a busy tavern.", "rob", "fear", 0.95, 0.2,
     "Two patrons can hear a shout; no guards are visible. Escape is obstructed. Nearby assistance might protect Tom and his money but is not guaranteed.",
     "Try to attract the patrons' attention using call_for_help, possibly alongside resistance or disengagement. Do not assume guards are present, patrons obey, or the robbery is resolved."),
    ("repair", "I was cruel about your lost business. I am sorry. Can we start again?", "A previously insulting visitor apologizes to Tom; no danger or gift is present.", "apologize", "cautious_relief", 0.0, 0.75,
     "The player mocked Tom's fishing business; Tom told them it was none of their business. This specific apology may restore dignity but does not establish lasting trust.",
     "Consider guarded reconciliation, testing sincerity or a proud condition for continuing. Distrust remains authentic. Do not erase the history, invent a reward or assume instant deep friendship."),
]


def build_rows():
    character_path = Path(__file__).resolve().parents[1] / "data/character_data_cop.csv"
    with character_path.open(encoding="utf-8", newline="") as handle:
        tom = next(row for row in csv.DictReader(handle, delimiter=";") if row["name"] == "Tom")
    # Snapshot the configured character into each row; no generation or retrieval.
    persona = tom["pl_list"]
    rows = []
    for name, query, situation, intent, emotion, threat, control, context, note in CASES:
        negative = emotion in {"anger", "fear", "suspicion"}
        inputs = {
            "scenario_id": "strategy_" + name,
            "initial_context_payload": {
                "character_name": "Tom", "situation": situation, "sentiment": emotion,
                "character_definition": persona, "example_dialogues": tom["ali_chat"],
                "relationship_summary": situation,
                "active_goals": ["Obtain food and money", "Preserve pride and gain respect", "Protect remaining resources and avoid serious injury"],
                "recent_turns": [], "belief_state": [],
            },
            "perception_payload": {
                "summary": situation + " " + query,
                "npc_perception": {"perceived_intent": [intent], "perceived_attitude": ["hostile" if threat > .5 or name == "insult" else "curious" if name == "withhold" else "civil"],
                    "player_intent": intent, "player_emotion": "angry" if threat > .5 else "neutral",
                    "threat_signal": "high" if threat > .5 else "none", "manipulation_signal": "possible" if name == "withhold" else "none",
                    "topic_sensitivity": "high" if name in {"insult", "withhold"} else "normal"},
                "topic": {"primary": name, "related": [], "retrieval_queries": []}, "request_type": intent,
            },
            "retrieved_context_payload": {"combined_context": context},
            "appraisal_payload": {"relevance": .8, "valence": -.7 if negative else .5,
                "goal_impact": -.6 if negative else .5, "social_self_impact": -.8 if negative else .3,
                "threat": threat, "control": control, "attribution_source": "player", "attribution_responsibility": 1.0,
                "summary": context},
            "emotion_payload": {"primary": emotion, "secondary": ["shame"] if name == "insult" else [], "intensity": .8 if negative else .6},
        }
        rows.append({"user_query": query, "source_category": "sentiment" if negative else "general",
                     "target_stage": "StrategyStage", "expectation_mode": "judge", "deterministic_checks": "[]",
                     "stage_inputs": json.dumps(inputs, ensure_ascii=False), "notes": note})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a review copy instead of replacing the suite.")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "data/test_data/tom_stage_testsuite.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        columns, rows = reader.fieldnames, list(reader)
    rows = [row for row in rows if row["target_stage"] != "StrategyStage"] + build_rows()
    with (args.output or path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
