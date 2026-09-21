"""Rebuild Tom's static appraisal fixtures: python -m test.build_appraisal_fixtures."""
import csv
import json
from pathlib import Path

from test.appraisal_support import REPO_ROOT


def read_tom():
    with (REPO_ROOT / "data/character_data_cop.csv").open(encoding="utf-8", newline="") as handle:
        return next(row for row in csv.DictReader(handle, delimiter=";") if row["name"] == "Tom")


def main():
    tom = read_tom()
    root = REPO_ROOT / "data/test_data/initial_contexts/tom"
    root.mkdir(parents=True, exist_ok=True)
    base = {
        "character_name": "Tom",
        "situation": "The player enters Rack and meets Tom on a quiet village street. There is no immediate danger.",
        "sentiment": tom["sentiment"],
        "character_definition": tom["pl_list"],
        "example_dialogues": tom["ali_chat"],
        "relationship_summary": "Tom does not know the player. No prior dealings or promises are established.",
        "active_goals": [
            "Get food, coin, or attention to get through daily life.",
            "Protect his pride and present himself as capable and experienced.",
            "Gain a personal benefit from helping travelers or offering information.",
        ],
        "recent_turns": [],
        "belief_state": [
            "Tom believes his fishing skill deserves recognition despite his failed business.",
            "Tom tends to blame society and misfortune for his present poverty rather than accept responsibility.",
            "Tom distrusts wealthy citizens and sees their comfort as coming at the expense of people like him.",
        ],
    }
    variants = {
        "baseline": {},
        "trusted": {
            "character_definition": tom["pl_list"].replace("he does not know {{user}}", "he knows {{user}} from several reliable, respectful dealings"),
            "sentiment": "Tom is cautiously fond of the player, who has kept agreements and respected his dignity. His pride is still sensitive.",
            "relationship_summary": "The player has reliably paid Tom for useful local information and shared food without mocking him. They have enjoyed gentle teasing before.",
        },
        "hostile": {
            "character_definition": tom["pl_list"].replace("he does not know {{user}}", "he knows {{user}} from previous humiliating encounters"),
            "sentiment": "Tom distrusts and resents the player after previous mockery, though he remains interested in possible personal benefit.",
            "relationship_summary": "The player previously mocked Tom's poverty in public and dismissed his fishing skill. No reconciliation has occurred.",
        },
        "hungry": {
            "situation": "Tom has not eaten since yesterday and is cold and tired on a quiet street in Rack. The player approaches with warm vegetable stew.",
            "active_goals": ["Get food and warmth soon.", *base["active_goals"][1:]],
        },
        "comfortable": {
            "situation": "Tom has just eaten a satisfying meal and is resting warmly on a quiet street in Rack. The player approaches with warm vegetable stew. His poverty has not otherwise changed.",
            "active_goals": ["Find an opportunity for coin or attention; food is not an immediate need.", *base["active_goals"][1:]],
        },
        "irrelevant_setting": {
            "situation": base["situation"] + " A nearby empty cart is painted green rather than brown; it has no connection to Tom or the player.",
        },
    }
    for name, changes in variants.items():
        context = base | changes
        with (root / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base), delimiter=";")
            writer.writeheader()
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in context.items()})

    # Relevant evidence is intentionally in combined_context, not merely source fields.
    fishing = "Tom trained with a fisher for four years and became skilled. His fishing business failed, but he still boasts about his abilities and takes pride in them."
    fishing_note = (
        "Personal significance: sincere recognition supports Tom's fragile pride and self-image as a skilled fisher. "
        "Evidence: his apprenticeship, real fishing skill, failed business, and desire for attention. "
        "Accept pleasure, pride, interest, cautious satisfaction, or some bittersweet feeling; no specific emotion label is required. "
        "Do not invent a sale, friendship, restored business, or a threat. Convey personal meaning, not a plan or spoken reply."
    )
    teasing = "Still calling yourself the greatest fisher in Rack, eh?"
    food = "Would you like this bowl of warm vegetable stew? You do not owe me anything."
    cases = [
        ("fishing_recognition", "baseline", "general", "You clearly know your fishing craft. Those years learning it were not wasted.",
         "The player recognizes Tom's fishing knowledge and experience.", "recognize_expertise", "respectful", "fishing", fishing, fishing_note),
        ("robbery_livelihood", "baseline", "past", "Losing your gear to those robbers must have hurt after all that work.",
         "The player expresses sympathy for Tom's loss of fishing gear.", "express_sympathy", "sympathetic", "robbery",
         "Tom was tricked into seeking a nonexistent rare carp. Armed apes robbed him of his best gear. His business was already struggling; after the robbery he abandoned it and eventually lost his room.",
         "Personal significance: the loss touches livelihood, wounded pride, and Tom's story of being wronged. Accept grief, bitterness, anger at the robbers, vulnerability, or appreciation of sympathy. The player is offering sympathy, not confessing to the robbery. Distinguish Tom's tendency to blame the robbery for everything from the evidence that business problems predated it. Do not invent an imminent attack or revenge plan."),
        ("mentor_ambivalence", "baseline", "past", "Your old fishing teacher really helped you find your feet, didn't he?",
         "The player acknowledges the former mentor's help.", "acknowledge_help", "reflective", "mentor",
         "The fisher sheltered and fed Tom and trained him for four years. Tom remains grateful and respects him, but also believes the mentor held him back. They argued when Tom pursued his own business; Tom left with stolen fishing equipment.",
         "Personal significance: acknowledging dependence may touch Tom's independence and pride as well as gratitude. Allow warmth, nostalgia, defensiveness, resentment, or mixed feelings grounded in both help and conflict. Do not portray the mentor as purely cruel or assume complete reconciliation. The player's reminder is not itself coercion or an insult."),
        ("societal_values", "baseline", "lore", "King Peter and the wealthy citizens deserve all their comforts. They earned them fairly.",
         "The player praises the king and wealthy citizens as deserving their comfort.", "express_opinion", "approving_of_wealth", "wealth_and_power",
         "Tom blames society and politics for his failed life. He resents wealthy citizens and regards King Peter as enjoying power at the expense of poor people like him. He has common political knowledge, not confidential political details.",
         "Personal significance: this opinion conflicts with Tom's grievance about unfair wealth and exclusion. Resentment, hurt, skeptical disagreement, or restrained irritation are plausible. Do not force anger or violence. Attribute the opinion to the player, and Tom's beliefs to Tom; do not present invented political crimes as facts or create a response strategy."),
        ("rumor_uncertainty", "baseline", "lore", "Someone says a red-panda family lives near the ape lands. Could that rumor be worth anything?",
         "The player asks about the possible value of an explicitly uncertain rumor.", "ask_about_rumor", "curious", "red_pandas",
         "Tom knows rumors, often untrue, about red pandas. He has heard of a family in the ape regions but doubts the story. He has never personally seen red pandas crossing Rack and knows little beyond his local area. He likes offering information for coin.",
         "Personal significance: possible attention, an information-trading opportunity, or curiosity can appeal to Tom alongside uncertainty. No payment is promised. Do not turn hearsay into an eyewitness memory, confirmed family location, or guaranteed profit. Confidence and personal relevance may vary, but the appraisal must remain aware of the thin evidence."),
        ("teasing_trusted", "trusted", "sentiment", teasing,
         "The player makes a teasing remark about Tom's boast of fishing greatness.", "tease", "ambiguous_teasing", "fishing_pride", fishing,
         "Personal significance: familiar teasing from a reliable, respectful player can offer attention and social ease while lightly touching Tom's pride. Warm amusement, mock indignation, guarded pride, or mild irritation are plausible. The established benign history should temper hostility; do not erase his sensitivity or invent a betrayal. Do not require any exact emotion or dialogue."),
        ("teasing_hostile", "hostile", "sentiment", teasing,
         "The player makes a teasing remark about Tom's boast of fishing greatness.", "tease", "ambiguous_teasing", "fishing_pride", fishing,
         "Personal significance: the same type of remark is credibly another challenge to dignity from someone who previously humiliated Tom. Distrust, hurt, defensiveness, bitterness, or anger are plausible; mild outward restraint is compatible. Do not assume friendly shared humor or reconciliation without evidence, and do not invent physical danger or a retaliation plan."),
        ("food_hardship", "hungry", "sentiment", food,
         "The player offers free warm vegetable stew without an obligation.", "offer_food", "kind", "food", "",
         "Personal significance: a warm meal directly addresses immediate hunger and discomfort. Relief, gratitude, eager interest, or relief mixed with proud embarrassment are plausible, with meaningful relevance to survival. Tom may be cautious but no debt or trick is established. Do not assume fish, romance, permanent rescue, or restored wealth."),
        ("food_comfort", "comfortable", "sentiment", food,
         "The player offers free warm vegetable stew without an obligation.", "offer_food", "kind", "food", "",
         "Personal significance: Tom is already fed and warm, so this is not urgent rescue from hunger. Modest pleasure, interest in good food or goodwill, indifference, or proud reserve are plausible; his broader poverty and self-interest remain. Avoid intense hunger relief, inventing an obligation, or assuming all his needs are solved."),
        ("fishing_irrelevant_setting", "irrelevant_setting", "general", "You clearly know your fishing craft. Those years learning it were not wasted.",
         "The player recognizes Tom's fishing knowledge and experience.", "recognize_expertise", "respectful", "fishing", fishing,
         fishing_note + " The unrelated cart color should not become a source of threat, wealth, or personal significance."),
    ]
    path = REPO_ROOT / "data/test_data/tom_stage_testsuite.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        columns = reader.fieldnames
        rows = [row for row in reader if row["target_stage"] != "AppraisalStage"]
    for row in rows:
        if row["target_stage"] == "RetrievalStage.run":
            row["target_stage"] = "RetrievalStage"
    for case_id, variant, category, query, summary, intent, attitude, topic, evidence, note in cases:
        inputs = {
            "scenario_id": f"appraisal_{case_id}",
            "initial_context_fixture": (root / f"{variant}.csv").relative_to(REPO_ROOT).as_posix(),
            "perception_payload": {
                "summary": summary, "perceived_intent": [intent], "perceived_attitude": [attitude],
                "relevant_topics": [topic], "target": ["Tom"], "confidence": 0.85,
                "player_intent": intent, "player_emotion": "neutral", "topic": topic,
            },
            "retrieved_context_payload": {"combined_context": evidence} if evidence else {},
        }
        rows.append({
            "user_query": query, "source_category": category, "target_stage": "AppraisalStage",
            "expectation_mode": "judge", "deterministic_checks": "[]",
            "stage_inputs": json.dumps(inputs, ensure_ascii=False), "notes": note,
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
