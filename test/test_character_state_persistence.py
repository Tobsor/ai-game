import unittest
from types import SimpleNamespace

from classes.Character import Character
from logger import configure_logging
from models import Faction, MetadataCategory
from workflow.models import InitialContext, PerceptionResult, ResponseResult, RetrievedContext, StateUpdate, StrategyResult
from workflow.stages.initial_context_stage import InitialContextStage
from workflow.stages.terminal_update_stage import TerminalUpdateStage


class FakeDB:
    def __init__(self):
        self.messages = []
        self.embeddings: list[dict] = []
        self.documents_by_category: dict[str, list[str]] = {}
        self.db = SimpleNamespace(get=self.get_documents)

    def add_embedding(self, id: str, text: str, metadata):
        if hasattr(metadata, "model_dump"):
            payload = metadata.model_dump(mode="json", exclude_none=True)
        else:
            payload = dict(metadata)
        self.embeddings.append({
            "id": id,
            "text": text,
            "metadata": payload,
        })
        category = str(payload.get("category", ""))
        self.documents_by_category.setdefault(category, []).append(text)

    def get_documents(self, where, include, limit=None):
        category = ""
        for item in where.get("$and", []):
            if "category" in item:
                category = str(item["category"])
                break

        documents = list(self.documents_by_category.get(category, []))
        if limit is not None:
            documents = documents[-limit:]

        return {"documents": documents}


class CharacterStatePersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_logging("VERBOSE")

    def create_character(self) -> Character:
        character = Character.__new__(Character)
        character.name = "Mira"
        character.id = "Mira1"
        character.faction = Faction.WORLD
        character.sentiment = "neutral"
        character.db = FakeDB()
        return character

    def test_change_sentiment_persists_sentiment_entry(self):
        character = self.create_character()

        character.change_sentiment("happy", "The player helped with the caravan.", tags=["sentiment", "gratitude"])

        self.assertEqual(character.sentiment, "happy: The player helped with the caravan.")
        self.assertEqual(character.db.embeddings[0]["metadata"]["category"], MetadataCategory.SENTIMENT.value)
        self.assertEqual(character.db.embeddings[0]["metadata"]["name"], "Mira")
        self.assertEqual(character.db.embeddings[0]["metadata"]["tags"], '["sentiment", "gratitude"]')
        self.assertTrue(character.db.embeddings[0]["metadata"]["gratitude"])

    def test_update_methods_persist_changed_state_with_character_tags(self):
        character = self.create_character()

        character.update_relationship(
            StateUpdate(changed=True, value="Trust improved.\nMira trusts the player more."),
            tags=["relationship", "trust"],
        )
        character.update_beliefs(
            StateUpdate(changed=True, value="The player keeps promises.\nThe player is dependable."),
            tags=["belief", "trust"],
        )
        character.update_goals(
            StateUpdate(changed=True, value="Keep the player nearby.\nInvite the player back to the market."),
            tags=["goal", "market"],
        )

        self.assertEqual(len(character.db.embeddings), 3)
        categories = [entry["metadata"]["category"] for entry in character.db.embeddings]
        self.assertIn(MetadataCategory.RELATIONS.value, categories)
        self.assertIn(MetadataCategory.BELIEF.value, categories)
        self.assertIn(MetadataCategory.GOAL.value, categories)
        self.assertTrue(any(entry["metadata"].get("trust") is True for entry in character.db.embeddings))
        self.assertEqual("Trust improved.\nMira trusts the player more.", character.db.embeddings[0]["text"])
        self.assertEqual("The player keeps promises.\nThe player is dependable.", character.db.embeddings[1]["text"])
        self.assertEqual("Keep the player nearby.\nInvite the player back to the market.", character.db.embeddings[2]["text"])

    def test_store_memory_persists_latest_user_and_assistant_turn(self):
        character = self.create_character()
        character.db.messages = [
            {"role": "system", "content": "seed"},
            {"role": "user", "content": "Can you help me with supplies?"},
            {"role": "assistant", "content": "I can offer you rope and herbs."},
        ]

        stored = character.store_memory(tags=["memory", "supplies"])

        self.assertTrue(stored)
        self.assertEqual(character.db.embeddings[0]["metadata"]["category"], MetadataCategory.MEMORY.value)
        self.assertIn("Player: Can you help me with supplies?", character.db.embeddings[0]["text"])
        self.assertIn("Mira: I can offer you rope and herbs.", character.db.embeddings[0]["text"])
        self.assertTrue(character.db.embeddings[0]["metadata"]["supplies"])

    def test_initial_context_stage_reads_persisted_goals_and_beliefs(self):
        character = self.create_character()
        character.situation = "At the market"
        character.pl_list = "Helpful trader"
        character.ali_chat = "Welcome, traveler."
        character.get_relations = lambda: {}
        character.get_sentiment_filter = lambda: {}
        character.db.query_text = lambda *args, **kwargs: ""
        character.documents_by_category = character.db.documents_by_category
        character.db.documents_by_category[MetadataCategory.GOAL.value] = ["Protect the market."]
        character.db.documents_by_category[MetadataCategory.BELIEF.value] = ["The player is reliable."]

        stage = InitialContextStage(character)

        self.assertEqual(stage.get_active_goals(), ["Protect the market."])
        self.assertEqual(stage.get_belief_state(), ["The player is reliable."])

    def test_terminal_update_stage_computes_tags_for_updates(self):
        stage = TerminalUpdateStage(SimpleNamespace(name="Mira"))
        initial_context = InitialContext(
            character_name="Mira",
            situation="At the market",
            sentiment="neutral",
            character_definition="Helpful trader",
            example_dialogues="Welcome, traveler.",
        )
        perception = PerceptionResult(
            raw_prompt="Can I buy herbs from you?",
            player_intent="buy_goods",
            player_emotion="curious",
            request_type="question",
            topic="herbs",
        )
        retrieved_context = RetrievedContext(knowledge_context="Mira sells herbs and travel supplies.")
        strategy = StrategyResult(
            conversation_goal="invite trade",
            conversation_move="answer_then_offer",
            immediate_actions=["open_trade"],
            new_sentiment="happy",
            sentiment_reasoning="The player is a promising customer.",
        )
        response = ResponseResult(reply="I can sell you herbs.", turn_prompt="prompt")

        result = stage.run(initial_context, perception, retrieved_context, strategy, response)

        self.assertIn("sentiment", result.sentiment_tags)
        self.assertIn("buy_goods", result.relationship_update.tags)
        self.assertIn("knowledge_context", result.belief_update.tags)
        self.assertIn("goal", result.goal_update.tags)
        self.assertIn("memory", result.memory_tags)

    def test_terminal_update_stage_update_methods_accept_explicit_tags(self):
        stage = TerminalUpdateStage(SimpleNamespace(name="Mira"))
        initial_context = InitialContext(
            character_name="Mira",
            situation="At the market",
            sentiment="neutral",
            character_definition="Helpful trader",
            example_dialogues="Welcome, traveler.",
        )
        perception = PerceptionResult(raw_prompt="Hello")
        retrieved_context = RetrievedContext()
        response = ResponseResult(reply="Hello there.", turn_prompt="prompt")

        relationship_update = stage.update_relationship(initial_context, perception, response, tags=["relationship", "trust"])
        belief_update = stage.update_beliefs(initial_context, perception, retrieved_context, response, tags=["belief"])
        goal_update = stage.update_goals(initial_context, perception, retrieved_context, response, tags=["goal"])

        self.assertEqual(relationship_update.tags, ["relationship", "trust"])
        self.assertEqual(belief_update.tags, ["belief"])
        self.assertEqual(goal_update.tags, ["goal"])


if __name__ == "__main__":
    unittest.main()
