from logger import get_logger
from workflow.models import InitialContext, TurnInput
from workflow.stages.base import Stage

logger = get_logger(__name__)


class InitialContextStage(Stage):
    def run(self, turn_input: TurnInput) -> InitialContext:
        logger.verbose("Building initial context for %s", self.character.name)
        return InitialContext(
            character_name=self.character.name,
            situation=self.character.situation,
            sentiment=self.character.sentiment,
            character_definition=self.character.pl_list,
            example_dialogues=self.character.ali_chat,
            relationship_summary=self.build_relationship_summary(),
            active_goals=self.get_active_goals(),
            recent_turns=self.get_recent_turns(),
            belief_state=self.get_belief_state(),
        )

    def build_relationship_summary(self) -> str:
        prompt = (
            f"What is {self.character.name}'s relationship to the player? "
            "Recall relevant relationship history, social context, and current sentiment."
        )
        filter_value = {
            "$or": [
                self.character.get_relations(),
                self.character.get_sentiment_filter(),
            ]
        }
        relation_summary = self.character.db.query_text(
            prompt=prompt,
            filter=filter_value,
            stage_name="InitialContextStage",
        ).strip()
        if relation_summary == "":
            return ""

        return "Relationship to player:\n" + relation_summary

    def get_active_goals(self) -> list[str]:
        prompt = (
            f"Summarize {self.character.name}'s core values, morality, short term goals, "
            "mid term goals, and long term goals based only on the retrieved character knowledge. "
            "Return a compact plain-text summary."
        )
        filter_value = {
            "$and": [
                {
                    "name": self.character.name,
                },
                {
                    "type": "character",
                },
                {
                    "category": "knowledge",
                },
            ]
        }
        goal_summary = self.character.db.query_text(
            prompt=prompt,
            filter=filter_value,
            stage_name="InitialContextStage",
        ).strip()
        if goal_summary == "":
            return []

        return [goal_summary]

    def get_recent_turns(self) -> list[str]:
        # TODO: Summarize recent turns into a compact cross-turn context block.
        return []

    def get_belief_state(self) -> list[str]:
        # TODO: Load durable NPC beliefs for downstream reasoning.
        return []
