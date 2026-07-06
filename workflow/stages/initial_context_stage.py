from logger import get_logger
from models import MetadataCategory
from workflow.models import InitialContext, TurnInput
from workflow.stages.base import Stage

logger = get_logger(__name__)


class InitialContextStage(Stage):
    MAX_RECENT_MESSAGES = 6
    RAW_RECENT_TURN_CHAR_LIMIT = 500

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
        if hasattr(self.character, "get_character_documents"):
            stored_goals = self.character.get_character_documents(MetadataCategory.GOAL)
            if len(stored_goals) > 0:
                return stored_goals

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
        recent_messages = self.get_recent_conversation_messages()
        if len(recent_messages) == 0:
            return []

        recent_excerpt = self.format_recent_messages(recent_messages)
        if not self.should_summarize_recent_messages(recent_messages, recent_excerpt):
            return [recent_excerpt]

        summary = self.summarize_recent_turns(recent_excerpt)
        if summary == "":
            return [recent_excerpt]

        return [summary]

    def get_belief_state(self) -> list[str]:
        if hasattr(self.character, "get_character_documents"):
            return self.character.get_character_documents(MetadataCategory.BELIEF)
        return []

    def get_recent_conversation_messages(self) -> list[dict[str, str]]:
        messages = getattr(self.character.db, "messages", [])
        if not isinstance(messages, list):
            return []

        recent_messages: list[dict[str, str]] = []
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant"} or content == "":
                continue
            if content.startswith("Conversation-start context."):
                continue

            recent_messages.append({"role": role, "content": content})
            if len(recent_messages) >= self.MAX_RECENT_MESSAGES:
                break

        recent_messages.reverse()
        return recent_messages

    def format_recent_messages(self, recent_messages: list[dict[str, str]]) -> str:
        formatted_lines: list[str] = []
        for message in recent_messages:
            role = message["role"]
            speaker = "Player" if role == "user" else self.character.name
            formatted_lines.append(f"{speaker}: {self.trim_recent_message(message['content'])}")

        return "Recent conversation excerpt:\n" + "\n".join(formatted_lines)

    def trim_recent_message(self, content: str, max_length: int = 220) -> str:
        trimmed = " ".join(content.split())
        if len(trimmed) <= max_length:
            return trimmed
        return trimmed[: max_length - 3].rstrip() + "..."

    def should_summarize_recent_messages(self, recent_messages: list[dict[str, str]], recent_excerpt: str) -> bool:
        if len(recent_messages) > 4:
            return True

        return len(recent_excerpt) > self.RAW_RECENT_TURN_CHAR_LIMIT

    def summarize_recent_turns(self, recent_excerpt: str) -> str:
        response = self.character.agent.run_prompt(
            prompt="\n".join([
                "Summarize only the immediate conversational state from the last few turns.",
                "Include only these four headings:",
                "Topic:",
                "Open loops:",
                "Momentum:",
                "Commitments:",
                "Focus on the current topic, unresolved questions or requests, emotional or social direction, and any recent promises, refusals, offers, threats, or instructions.",
                "Do not include long-term relationship summaries, stable beliefs, long-term goals, world lore, or facts not directly present in these turns.",
                "Keep the total output compact and concrete.",
                recent_excerpt,
            ]),
            stage_name="InitialContextStage.recent_turns",
            payload={"recent_excerpt": recent_excerpt},
        )
        return response.content.strip()
