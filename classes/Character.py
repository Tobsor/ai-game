from classes.ChromaDBHelper import ChromaDBHelper
from classes.NpcAgent import NPCAgent
from ai import AISettings, get_ai_settings
from models import Character as CharacterType, Faction, Metadata, MetadataType, MetadataCategory, CognitiveAction, NPCAction, Sentiment
from typing import Any
from fastapi import WebSocket
from server_models import ChatRequest
import json
from uuid import uuid4
from workflow import TurnInput, TurnPipeline
from workflow.models import InitialContext

from logger import get_logger


def format_prompt(title: str, sections: list[tuple[str, str]]) -> str:
    parts = [title.strip()]

    for heading, content in sections:
        text = content.strip()
        if text == "":
            continue
        parts.append(f"{heading}:\n{text}")

    return "\n\n".join(parts)

annotation_mapping = {
    "job": "Job",
    "past": "Past / Background",
    "loyalty": "Loyalty / Trustworthy",
    "helping": "Helping attitude",
    "dislikes": "Dislikes",
    "likes": "Likes",
    "faction_loyalty": "Faction Loyalty",
    "knowlegdeable": "Knowleagdeable",
    "intelligence": "Intelligence"
}

logger = get_logger(__name__)

class Character:
    id: str
    name: str
    faction: Faction
    pl_list: str
    knowledge: str
    past: str
    ali_chat: str
    relations: str
    situation: str
    sentiment: str
    agent: NPCAgent
    db: ChromaDBHelper

    def __init__(
        self,
        char_data: dict[str | Any, str | Any] | None,
        situation,
        settings: AISettings | None = None,
    ):
        parsed = CharacterType(**char_data) # type: ignore

        self.name = parsed.name
        self.faction = parsed.faction
        self.id = parsed.name + str(parsed.faction)
        self.pl_list = parsed.pl_list
        self.knowledge = parsed.knowledge
        self.past = parsed.past
        self.ali_chat = parsed.ali_chat
        self.relations = parsed.relations
        self.situation = situation
        self.ai_settings = settings or get_ai_settings()
        self.db = ChromaDBHelper(self.ai_settings)
        self.agent = NPCAgent(self.ai_settings)
        self.talk_ongoing = True
        self.pipeline = TurnPipeline(self)

        self.sentiment = self.compute_sentiment()

    async def initiate_conversation(self, socket: WebSocket):
        logger.info("Initiating conversation with %s", self.name)
        self.initialize_message_loop_context()

        greeting_prompt = "Create the opening greeting for the player using the seeded character context."

        greeting = self.db.generate_text(greeting_prompt, stage_name="Greeting")
        await socket.send_json({ "event": "message", "data": greeting })

        logger.verbose("Greeting completed for %s", self.name)

        while self.talk_ongoing:
            user_prompt = None

            try:
                payload = await socket.receive_text()
                logger.info("Received message")
                data = json.loads(payload)
                request = ChatRequest(**data)

                user_prompt = request.prompt
                logger.debug("Request end flag: %s", request.end)
                self.talk_ongoing = request.end != True
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error("Error happened: %s", exc)
                await socket.send_json({"event": "error", "data": f"Invalid request: {exc}"})
                await socket.close(code=1003)
                return

            if(self.talk_ongoing == False):
                logger.info("Ending conversation")
                return

            answer = self.prompt(prompt=user_prompt)
            
            logger.verbose("Response sent to client for %s", self.name)
            await socket.send_json({ "event": "message", "data": answer })

    def compute_sentiment(self):
        sentiment = self.get_sentiment()
        if sentiment.strip() == "":
            return ""

        return sentiment

    def build_system_prompt(self) -> str:
        return "\n".join([
            f"Enter RP mode. You are {self.name}.",
            "Stay in character at all times.",
            "Speak in first person.",
            "Produce only perceivable dialogue or first-person nonverbal actions.",
            "Do not reveal internal thoughts directly.",
            "Maintain character fidelity throughout the conversation.",
        ])

    def build_seed_context_prompt(self, initial_context: InitialContext) -> str:
        return format_prompt(
            "Conversation-start context. Treat the following as the initial state at the beginning of the conversation. Later turns may supersede these details through the unfolding chat.",
            [
                ("Character definition", initial_context.character_definition),
                ("Example dialogues", initial_context.example_dialogues),
                ("Initial situation", initial_context.situation),
                ("Initial sentiment towards player", initial_context.sentiment),
                ("Initial relationship summary", initial_context.relationship_summary),
                ("Initial long-term goals", "\n".join(initial_context.active_goals)),
                ("Initial beliefs", "\n".join(initial_context.belief_state)),
            ],
        )

    def build_initial_context(self) -> InitialContext:
        return self.pipeline.initial_context_stage.run(TurnInput(prompt=""))

    def initialize_message_loop_context(self) -> None:
        initial_context = self.build_initial_context()
        self.db.seed_response_context(
            system_prompt=self.build_system_prompt(),
            seed_context_prompt=self.build_seed_context_prompt(initial_context),
        )
    
    def get_sentiment_filter(self):
        return self.get_character_category_filter(MetadataCategory.SENTIMENT)

    def get_character_category_filter(self, category: MetadataCategory):
        return {
            "$and": [
                {
                    "name": self.name,
                },
                {
                    "type": MetadataType.CHARACTER.value,
                },
                {
                    "category": category.value,
                },
            ]
        }

    def get_sentiment(self):
        documents = self.get_character_documents(MetadataCategory.SENTIMENT, limit=3)
        sentiment_entries = [str(doc).strip() for doc in reversed(documents) if str(doc).strip()]
        return "\n".join(sentiment_entries)

    def get_character_documents(self, category: MetadataCategory, limit: int | None = None) -> list[str]:
        kwargs: dict[str, Any] = {
            "where": self.get_character_category_filter(category),
            "include": ["documents"],
        }
        if limit is not None:
            kwargs["limit"] = limit

        results = self.db.db.get(**kwargs)
        documents = results.get("documents") or []
        if not isinstance(documents, list):
            return []

        return [str(doc).strip() for doc in documents if str(doc).strip()]
    
    def get_memories(self):
        return self.get_character_category_filter(MetadataCategory.MEMORY)
    
    def get_past(self):
        return self.get_character_category_filter(MetadataCategory.PAST)
    
    def get_faction_knowledge(self):
        return {
            "$and" : [
                {
                    "faction": self.faction.value,
                },
                {
                    "type": MetadataCategory.LORE.value
                }
            ]
        }


    def get_world_knowledge(self):
        return {
            "$and" : [
                {
                    "faction": Faction.WORLD.value,
                },
                {
                    "category": MetadataCategory.LORE.value
                }
            ]
        }
    
    def get_relations(self):
        return self.get_character_category_filter(MetadataCategory.RELATIONS)

    def get_goals(self):
        return self.get_character_category_filter(MetadataCategory.GOAL)

    def get_beliefs(self):
        return self.get_character_category_filter(MetadataCategory.BELIEF)
    
    # Agent like behavior:
    # - remember: Query past related information
    # - recall: Query knowledge bases, what the character knows
    # - social_interaction: Query social / relationship related info
    # - introspect: Query reflection information (personal goals, fears, feelings)
    # - plan_task: Query game related information (quest, happenings)
    # - clarify: Push back to the user, clarify
    # - deceive: try to deceive the user
    # - threat_assess: threatening the user

    def cognitive_action(self, actions: list[CognitiveAction], reasoning: str):
        """
        Tool function: When the character needs to perform a cognitive action such as remembering, recalling knowledge or interact socially to respond to the user appropriately

        Args:
            actions: A list of cognitive actions. Cognitive actions can be: remember (the npcs needs to remember past events), recall knowledge (the npc must consult his knowledge), social interaction
            reasoning: A short reasoning why the npc needs to perform this cognitive actions to answer the user

        Returns:
            An $or metadata construction including all filters to be applied when quering documents on chromadb
        """

        logger.verbose("Invoked cognitive action with: %s", actions)
        if isinstance(actions, list) == False:
            logger.error("Invalid args provided to cognitive action: %s", actions)
            return None

        all_filters = []

        for action in actions:
            try:
                CognitiveAction(action)
            except ValueError:
                logger.error("Invalid cognitive action detected: %s", action)
                continue

            match action:
                case CognitiveAction.REMEMBER.value:
                    """ The character attempts to remember past personal information """
                    all_filters.append(self.get_memories())
                    all_filters.append(self.get_past())

                case CognitiveAction.RESEARCH.value:
                    """ The character attempts to recall general information """
                    all_filters.append(self.get_faction_knowledge())
                    all_filters.append(self.get_world_knowledge())

                case CognitiveAction.RECALLKNOWLEDGE.value:
                    """ The character attempts to recall general information """
                    all_filters.append(self.get_faction_knowledge())
                    all_filters.append(self.get_world_knowledge())

                case CognitiveAction.SOCIAL.value:
                    """ The character looks up information to engage socially into the conversation with the user """
                    all_filters.append(self.get_relations())
                    all_filters.append(self.get_sentiment_filter())

        return {
            "$or": all_filters
        }
    
    def generate_npc_intention(self, intention: list[str], reasoning: str):
        """
        Tool function: A function that generates a context string representing the intention of a npc with which he responds to the user
        
        Args:
            intention: A list of intentions which as a sum descrive the npc's intention
            reasoning: A short explanation on what intention the npc follows when responding.
        """
        logger.verbose("Invoked npc intention with: %s", intention)
        if isinstance(intention, list) == False:
            logger.error("Malformed intention: %s", intention)
            return ""
        
        return str(intention) + ": " + reasoning
    
    def immediate_actions(self, action: NPCAction):
        """
        Tool function: When the character takes an immediate action as a consequence of the user prompt

        Args:
            action: An immediate action that the NPC takes after the current interaction. Supported actions are: end_conversation.

        Returns:
            Boolean whether the NPC continues the conversation with the user or not
        """
        logger.verbose("Invoked immediate action with: %s", action)
        try:
            NPCAction(action)
        except ValueError:
            logger.error("Invalid action value detected: %s", action)
            return NPCAction.KEEP_TALKING
    
        self.talk_ongoing = action != NPCAction.END_CONVERSATION
    
    def change_sentiment(self, new_sentiment: str, reasoning: str, tags: list[str] | None = None):
        """
        Tool function: When the character experiences a change in sentiment as consequence of the user prompt

        Args:
            new_sentiment: A short explanation on how the character now feels after that user interaction
            reasoning: A short explanation on why the new state was selected and how the character now feels
        """
        logger.verbose("Invoked new sentiment with: %s", new_sentiment)
        try: 
            Sentiment(new_sentiment)
        except ValueError:
            logger.error("Invalid sentiment value: %s", new_sentiment)
            return
        
        self.sentiment = new_sentiment + ": " + reasoning
        self.add_character_state_embedding(
            category=MetadataCategory.SENTIMENT,
            text=self.sentiment,
            tags=tags,
        )

    def prompt(self, prompt: str):
        if(prompt.strip() == ""):
            return ""

        self.initialize_message_loop_context()
        result = self.pipeline.run(TurnInput(prompt=prompt))
        self.apply_turn_updates(result.terminal_update)

        return result.response.reply

    def apply_turn_updates(self, terminal_update):
        logger.verbose("Applying terminal updates")

        if terminal_update.sentiment is not None:
            self.change_sentiment(
                new_sentiment=terminal_update.sentiment,
                reasoning=terminal_update.sentiment_reasoning,
                tags=terminal_update.sentiment_tags,
            )

        self.immediate_actions(terminal_update.immediate_actions)
        self.update_relationship(terminal_update.relationship_update, tags=terminal_update.relationship_update.tags)
        self.update_beliefs(terminal_update.belief_update, tags=terminal_update.belief_update.tags)
        self.update_goals(terminal_update.goal_update, tags=terminal_update.goal_update.tags)

        if terminal_update.store_memory:
            self.store_memory(tags=terminal_update.memory_tags)

        self.trigger_external_actions(terminal_update.external_actions)

    def update_relationship(self, relationship_update, tags: list[str] | None = None):
        self.persist_state_update(MetadataCategory.RELATIONS, relationship_update, tags=tags)
        return relationship_update

    def update_beliefs(self, belief_update, tags: list[str] | None = None):
        self.persist_state_update(MetadataCategory.BELIEF, belief_update, tags=tags)
        return belief_update

    def update_goals(self, goal_update, tags: list[str] | None = None):
        self.persist_state_update(MetadataCategory.GOAL, goal_update, tags=tags)
        return goal_update

    def store_memory(self, tags: list[str] | None = None):
        memory_entry = self.build_latest_memory_entry()
        if memory_entry == "":
            return False

        self.add_character_state_embedding(
            category=MetadataCategory.MEMORY,
            text=memory_entry,
            tags=tags,
        )
        return True

    def trigger_external_actions(self, actions: list[str]):
        # TODO: Hand off non-dialogue world actions to the game simulation layer.
        return actions

    def persist_state_update(self, category: MetadataCategory, state_update, tags: list[str] | None = None) -> None:
        if getattr(state_update, "changed", False) != True:
            return

        entry = str(getattr(state_update, "value", "")).strip()
        if entry == "":
            return

        self.add_character_state_embedding(
            category=category,
            text=entry,
            tags=tags if tags is not None else getattr(state_update, "tags", []),
        )

    def build_latest_memory_entry(self) -> str:
        last_user_message = ""
        last_assistant_message = ""

        for message in reversed(self.db.messages):
            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if content == "":
                continue

            if role == "assistant" and last_assistant_message == "":
                last_assistant_message = content
                continue

            if role == "user" and last_user_message == "":
                last_user_message = content

            if last_user_message != "" and last_assistant_message != "":
                break

        if last_user_message == "" and last_assistant_message == "":
            return ""

        memory_parts = [
            f"Player: {last_user_message}" if last_user_message != "" else "",
            f"{self.name}: {last_assistant_message}" if last_assistant_message != "" else "",
        ]
        return "\n".join(part for part in memory_parts if part != "")

    def add_character_state_embedding(self, category: MetadataCategory, text: str, tags: list[str] | None = None) -> None:
        if text.strip() == "":
            return

        self.db.add_embedding(
            id=self.create_state_embedding_id(category),
            text=text.strip(),
            metadata=self.build_character_embedding_metadata(category=category, tags=tags),
        )

    def create_state_embedding_id(self, category: MetadataCategory) -> str:
        return f"runtime-{self.id}-{category.value}-{uuid4().hex}"

    def build_character_embedding_metadata(self, category: MetadataCategory, tags: list[str] | None = None) -> dict[str, Any]:
        metadata = Metadata(
            faction=self.faction,
            type=MetadataType.CHARACTER,
            category=category,
            name=self.name,
        ).model_dump(mode="json", exclude_none=True)
        normalized_tags = self.normalize_tags(tags or [])
        if len(normalized_tags) > 0:
            metadata["tags"] = json.dumps(normalized_tags, ensure_ascii=True)
            for tag in normalized_tags:
                metadata[tag] = True
        return metadata

    def normalize_tags(self, tags: list[str]) -> list[str]:
        normalized_tags: list[str] = []
        seen_tags: set[str] = set()
        for tag in tags:
            normalized = str(tag).strip()
            if normalized == "" or normalized in seen_tags:
                continue
            seen_tags.add(normalized)
            normalized_tags.append(normalized)
        return normalized_tags
