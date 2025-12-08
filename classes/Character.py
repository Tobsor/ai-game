from classes.ChromaDBHelper import ChromaDBHelper
from models import Character as CharacterType, Faction, MetadataType, MetadataCategory
from typing import Any

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

class Character:
    id: str
    name: str
    faction: Faction
    pl_list: str
    ali_chat: str
    situation: str
    init_sentiment: str
    db: ChromaDBHelper

    def __init__(self, char_data: dict[str | Any, str | Any] | None, situation):
        parsed = CharacterType(**char_data) # type: ignore

        self.name = parsed.name
        self.faction = parsed.faction
        self.id = parsed.name + str(parsed.faction)
        self.pl_list = parsed.pl_list
        self.ali_chat = parsed.ali_chat
        self.situation = situation
        self.db = ChromaDBHelper()

        self.init_sentiment = self.get_sentiment()

    def initiate_conversion(self):
        print("initiating conversation...")
        greeting_prompt = f"""
            Enter RP mode. You are {self.name}. Stay in character at all times, speaking in first person as {self.name}:

            Situation:
            {self.situation}

            Sentiment towards player:
            {self.init_sentiment}

            Follow this character definition:
            {self.pl_list}

            Example dialogues:
            {self.ali_chat}

            You shall initiate the first greeting.
            <|model|>{{model's response goes here}}
        """

        response = self.db.generate_text(greeting_prompt)

        print(response)

    def create_answer_prompt(self, prompt: str, sentiment: str, context: str):
        return f"""
            Enter RP mode. You are {self.name}. Stay in character at all times, speaking in first person as {self.name}:

            Situation:
            {self.situation}

            General context:
            {context}

            Sentiment towards player:
            {sentiment}

            Follow this character definition:
            {self.pl_list}

            Example dialogues:
            {self.ali_chat}

            You shall reply to the user while staying in character.
            <|user|>{prompt}
            <|model|>{{model's response goes here}}
        """

    def get_sentiment(self):
        prompt = "How does {{char}} feel about {{user}}? What is {{cahr}} sentiment towards {{user}}?"
        filter = {
           "$and" : [
                {
                    "name": self.name,
                },
                {
                    "type": MetadataType.CHARACTER.value,
                },
                {
                    "category": MetadataCategory.SENTIMENT.value
                }
            ]
        }

        return self.db.query_text(prompt=prompt, filter=filter)
    
    def get_memories(self):
        return {
            "$and" : [
                {
                    "name": self.name,
                },
                {
                    "type": MetadataType.CHARACTER.value,
                },
                {
                    "category": MetadataCategory.MEMORY.value
                }
            ]
        }
    
    def get_past(self):
        return {
            "$and" : [
                {
                    "name": self.name,
                },
                {
                    "type": MetadataType.CHARACTER.value,
                },
                {
                    "category": MetadataCategory.PAST.value
                }
            ]
        }
    
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
        return {
            "$and" : [
                {
                    "char_name": self.name,
                },
                {
                    "type": MetadataType.CHARACTER.value,
                },
                {
                    "category": MetadataCategory.RELATIONS.value
                }
            ]
        }

    
    def get_context(self, prompt: str):
        memories_query = self.get_memories()
        past_query = self.get_past()
        faction_query = self.get_faction_knowledge()
        world_query = self.get_world_knowledge()
        relation_query = self.get_relations()

        query_filter = { 
            "$or": [
                memories_query,
                past_query,
                faction_query,
                world_query,
                relation_query
            ]
        }

        return self.db.query_text(prompt=prompt, filter=query_filter)
    
    # Agent like behavior:
    # - remember: Query past related information
    # - recall: Query knowledge bases, what the character knows
    # - social_interaction: Query social / relationship related info
    # - introspect: Query reflection information (personal goals, fears, feelings)
    # - plan_task: Query game related information (quest, happenings)
    # - clarify: Push back to the user, clarify
        
    def prompt(self, prompt: str) -> str:
        if(prompt.strip() == ""):
            return ""
        
        print("preparting protmp")
        
        context = self.get_context(prompt)
        # sentiment = self.get_sentiment()

        # start = time.time()

        print(context)

        final_prompt = self.create_answer_prompt(prompt, self.init_sentiment, context)

        # end = time.time()
        # print("Creating answer prompt took: " + str(end - start))

        response = self.db.generate_text(final_prompt)
        # end2 = time.time()
        # print("Generating prompt took: " + str(end2 - start))

        return response