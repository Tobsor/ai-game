from classes.ChromaDBHelper import ChromaDBHelper

annotation_mapping = {
    "job": "Job",
    "hook": "Hook",
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
    def __init__(self, char_data, situation):
        print(char_data)
        self.name = char_data.get("name")
        self.faction = char_data.get("faction")
        self.id = char_data.get("name") + char_data.get("faction")
        self.pl_list = char_data.get("pl_list")
        self.ali_chat = char_data.get("ali_chat")
        self.situation = situation
        self.db = ChromaDBHelper()

    def create_answer_prompt(self, prompt, sentiment, context):
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

    def add_embedding(self, attr):
        if attr.get("value") == "":
            return

        annotation = self.create_prefix(category=attr.get("category"))
        label = annotation + attr.get("value")

        self.db.add_embedding(
            attr.get("id"),
            label,
            {"faction": self.faction, "type": "character", "category": attr.get("category"), "name": self.name}
        )
    
    def get_sentiment(self, prompt):
        filter = {
            "$and" : [
                {
                    "char_name": self.name,
                },
                {
                    "type": "sentiment"
                }
            ]
        }

        return self.db.query_docs(prompt=prompt, filter=filter, concat=True)
    
    def get_memories(self, prompt):
        filter = {
            "$and" : [
                {
                    "char_name": self.name,
                },
                {
                    "type": "memory"
                }
            ]
        }

        return self.db.query_docs(prompt=prompt, filter=filter, concat=True)
    
    def get_past(self, prompt):
        filter = {
            "$and" : [
                {
                    "char_name": self.name,
                },
                {
                    "type": "past"
                }
            ]
        }

        return self.db.query_docs(prompt=prompt, filter=filter, concat=True)
    
    def get_faction_knowledge(self, prompt):
        filter = {
            "$and" : [
                {
                    "faction": self.faction,
                },
                {
                    "type": "lore"
                }
            ]
        }

        return self.db.query_docs(prompt=prompt, filter=filter, concat=True)

    def get_world_knowledge(self, prompt):
        filter = {
            "$and" : [
                {
                    "faction": "world",
                },
                {
                    "type": "lore"
                }
            ]
        }

        return self.db.query_docs(prompt=prompt, filter=filter, concat=True)
    
    def get_relations(self, prompt):
        filter = {
            "$and" : [
                {
                    "char_name": self.name,
                },
                {
                    "type": "relation"
                }
            ]
        }

        return self.db.query_docs(prompt=prompt, filter=filter, concat=True)
    
    def compute_context(self, past, memories, world_knowledge, faction_knowledge, relations):
        return past
        
    def prompt(self, prompt):
        if(prompt.strip() == ""):
            return "", True
        
        sentiment = self.get_sentiment(prompt)
        past = self.get_past(prompt)
        memories = self.get_memories(prompt)
        world_knowledge = self.get_faction_knowledge(prompt)
        faction_knowledge = self.get_world_knowledge(prompt)
        relations = self.get_relations(prompt)

        context = self.compute_context(past, memories, world_knowledge, faction_knowledge, relations)

        final_prompt = self.create_answer_prompt(prompt, sentiment, context)

        response = self.db.generate_text(final_prompt)

        return response
        