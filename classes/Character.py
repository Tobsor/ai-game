from classes.ChromaDBHelper import ChromaDBHelper

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
    def __init__(self, name, faction, context=""):
        self.name = name
        self.faction = faction
        self.id = str(name) + str(faction)
        self.db = ChromaDBHelper()
        self.query = {
            "$or": [
                {
                    "$and" : [
                        {
                            "name": name,
                        },
                        {
                            "type": "character"
                        }
                    ]
                },
                {
                    "$and" : [
                        {
                            "faction": faction,
                        },
                        {
                            "type": "faction"
                        }
                    ]
                }
            ]
        }

        self.system_prompt = f"""
        You impersonate a character in a fantasy world. Please use this context to impersonate the character the best:

        You are {name}. {context}
        """

    def create_answer_prompt(self, prompt, context):
        return f"""
        {self.system_prompt}

        Consider the following context for this specific prompt:        

        {context}

        Now, answer the following user query:
        {prompt}

        Also, please make a short summary how the character now feels about the player or a change of sentiment after the given prompt. Only do so if it has impact on the further conversation and attach it after a |
        """

    def create_prefix(self, category):
        return "[Faction: " + self.faction + ", Name: "+ self.name + ", category: " + annotation_mapping.get(category) + "]: "
    
    def map_data_obj_creation(self, value, category):
        return list(map(lambda x: {"value": x[1], "category": category, "id": str(self.id) + category + str(x[0])}, enumerate(value.split("."))))
    
    def add_attribute(self, key, value):
        all_attr = self.map_data_obj_creation(value, key)

        for attr in all_attr:
            self.add_embedding(attr)

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

    def prompt(self, prompt):
        if(prompt.strip() == ""):
            return "What did you say?"        
        
        context = self.db.query_docs(prompt=prompt, filter=self.query, concat=True)
        final_prompt = self.create_answer_prompt(prompt, context)

        response = self.db.generate_text(final_prompt)

        message, sentiment_change = response.split("|")

        print("sentiment_change ==> " + sentiment_change)

        return message