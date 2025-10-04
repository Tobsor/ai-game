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

no_change = "No sentiment change"
end_keyword = "END"
continue_keyword = "CONTINUE"

class Character:
    def __init__(self, char_data, situation):
        self.name = char_data.name
        self.faction = char_data.faction
        self.id = char_data.name + char_data.faction
        self.situation = situation
        self.db = ChromaDBHelper()
        self.query = {
            "$or": [
                {
                    "$and" : [
                        {
                            "name": char_data.name,
                        },
                        {
                            "type": "character"
                        }
                    ]
                },
                {
                    "$and" : [
                        {
                            "faction": char_data.faction,
                        },
                        {
                            "type": "faction"
                        }
                    ]
                }
            ]
        }
        self.pl_list = char_data.pl_list
        self.speech = char_data.speech
        self.hook = char_data.hook
        self.greeting = char_data.greeting
        self.system_prompt = f"""
        Enter RP mode. You shall reply to the protagnoist, a red panda, while staying in character. Your responses must be detailed, creative, immersive, and drive the scenario forward. You will follow {name}:
        {char_data.pl_list}
        Hook/Motivation: {char_data.hook}
        Speech style: {char_data.speech}
        """

        self.db.init_context(self.system_prompt)

    def create_answer_prompt(self, prompt, example, sentiment, context):
        return f"""
        {self.system_prompt}

        Sentiment towards player:
        {sentiment}

        Example conversion:
        {example}

        General context:
        {context}

        You shall reply to the user while staying in character.
        <|user|>{prompt}
        <|model|>{{model's response goes here}}
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

    def get_example(self, prompt):
        filter = {
            "$and" : [
                {
                    "char_name": self.name,
                },
                {
                    "type": "example"
                }
            ]
        }

        return self.db.query_docs(prompt=prompt, filter=filter, concat=True)
    
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
    
    def get_context(self, prompt):
        filter = {
            "$or": [
                {
                    "$and" : [
                        {
                            "char_name": self.name,
                        },
                        {
                            "$or": [
                                {
                                    "type": "relations"
                                },
                                {
                                    "type": "memory"
                                },
                                {
                                    "$and": [
                                        {
                                            "type": "relations"
                                        },
                                        {
                                            "char_name": self.name
                                        }
                                    ]
                                },
                                {
                                    "$and": [
                                        {
                                            "type": "faction"
                                        },
                                        {
                                            "name": self.faction
                                        }
                                    ]
                                }
                            ]
                        },
                    ],
                },
                {
                    "type": "world"
                }
            ]
        }

        return self.db.query_docs(prompt=prompt, filter=filter, concat=True)

    def prompt(self, prompt):
        if(prompt.strip() == ""):
            return "", True
        
        example = self.get_example(prompt)
        sentiment = self.get_sentiment(prompt)
        context = self.get_context(prompt)

        final_prompt = self.create_answer_prompt(prompt, example, sentiment, context)

        response = self.db.generate_text(final_prompt)

        return response
        