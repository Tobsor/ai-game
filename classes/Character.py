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

no_change = "No sentiment change"
end_keyword = "END"
continue_keyword = "CONTINUE"

class Character:
    def __init__(self, name, faction, context="", sentiment=""):
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
        self.context = context
        self.system_prompt = f"""
        Enter RP mode. You shall reply to the protagnoist, a red panda, while staying in character. Your responses must be detailed, creative, immersive, and drive the scenario forward. You will follow {name}'s persona as follows:
        {context}
        """

        self.db.init_context(self.system_prompt)
        self.current_sentiment = sentiment

    # def create_answer_prompt(self, prompt, context):
    #     return f"""
    #     Always answer alligned with the character description provided in the first system prompt! Consider the following context for this specific prompt:

    #     {context}

    #     Your current sentiment towards the player is the following:
    #     {self.current_sentiment}

    #     Give an answer in the following format

    #     [Characters answer]|[Sentiment change]|[End]

    #     Characters answer: Answer {prompt}, only stating the spoken word
    #     Sentiment change: Evaluate the emotional reaction of the character towards the players prompt and summarize an eventual sentiment change **in 10 words or fewer** or default to {no_change}
    #     End: Decide if the character is willing to end the discussion with the player. If he does, add {end_keyword} here, else {continue_keyword}
    #     """

    def create_answer_prompt(self, prompt, context):
        return f"""
        <|system|>Enter RP mode. Pretend to be {self.name} whose persona follows:

        Current sentiment towards the player: {self.current_sentiment}
        Context towards user prompt: {context}

        You shall reply to the user while staying in character, and generate long responses.
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

    def prompt(self, prompt):
        if(prompt.strip() == ""):
            return "", True
        
        context = self.db.query_docs(prompt=prompt, filter=self.query, concat=True)
        final_prompt = self.create_answer_prompt(prompt, context)

        response = self.db.generate_text(final_prompt)

        # message, sentiment_change, end = response.split("|")

        # if sentiment_change.strip() != no_change:
        #     self.current_sentiment = sentiment_change

        # print("sentiment_change ==> " + sentiment_change)

        return response
        