from classes.ChromaDBHelper import ChromaDBHelper

annotation_mapping = {
    "biological_properties": "Biological propertes",
    "phys_attr": "Physical property",
    "nature": "Character / Nature",
    "strengths": "strengths",
    "weakness": "weakness",
    "habital_area": "habital_area",
    "quantity": "quantity",
    "goal": "goal"
}

class Faction:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.db = ChromaDBHelper()

    def create_prefix(self, category):
        return "[Name: "+ str(self.name) + ", category: " + str(annotation_mapping.get(category)) + "]: "
    
    def map_data_obj_creation(self, value, category):
        return list(map(
            lambda x: {"value": x[1], "category": category, "id": str(self.id) + category + str(x[0])},
            enumerate(value.split("."))
        ))
    
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
            {"type": "faction", "category": attr.get("category"), "name": self.name}
        )
