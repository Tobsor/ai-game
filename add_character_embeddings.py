import csv
from classes.Character import Character
from classes.ChromaDBHelper import ChromaDBHelper

db = ChromaDBHelper()
chunk_size = 100
overlap_size = 50
fields_to_chunk = ["knowledge", "past", "relations"]

def create_chunks(text, chunk_size, overlap_size):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap_size

    return chunks

def create_id(id, faction, category, name):
        return "f-" + faction + "_n-" + name + "_category-" + category + "-" + str(id)

def add_embedding(id, value, name, faction, category):
    if value == "":
        return

    annotation = create_id(id, faction, category, name)

    db.add_embedding(
        annotation,
        value,
        {"faction": faction, "type": "character", "category": category, "name": name}
    )

def add_character_embeddings():
    with open('./data/character_data_cop.csv', mode ='r') as file:
        csvFile = csv.DictReader(file, delimiter=';')

        print("retrieved char data")

        for character in csvFile:
            faction = character.get("faction")
            name = character.get("name")

            print("adding attributes for " + name)

            for field in fields_to_chunk:
                text = character.get(field)

                chunks = create_chunks(text, chunk_size, overlap_size)
                print("created chunks")

                for (id, chunk) in enumerate(chunks):
                    add_embedding(id, chunk, name, faction, field)
                    print("chunk " + str(id) + " added")


        print(name + " embeddings done")

add_character_embeddings()