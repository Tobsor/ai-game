import csv
from classes.ChromaDBHelper import ChromaDBHelper
from models import Character, Faction, MetadataType, MetadataCategory, Metadata

db = ChromaDBHelper()
chunk_size = 100
overlap_size = 50
fields_to_chunk = [MetadataCategory.KNOWLEDGE, MetadataCategory.PAST, MetadataCategory.RELATIONS]

def create_chunks(text: str, chunk_size: int, overlap_size: int):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap_size

    return chunks

def create_id(index: int, faction: Faction, category: MetadataCategory, name: str):
    return "f-" + faction.value + "_n-" + name + "c-" + category.value + "-" + str(index)

def add_embedding(index: int, value: str, name: str, faction: Faction, category: MetadataCategory):
    if value == "":
        return

    annotation = create_id(index, faction, category, name)
    metadata = Metadata(faction=faction, type=MetadataType.CHARACTER, category=category, name=name)

    db.add_embedding(
        annotation,
        value,
        metadata
    )

def add_character_embeddings():
    with open('./data/character_data_cop.csv', mode ='r') as file:
        csvFile = csv.DictReader(file, delimiter=';')

        print("retrieved char data")

        for character in csvFile:
            character_data = Character(**character) # type: ignore

            print("adding attributes for " + character_data.name)

            for field in fields_to_chunk:
                text = getattr(character_data, field.value)

                chunks = create_chunks(text, chunk_size, overlap_size)
                print("created chunks")

                for (id, chunk) in enumerate(chunks):
                    add_embedding(id, chunk, character_data.name, character_data.faction, field)
                    print("chunk " + str(id) + " added")

            print(character_data.name + " embeddings done")

add_character_embeddings()