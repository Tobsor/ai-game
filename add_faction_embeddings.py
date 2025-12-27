import csv
from classes.ChromaDBHelper import ChromaDBHelper
from models import FactionData, Faction, Metadata, MetadataType, MetadataCategory
from logger import configure_logging, get_logger

db = ChromaDBHelper()
chunk_size = 100
overlap_size = 50
fields_to_chunk: list[MetadataCategory] = [MetadataCategory.LORE]

configure_logging()
logger = get_logger(__name__)

def create_chunks(text: str, chunk_size: int, overlap_size: int):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap_size

    return chunks

def create_id(index: int, faction: Faction, category: MetadataCategory):
        return "f-" + faction.value + "_type-" + category.value + "-" + str(index)

def add_embedding(index: int, value: str, faction: Faction, category: MetadataCategory):
    if value == "":
        return

    annotation = create_id(index, faction, category)
    metadata = Metadata(faction=faction, type=MetadataType.FACTION, category=category)

    db.add_embedding(
        annotation,
        value,
        metadata,
    )

def add_faction_embeddings():
    with open('./data/faction_data/faction_data.csv', mode ='r') as file:
        csvFile = csv.DictReader(file, delimiter=';')

        logger.info("Retrieved faction data")

        for faction in csvFile:
            faction_data = FactionData(**faction) # type: ignore

            logger.info("Adding attributes for %s", faction_data.faction)

            for field in fields_to_chunk:
                text = getattr(faction_data, field.value)

                chunks = create_chunks(text, chunk_size, overlap_size)
                logger.info("Created chunks")

                for (id, chunk) in enumerate(chunks):
                    add_embedding(id, chunk, faction_data.faction, field)
                    logger.info("Chunk %s added", id)

            logger.info("%s embeddings done", faction_data.faction)

add_faction_embeddings()        
