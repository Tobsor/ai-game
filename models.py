from enum import Enum
from pydantic import BaseModel
from typing import Optional

class Faction(Enum):
    RACCOON = "raccoon"
    JELLIFISH = "jellyfish"
    ANTS = "ants"
    APES = "apes"
    CHAMELEON = "chameleon"
    FENNEK = "fennek"
    WORLD = "1"

class MetadataType(Enum):
    FACTION = "faction"
    CHARACTER = "character"

class PromptCategory(Enum):
    GENERAL = "general"
    SENTIMENT = "sentiment"
    LORE = "lore"
    MANIPULATION = "manipulation"
    JAILBREAK = "jailbreak"
    PAST = "past"

class MetadataCategory(Enum):
    KNOWLEDGE = "knowledge"
    PAST = "past"
    SENTIMENT = "sentiment"
    MEMORY = "memory"
    RELATIONS = "relations"
    LORE = "lore"

class FactionData(BaseModel):
    faction: Faction
    lore: str

class Character(BaseModel):
    name: str
    faction: Faction
    pl_list: str
    ali_chat: str
    knowledge: str
    past: str
    relations: str
    sentiment: str

class Metadata(BaseModel):
    faction: Optional[Faction] = None
    name: Optional[str] = None
    type: Optional[MetadataType] = None
    category: Optional[MetadataCategory] = None

class TestPrompt(BaseModel):
    user_query: str
    npc_response: Optional[str] = None
    category: PromptCategory
    authors_note: str