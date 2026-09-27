from abc import ABC, abstractmethod
from workflow.enrichment import ContextEnrichment


class Stage(ABC):
    def __init__(self, character):
        self.character = character


class LLMStage(Stage, ABC):
    @abstractmethod
    def get_prompt(self, *args, **kwargs) -> str:
        """Build the stage-owned prompt passed to the decision model."""


class EnrichedStage(LLMStage):
    def __init__(self, character, enrichment=None):
        super().__init__(character)
        self.enrichment = enrichment or ContextEnrichment(character)
