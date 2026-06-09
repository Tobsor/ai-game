from abc import ABC, abstractmethod


class Stage(ABC):
    def __init__(self, character):
        self.character = character


class LLMStage(Stage, ABC):
    @abstractmethod
    def get_prompt(self, *args, **kwargs) -> str:
        """Build the stage-owned prompt passed to the decision model."""
