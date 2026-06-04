import unittest
from unittest.mock import patch

from ai.settings import BUILT_IN_PROFILES
from models import Faction
from classes.Character import Character


class CharacterSettingsWiringTests(unittest.TestCase):
    def test_character_passes_shared_settings_to_agent_and_db(self):
        settings = BUILT_IN_PROFILES["local_current_hardcoded"]
        char_data = {
            "name": "Mira",
            "faction": Faction.WORLD,
            "pl_list": "Helpful trader",
            "ali_chat": "Welcome, traveler.",
            "knowledge": "",
            "past": "",
            "relations": "",
            "sentiment": "neutral",
        }

        with patch("classes.Character.ChromaDBHelper") as chroma_helper_cls, patch("classes.Character.NPCAgent") as npc_agent_cls, patch.object(Character, "compute_sentiment", return_value="neutral"):
            character = Character(char_data=char_data, situation="At the market", settings=settings)

        self.assertIs(character.ai_settings, settings)
        chroma_helper_cls.assert_called_once_with(settings)
        npc_agent_cls.assert_called_once_with(settings)


if __name__ == "__main__":
    unittest.main()
