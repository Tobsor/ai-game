import csv
from classes.Character import Character
from test.NPCJudge import NpcJudge
from models import TestPrompt

situation="{{user}} enters the village of Rack and stumbles upon {{char}}. {{char}} initiates the contact to {{user}}"

def test_character(char_name: str):
    with open('./data/character_data_cop.csv', mode ='r') as file:
        csvFile = csv.DictReader(file, delimiter=';')

        character_raw = next((x for x in csvFile if x.get("name") == char_name), None)

        npc_judge = NpcJudge()
        character = Character(char_data=character_raw, situation=situation)

        with open('./data/test_data/' + character.name + '_testsuite.csv', mode ='r') as file:
            testFile = csv.DictReader(file, delimiter=';')

            all_prompts = [TestPrompt(**r) for r in list(testFile)] # type: ignore

            npc_judge.evaluate_prompts(prompts=all_prompts, character=character)

test_character("Tom")