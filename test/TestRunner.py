import keyboard
import csv
import json
from classes.Character import Character
from test.AgentTest import AgentTest
from models import StageTestPrompt
from logger import configure_logging, get_logger

situation="{{user}} enters the village of Rack and stumbles upon {{char}}. {{char}} initiates the contact to {{user}}"
all_characters = []
selected_character = 0

configure_logging()
logger = get_logger(__name__)

def test_agent(character: Character):
    npc_judge = AgentTest()
    with open('./data/test_data/' + character.name.lower() + '_stage_testsuite.csv', mode ='r', encoding="utf-8") as file:
        testFile = csv.DictReader(file, delimiter=';')

        rows = list(testFile)
        for row in rows:
            for field_name in ("deterministic_checks", "judge_metrics", "stage_inputs"):
                raw_value = row.get(field_name)
                if raw_value:
                    row[field_name] = json.loads(raw_value)

        all_prompts = [StageTestPrompt(**r) for r in rows] # type: ignore
        npc_judge.evaluate_prompts(prompts=all_prompts, character=character)

def show_menu(options):
    logger.info("\n" * 30)
    logger.info("Choose to talk to character:")
    for i, option in enumerate(options):
        logger.info("{0} {1} {2}".format(">" if selected_character == i else " ", option, "<" if selected_character == i else " "))

def up():
    if selected_character == 0:
        set_char(len(character_options) - 1)
    else:
        set_char(selected_character - 1)
    show_menu(character_options)

def down():
    if selected_character == len(character_options) - 1:
        set_char(0)
    else:
        set_char(selected_character + 1)
    show_menu(character_options)

def set_char(index: int):
    global selected_character
    selected_character = index

with open('./data/character_data_cop.csv', mode ='r') as file:
        csvFile = csv.DictReader(file, delimiter=';')

        logger.info("Retrieved factions")

        for character in csvFile:
            all_characters.append(character)

character_options = [char.get("name") for char in all_characters]

show_menu(character_options)

keyboard.add_hotkey('up', up)
keyboard.add_hotkey('down', down)

keyboard.wait("enter")

npc = Character(all_characters[selected_character], situation)
test_agent(npc)
