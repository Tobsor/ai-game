import keyboard
import csv
import json
from classes.Character import Character
from test.NPCJudge import NpcJudge
from test.AgentTest import AgentTest
from models import TestPrompt, AgentTestPrompt
from logger import configure_logging, get_logger

situation="{{user}} enters the village of Rack and stumbles upon {{char}}. {{char}} initiates the contact to {{user}}"
all_characters = []
test_options = ["Character roleplay", "Character Agent"]

is_test = False

selected_test = 0
selected_character = 0

configure_logging()
logger = get_logger(__name__)

def test_character(character: Character):
    npc_judge = NpcJudge()

    with open('./data/test_data/' + character.name + '_testsuite.csv', mode ='r') as file:
        testFile = csv.DictReader(file, delimiter=';')

        all_prompts = [TestPrompt(**r) for r in list(testFile)] # type: ignore

        npc_judge.evaluate_prompts(prompts=all_prompts, character=character)

def test_agent(character: Character):
    npc_judge = AgentTest()
    with open('./data/test_data/' + character.name + '_testsuite_agent.csv', mode ='r') as file:
        testFile = csv.DictReader(file, delimiter=';')

        rows = list(testFile)
        for row in rows:
            expected_args = row.get("expected_args")
            if expected_args:
                row["expected_args"] = json.loads(expected_args)
                
        all_prompts = [AgentTestPrompt(**r) for r in rows] # type: ignore
        npc_judge.evaluate_prompts(prompts=all_prompts, character=character, tools=[
                    Character.cognitive_action,
                    Character.generate_npc_intention,
                    Character.immediate_action,
                    Character.change_sentiment
                ])

def show_menu(options):
    selected = selected_character
    if is_test:
        selected = selected_test

    logger.info("\n" * 30)
    logger.info("Choose to talk to character:")
    for i, option in enumerate(options):
        logger.info("{0} {1} {2}".format(">" if selected == i else " ", option, "<" if selected == i else " "))

def up():
    selected = selected_character
    setter = set_char
    options = character_options
    if is_test:
        selected = selected_test
        setter = set_test
        options = test_options

    if selected == 0:
        setter(len(options) - 1)
    else:
        setter(selected - 1)
    show_menu(options)

def down():
    selected = selected_character
    setter = set_char
    options = character_options
    if is_test:
        selected = selected_test
        setter = set_test
        options = test_options

    if selected == len(options):
        setter(0)
    else:
        setter(selected + 1)
    show_menu(options)

def set_char(index: int):
    global selected_character
    selected_character = index

def set_test(index: int):
    global selected_test
    selected_test = index

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

show_menu(test_options)

is_test = True

keyboard.wait("enter")

npc = Character(all_characters[selected_character], situation)

if selected_test == 0:
    test_character(npc)
else:
    test_agent(npc)
