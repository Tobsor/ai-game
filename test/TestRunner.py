import csv
import json
from pathlib import Path
from typing import TypeVar

import keyboard

from classes.Character import Character
from logger import configure_logging, get_logger
from models import StageName, StageTestPrompt
from test.AgentTest import AgentTest


situation = "{{user}} enters the village of Rack and stumbles upon {{char}}. {{char}} initiates the contact to {{user}}"
STAGE_ALL = "All stages"

configure_logging()
logger = get_logger(__name__)

T = TypeVar("T")


def choose_option(title: str, options: list[T], label_for_option) -> T:
    if len(options) == 0:
        raise ValueError(f"No options available for menu: {title}")

    selected_index = 0

    def show_menu() -> None:
        logger.info("\n" * 30)
        logger.info(title)
        for index, option in enumerate(options):
            marker_left = ">" if selected_index == index else " "
            marker_right = "<" if selected_index == index else " "
            logger.info("%s %s %s", marker_left, label_for_option(option), marker_right)

    def set_index(index: int) -> None:
        nonlocal selected_index
        selected_index = index

    def up() -> None:
        set_index((selected_index - 1) % len(options))
        show_menu()

    def down() -> None:
        set_index((selected_index + 1) % len(options))
        show_menu()

    show_menu()
    up_hotkey = keyboard.add_hotkey("up", up)
    down_hotkey = keyboard.add_hotkey("down", down)
    try:
        keyboard.wait("enter")
        return options[selected_index]
    finally:
        keyboard.remove_hotkey(up_hotkey)
        keyboard.remove_hotkey(down_hotkey)


def read_characters(path: Path) -> list[dict[str, str]]:
    with path.open(mode="r", encoding="utf-8") as file:
        csv_file = csv.DictReader(file, delimiter=";")
        logger.info("Retrieved factions")
        return list(csv_file)


def read_stage_prompts(character: Character) -> list[StageTestPrompt]:
    path = Path("./data/test_data") / f"{character.name.lower()}_stage_testsuite.csv"
    with path.open(mode="r", encoding="utf-8") as file:
        test_file = csv.DictReader(file, delimiter=";")
        rows = list(test_file)

    for row in rows:
        for field_name in ("deterministic_checks", "judge_metrics", "stage_inputs"):
            raw_value = row.get(field_name)
            if raw_value:
                row[field_name] = json.loads(raw_value)

    return [StageTestPrompt(**row) for row in rows]  # type: ignore[arg-type]


def get_stage_options(prompts: list[StageTestPrompt]) -> list[str | StageName]:
    stages_in_suite = {prompt.target_stage for prompt in prompts}
    ordered_stages = [stage for stage in StageName if stage in stages_in_suite]
    return [STAGE_ALL, *ordered_stages]


def filter_prompts_by_stage(prompts: list[StageTestPrompt], selected_stage: str | StageName) -> list[StageTestPrompt]:
    if selected_stage == STAGE_ALL:
        return prompts
    return [prompt for prompt in prompts if prompt.target_stage == selected_stage]


def stage_label(option: str | StageName) -> str:
    if isinstance(option, StageName):
        return option.value
    return option


def test_agent(character: Character) -> None:
    all_prompts = read_stage_prompts(character)
    selected_stage = choose_option(
        title="Choose stage to test:",
        options=get_stage_options(all_prompts),
        label_for_option=stage_label,
    )
    prompts_to_run = filter_prompts_by_stage(all_prompts, selected_stage)
    logger.info(
        "Running %s prompt(s) for %s.",
        len(prompts_to_run),
        stage_label(selected_stage),
    )
    AgentTest().evaluate_prompts(prompts=prompts_to_run, character=character)


all_characters = read_characters(Path("./data/character_data_cop.csv"))
selected_character = choose_option(
    title="Choose character to test:",
    options=all_characters,
    label_for_option=lambda character: character.get("name", ""),
)

npc = Character(selected_character, situation)
test_agent(npc)
