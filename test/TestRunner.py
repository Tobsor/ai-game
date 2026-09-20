import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, TypeVar

from classes.Character import Character
from logger import configure_logging, get_logger
from models import PromptCategory, StageExpectationMode, StageName, StageTestPrompt
from test.AgentTest import AgentTest


situation = "{{user}} enters the village of Rack and stumbles upon {{char}}. {{char}} initiates the contact to {{user}}"
STAGE_ALL = "All stages"

configure_logging()
logger = get_logger(__name__)

T = TypeVar("T")


def choose_option(title: str, options: list[T], label_for_option: Callable[[T], str]) -> T:
    if len(options) == 0:
        raise ValueError(f"No options available for menu: {title}")
    if not sys.stdin.isatty():
        selected_option = options[0]
        logger.info(
            "%s no interactive terminal detected; selected %s.",
            title,
            label_for_option(selected_option),
        )
        return selected_option

    import keyboard

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
    if not path.exists():
        raise FileNotFoundError(f"Character CSV not found: {path}")

    with path.open(mode="r", encoding="utf-8") as file:
        csv_file = csv.DictReader(file, delimiter=";")
        logger.info("Retrieved factions")
        return list(csv_file)


def stage_prompt_path_for(character: Character, data_dir: Path) -> Path:
    return data_dir / f"{character.name.lower()}_stage_testsuite.csv"


def build_smoke_stage_prompts(character: Character) -> list[StageTestPrompt]:
    return [
        StageTestPrompt(
            user_query="Hello there.",
            source_category=PromptCategory.GENERAL,
            target_stage=StageName.RETRIEVAL_RUN,
            expectation_mode=StageExpectationMode.DETERMINISTIC,
            deterministic_checks=[
                {
                    "metric_name": "no_context_without_tool_calls",
                    "path": "stage_output.combined_context",
                    "operator": "equals",
                    "value": "no information",
                }
            ],
            stage_inputs={
                "perception_payload": {
                    "npc_perception": {
                        "perceived_intent": ["greet"],
                        "perceived_attitude": ["neutral"],
                        "player_intent": "greet",
                        "player_emotion": "neutral",
                        "threat_signal": "none",
                        "manipulation_signal": "none",
                        "topic_sensitivity": "normal",
                    },
                    "request_type": "greeting",
                    "topic": {"primary": "greeting", "related": [], "retrieval_queries": []},
                },
                "gap_analysis_payload": {"tool_calls": []},
            },
            notes=f"Smoke fallback because no stage-aware suite exists for {character.name}.",
        )
    ]


def read_stage_prompts(character: Character) -> list[StageTestPrompt]:
    path = stage_prompt_path_for(character, Path("./data/test_data"))
    if not path.exists():
        legacy_path = path.with_name(f"{character.name.lower()}_testsuite.csv")
        if legacy_path.exists():
            logger.warning(
                "Stage-aware test suite not found: %s. Found legacy suite %s; running smoke fallback.",
                path,
                legacy_path,
            )
            return build_smoke_stage_prompts(character)
        raise FileNotFoundError(f"Stage-aware test suite not found: {path}")

    with path.open(mode="r", encoding="utf-8") as file:
        test_file = csv.DictReader(file, delimiter=";")
        rows = list(test_file)

    for row in rows:
        row.pop("judge_metrics", None)
        for field_name in ("deterministic_checks", "stage_inputs"):
            raw_value = row.get(field_name)
            if raw_value:
                row[field_name] = json.loads(raw_value)

    return [StageTestPrompt(**row) for row in rows]  # type: ignore[arg-type]


def get_stage_options(prompts: list[StageTestPrompt]) -> list[str | StageName]:
    return [STAGE_ALL, *StageName]


def filter_prompts_by_stage(prompts: list[StageTestPrompt], selected_stage: str | StageName) -> list[StageTestPrompt]:
    if selected_stage == STAGE_ALL:
        return prompts
    return [prompt for prompt in prompts if prompt.target_stage == selected_stage]


def stage_label(option: str | StageName) -> str:
    if isinstance(option, StageName):
        return option.value
    return option


def select_by_label(options: list[T], selected_label: str, label_for_option: Callable[[T], str], option_name: str) -> T:
    normalized_selected_label = selected_label.strip().lower()
    for option in options:
        if label_for_option(option).strip().lower() == normalized_selected_label:
            return option

    labels = ", ".join(label_for_option(option) for option in options)
    raise ValueError(f"Unknown {option_name} '{selected_label}'. Available options: {labels}")


def select_character(characters: list[dict[str, str]], selected_name: str | None) -> dict[str, str]:
    if selected_name is not None:
        return select_by_label(
            options=characters,
            selected_label=selected_name,
            label_for_option=lambda character: character.get("name", ""),
            option_name="character",
        )

    return choose_option(
        title="Choose character to test:",
        options=characters,
        label_for_option=lambda character: character.get("name", ""),
    )


def select_stage(prompts: list[StageTestPrompt], selected_stage_name: str | None) -> str | StageName:
    stage_options = get_stage_options(prompts)
    if selected_stage_name is not None:
        return select_by_label(
            options=stage_options,
            selected_label=selected_stage_name,
            label_for_option=stage_label,
            option_name="stage",
        )

    return choose_option(
        title="Choose stage to test:",
        options=stage_options,
        label_for_option=stage_label,
    )


def test_agent(character: Character, selected_stage_name: str | None = None) -> None:
    all_prompts = read_stage_prompts(character)
    selected_stage = select_stage(all_prompts, selected_stage_name)
    prompts_to_run = filter_prompts_by_stage(all_prompts, selected_stage)
    if len(prompts_to_run) == 0:
        logger.warning("No prompts configured for %s.", stage_label(selected_stage))
    logger.info(
        "Running %s prompt(s) for %s.",
        len(prompts_to_run),
        stage_label(selected_stage),
    )
    AgentTest().evaluate_prompts(prompts=prompts_to_run, character=character)


if __name__ == "__main__":
    all_characters = read_characters(Path("./data/character_data_cop.csv"))
    selected_character = choose_option(
        title="Choose character to test:",
        options=all_characters,
        label_for_option=lambda character: character.get("name", ""),
    )
    npc = Character(selected_character, situation)
    test_agent(npc)
