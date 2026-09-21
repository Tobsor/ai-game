"""Run Tom's isolated appraisal cases and export real scores: python -m test.run_appraisal."""
from types import SimpleNamespace

from ai import get_ai_settings
from classes.NpcAgent import NPCAgent
from models import StageName
from test.AgentTest import AgentTest
from test.TestRunner import read_stage_prompts
from test.build_appraisal_fixtures import read_tom
from workflow.stages.appraisal_stage import AppraisalStage


def main():
    # No Character constructor: it initializes Chroma and computes live sentiment.
    data = read_tom()
    character = SimpleNamespace(name=data["name"])
    settings = get_ai_settings()
    character.agent = NPCAgent(settings)
    character.pipeline = SimpleNamespace(appraisal_stage=AppraisalStage(character))
    prompts = [prompt for prompt in read_stage_prompts(character) if prompt.target_stage == StageName.APPRAISAL]
    if not prompts:
        raise ValueError("No appraisal scenarios found in Tom's suite.")
    runner = AgentTest(settings)
    runner.evaluate_prompts(character, prompts)
    print("Appraisal outputs: test/results/tom_stage_testsuite_{prompts,results}.csv")


if __name__ == "__main__":
    main()
