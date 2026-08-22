from typing import Any

from logger import get_logger
from workflow.models import GapAnalysisResult, PerceptionResult
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class GapAnalysisStage(LLMStage):
    def get_prompt(self, perception: PerceptionResult) -> str:
        return format_prompt(
            "Analyze the current perception result and decide whether the NPC still has knowledge gaps before answering. If additional retrieval or context gathering is needed, express that decision through the provided retrieval tools. If no retrieval is needed, do not call any tools.",
            [
                ("Player input", perception.raw_prompt),
                ("Detected intent", perception.player_intent),
                ("Detected emotion", perception.player_emotion),
                ("Request type", perception.request_type),
                ("Topic", perception.topic),
                ("Threat signal", perception.threat_signal),
                ("Manipulation signal", perception.manipulation_signal),
                ("Topic sensitivity", perception.topic_sensitivity),
                ("Perception-stage tool context", self.describe_tool_calls(perception.tool_calls)),
                (
                    "Tool usage rules",
                    "\n".join([
                        "If more context is needed, call the relevant retrieval tools directly.",
                        "Tool calls should reflect the concrete context collection needed for downstream retrieval.",
                        "Only call tools when retrieval is actually required.",
                        "Only use tools that retrieve additional context such as memory, relationship history, knowledge, or social context.",
                    ]),
                ),
                ("Output rules", "No structured text output is required. The tool calls are the decision payload."),
            ],
        )

    def run(self, perception: PerceptionResult) -> GapAnalysisResult:
        logger.verbose("Running gap analysis with %s tool calls", len(perception.tool_calls))
        stage_prompt = self.get_prompt(perception)
        response = self.character.agent.run_prompt(
            prompt=stage_prompt,
            tools=[
                self.recall_memory,
                self.recall_relationship,
                self.recall_knowledge,
                self.evaluate_social_context,
            ],
            stage_name="GapAnalysisStage",
            payload={
                "input_prompt": perception.raw_prompt,
                "stage_prompt": stage_prompt,
            },
        )
        return GapAnalysisResult(tool_calls=list(response.tool_calls))

    def describe_tool_calls(self, tool_calls: list[Any]) -> str:
        if len(tool_calls) == 0:
            return "No tool calls were proposed during perception."

        parts: list[str] = []
        for tool_call in tool_calls:
            parts.append(f"{tool_call.function.name}: {tool_call.function.arguments}")
        return "\n".join(parts)

    def recall_memory(self, reasoning: str) -> dict[str, str]:
        return {"reasoning": reasoning}

    def recall_relationship(self, reasoning: str) -> dict[str, str]:
        return {"reasoning": reasoning}

    def recall_knowledge(self, reasoning: str) -> dict[str, str]:
        return {"reasoning": reasoning}

    def evaluate_social_context(self, reasoning: str) -> dict[str, str]:
        return {"reasoning": reasoning}
