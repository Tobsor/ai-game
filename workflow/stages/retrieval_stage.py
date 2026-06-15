from logger import get_logger
from workflow.models import GapAnalysisResult, PerceptionResult, RetrievedContext
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class RetrievalStage(LLMStage):
    def get_prompt(self, perception: PerceptionResult, gap_analysis: GapAnalysisResult) -> str:
        return format_prompt(
            "Gather the contextual knowledge the NPC should consult before composing a reply by executing the retrieval-oriented decisions produced during gap analysis.",
            [
                ("Player input", perception.raw_prompt),
                ("Perception summary", f"intent={perception.player_intent}, request_type={perception.request_type}, topic={perception.topic}"),
                ("Gap-analysis tool calls", str([tool_call.function.arguments for tool_call in gap_analysis.tool_calls])),
                ("Expected result", "The retrieved context needed for the NPC's response."),
            ],
        )

    def run(self, perception: PerceptionResult, gap_analysis: GapAnalysisResult) -> RetrievedContext:
        logger.verbose("Running retrieval stage with %s gap-analysis tool calls", len(gap_analysis.tool_calls))
        memory_context = "no information"
        relationship_context = "no information"
        knowledge_context = "no information"
        social_context = "no information"

        for tool_call in gap_analysis.tool_calls:
            tool_name = tool_call.function.name
            tool_arguments = tool_call.function.arguments

            if tool_name == "recall_memory":
                memory_context = self.recall_memory(perception, tool_arguments)
            elif tool_name == "recall_relationship":
                relationship_context = self.recall_relationship(perception, tool_arguments)
            elif tool_name == "recall_knowledge":
                knowledge_context = self.recall_knowledge(perception, tool_arguments)
            elif tool_name == "evaluate_social_context":
                social_context = self.evaluate_social_context(perception, tool_arguments)

        combined_parts = [
            text for text in [
                memory_context,
                relationship_context,
                knowledge_context,
                social_context,
            ]
            if text != "no information"
        ]
        raw_combined_context = "\n".join(combined_parts) if len(combined_parts) > 0 else "no information"
        combined_context = self.summarize_retrieved_context(perception, raw_combined_context)

        return RetrievedContext(
            combined_context=combined_context,
            memory_context=memory_context,
            relationship_context=relationship_context,
            knowledge_context=knowledge_context,
            social_context=social_context,
        )

    # TODO: adjust tag based filtering for db queries for all helper functions
    def recall_memory(self, perception: PerceptionResult, tool_arguments: dict) -> str:
        return self.character.db.query_text(
            prompt=self.build_tool_prompt(
                "Recall prior memories or past events that help the NPC answer the player's message.",
                perception,
                tool_arguments,
            ),
            stage_name="RetrievalStage.run",
        ).strip() or "no information"

    def recall_relationship(self, perception: PerceptionResult, tool_arguments: dict) -> str:
        return self.character.db.query_text(
            prompt=self.build_tool_prompt(
                "Recall relationship history, trust, sentiment, and shared context with the player.",
                perception,
                tool_arguments,
            ),
            stage_name="RetrievalStage.run",
        ).strip() or "no information"

    def recall_knowledge(self, perception: PerceptionResult, tool_arguments: dict) -> str:
        return self.character.db.query_text(
            prompt=self.build_tool_prompt(
                "Recall world knowledge, faction knowledge, or topic-specific knowledge relevant to the player's message.",
                perception,
                tool_arguments,
            ),
            stage_name="RetrievalStage.run",
        ).strip() or "no information"

    def evaluate_social_context(self, perception: PerceptionResult, tool_arguments: dict) -> str:
        return self.character.db.query_text(
            prompt=self.build_tool_prompt(
                "Recall social context that affects how the NPC should interpret or answer the player's message.",
                perception,
                tool_arguments,
            ),
            stage_name="RetrievalStage.run",
        ).strip() or "no information"

    def summarize_retrieved_context(self, perception: PerceptionResult, raw_context: str) -> str:
        if raw_context == "no information":
            return raw_context

        response = self.character.agent.run_prompt(
            prompt=self.build_summary_prompt(perception, raw_context),
            stage_name="RetrievalStage.summarize",
            payload={
                "player_prompt": perception.raw_prompt,
                "retrieved_context": raw_context,
            },
        )
        summary = response.content.strip()
        return summary or "no information"

    def build_tool_prompt(self, instruction: str, perception: PerceptionResult, tool_arguments: dict) -> str:
        reasoning = str(tool_arguments.get("reasoning", "")).strip()
        return "\n".join([
            instruction,
            f"Player input: {perception.raw_prompt}",
            f"Perceived intent: {perception.player_intent}",
            f"Perceived topic: {perception.topic}",
            f"Retrieval reason: {reasoning if reasoning != '' else 'not provided'}",
            "Return only the retrieved context as plain text.",
        ])

    def build_summary_prompt(self, perception: PerceptionResult, raw_context: str) -> str:
        return "\n".join([
            "Summarize the retrieved context for downstream response generation.",
            "Refer every included context snippet directly to the player's prompt input.",
            "Include only evidently relevant context snippets.",
            "Do not invent facts, explanations, or links that are not explicitly supported by the retrieved context.",
            "If nothing in the retrieved context is evidently relevant, return exactly: no information",
            f"Player input: {perception.raw_prompt}",
            f"Perceived intent: {perception.player_intent}",
            f"Perceived topic: {perception.topic}",
            "Retrieved context:",
            raw_context,
            "Return only the concise summarized context as plain text.",
        ])
