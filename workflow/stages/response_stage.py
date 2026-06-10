from logger import get_logger
from workflow.models import InitialContext, PerceptionResult, ResponseResult, RetrievedContext, StrategyResult
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class ResponseStage(LLMStage):
    def get_prompt(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
    ) -> str:
        return format_prompt(
            f"Enter RP mode. You are {initial_context.character_name}. Stay in character at all times, speak in first person, and produce only perceivable dialogue or first-person nonverbal actions.",
            [
                ("Situation", initial_context.situation),
                ("Sentiment towards player", initial_context.sentiment),
                ("Character definition", initial_context.character_definition),
                ("Example dialogues", initial_context.example_dialogues),
                ("Player input", perception.raw_prompt),
                ("Retrieved context", retrieved_context.combined_context),
                ("Relationship summary", initial_context.relationship_summary),
                ("Active goals", ", ".join(initial_context.active_goals)),
                ("Recent turns", " | ".join(initial_context.recent_turns)),
                ("Beliefs", ", ".join(initial_context.belief_state)),
                (
                    "Response strategy",
                    "\n".join([
                        f"intention={strategy.intention}",
                        f"conversation_goal={strategy.conversation_goal}",
                        f"risk_level={strategy.risk_level}",
                        f"disclosure_level={strategy.disclosure_level}",
                        f"social_strategy={strategy.social_strategy}",
                        f"tone={strategy.tone}",
                        f"verbosity={strategy.verbosity}",
                        f"conversation_move={strategy.conversation_move}",
                    ]),
                ),
                (
                    "Reply instructions",
                    "\n".join([
                        "Mention nonverbal content only from the first-person perspective.",
                        "Do not reveal internal thoughts directly.",
                        "Reply to the player while staying fully in character.",
                        f"<|user|>{perception.raw_prompt}",
                        "<|model|>{model's response goes here}",
                    ]),
                ),
            ],
        )

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
    ) -> ResponseResult:
        final_prompt = self.get_prompt(initial_context, perception, retrieved_context, strategy)

        logger.verbose("Response stage assembled final prompt")
        logger.debug("Final prompt: %s", final_prompt)
        response = self.character.db.generate_text(final_prompt, stage_name="ResponseStage")

        return ResponseResult(reply=response, final_prompt=final_prompt)
