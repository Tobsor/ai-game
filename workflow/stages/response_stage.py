from logger import get_logger
from workflow.models import InitialContext, PerceptionResult, ResponseResult, RetrievedContext, StrategyResult
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class ResponseStage(LLMStage):
    def get_turn_prompt(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
    ) -> str:
        return format_prompt(
            "Respond to the current turn using the latest player input, retrieved context, and response strategy.",
            [
                ("Player input", perception.raw_prompt),
                ("Recent conversation state", "\n".join(initial_context.recent_turns)),
                ("Retrieved context", retrieved_context.combined_context),
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
                        f"Respond to the following user input: {perception.raw_prompt}",
                    ]),
                ),
            ],
        )

    def get_prompt(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
    ) -> str:
        return self.get_turn_prompt(initial_context, perception, retrieved_context, strategy)

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
    ) -> ResponseResult:
        turn_prompt = self.get_turn_prompt(initial_context, perception, retrieved_context, strategy)

        logger.verbose("Response stage assembled turn prompt")
        logger.debug("Turn prompt: %s", turn_prompt)

        response = self.character.db.generate_text(turn_prompt, stage_name="ResponseStage")

        return ResponseResult(
            reply=response,
            turn_prompt=turn_prompt,
        )
