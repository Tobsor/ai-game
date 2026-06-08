from typing import Any

from logger import get_logger
from models import CognitiveAction
from workflow.models import (
    GapAnalysisResult,
    InitialContext,
    PerceptionResult,
    ResponseResult,
    RetrievalPlan,
    RetrievedContext,
    StateUpdate,
    StrategyResult,
    TerminalUpdateResult,
    TurnInput,
)

logger = get_logger(__name__)


class InitialContextStage:
    def __init__(self, character: Any):
        self.character = character

    def run(self, turn_input: TurnInput) -> InitialContext:
        logger.verbose("Building initial context for %s", self.character.name)
        return InitialContext(
            character_name=self.character.name,
            situation=self.character.situation,
            sentiment=self.character.sentiment,
            character_definition=self.character.pl_list,
            example_dialogues=self.character.ali_chat,
            relationship_summary=self.build_relationship_summary(),
            active_goals=self.get_active_goals(),
            recent_turns=self.get_recent_turns(),
            belief_state=self.get_belief_state(),
        )

    def build_relationship_summary(self) -> str:
        prompt = (
            f"What is {self.character.name}'s relationship to the player? "
            "Recall relevant relationship history, social context, and current sentiment."
        )
        filter_value = {
            "$or": [
                self.character.get_relations(),
                self.character.get_sentiment_filter(),
            ]
        }
        relation_summary = self.character.db.query_text(
            prompt=prompt,
            filter=filter_value,
            stage_name="InitialContextStage",
        ).strip()
        if relation_summary == "":
            return ""

        return "Relationship to player:\n" + relation_summary

    def get_active_goals(self) -> list[str]:
        prompt = (
            f"Summarize {self.character.name}'s core values, morality, short term goals, "
            "mid term goals, and long term goals based only on the retrieved character knowledge. "
            "Return a compact plain-text summary."
        )
        filter_value = {
            "$and": [
                {
                    "name": self.character.name,
                },
                {
                    "type": "character",
                },
                {
                    "category": "knowledge",
                },
            ]
        }
        goal_summary = self.character.db.query_text(
            prompt=prompt,
            filter=filter_value,
            stage_name="InitialContextStage",
        ).strip()
        if goal_summary == "":
            return []

        return [goal_summary]

    def get_recent_turns(self) -> list[str]:
        # TODO: Summarize recent turns into a compact cross-turn context block.
        return []

    def get_belief_state(self) -> list[str]:
        # TODO: Load durable NPC beliefs for downstream reasoning.
        return []


class PerceptionStage:
    def __init__(self, character: Any):
        self.character = character

    def run(self, turn_input: TurnInput, initial_context: InitialContext) -> PerceptionResult:
        logger.verbose("Running perception stage for prompt length=%s", len(turn_input.prompt))
        tool_calls = self.character.agent.prompt_agent(
            prompt=turn_input.prompt,
            sentiment=initial_context.sentiment,
            name=initial_context.character_name,
            pl_list=initial_context.character_definition,
            situation=initial_context.situation,
            tools=[
                self.character.cognitive_action,
                self.character.generate_npc_intention,
                self.character.immediate_action,
                self.character.change_sentiment,
                self.character.flag_jailbreak,
            ],
            stage_name="PerceptionStage",
        ) or []

        normalized_prompt = self.normalize_jailbreak_prompt(turn_input.prompt, tool_calls)

        return PerceptionResult(
            raw_prompt=turn_input.prompt,
            normalized_prompt=normalized_prompt,
            jailbreak_detected=normalized_prompt != turn_input.prompt,
            player_intent=self.detect_player_intent(turn_input.prompt, initial_context),
            player_emotion=self.detect_player_emotion(turn_input.prompt, initial_context),
            request_type=self.detect_request_type(turn_input.prompt, initial_context),
            topic=self.detect_topic(turn_input.prompt, initial_context),
            is_ambiguous=self.detect_ambiguity(turn_input.prompt, initial_context),
            threat_signal=self.detect_threat_signal(turn_input.prompt, initial_context),
            manipulation_signal=self.detect_manipulation_signal(turn_input.prompt, initial_context),
            topic_sensitivity=self.detect_topic_sensitivity(turn_input.prompt, initial_context),
            tool_calls=tool_calls,
            retrieval_reasoning="",
        )

    def normalize_jailbreak_prompt(self, prompt: str, tool_calls: list[Any]) -> str:
        for tool_call in tool_calls:
            if tool_call.function.name != "flag_jailbreak":
                continue

            normalized = tool_call.function.arguments.get("normalized_user_prompt", "").strip()
            if normalized != "":
                return normalized

        # TODO: Replace this passthrough with robust jailbreak normalization rules.
        return prompt

    def detect_player_intent(self, prompt: str, initial_context: InitialContext) -> str:
        # TODO: Infer the player's concrete conversation intent from the turn.
        return "unknown"

    def detect_player_emotion(self, prompt: str, initial_context: InitialContext) -> str:
        # TODO: Detect emotional signals in the player's message.
        return "neutral"

    def detect_request_type(self, prompt: str, initial_context: InitialContext) -> str:
        # TODO: Classify the request type to support retrieval and strategy selection.
        return "general"

    def detect_topic(self, prompt: str, initial_context: InitialContext) -> str:
        # TODO: Extract or summarize the main topic for downstream reasoning.
        return ""

    def detect_ambiguity(self, prompt: str, initial_context: InitialContext) -> bool:
        # TODO: Detect when the user's request is too ambiguous for a direct answer.
        return False

    def detect_threat_signal(self, prompt: str, initial_context: InitialContext) -> str:
        # TODO: Classify threat signals to support future risk-aware behaviors.
        return "none"

    def detect_manipulation_signal(self, prompt: str, initial_context: InitialContext) -> str:
        # TODO: Detect manipulation or coercion signals in the user prompt.
        return "none"

    def detect_topic_sensitivity(self, prompt: str, initial_context: InitialContext) -> str:
        # TODO: Detect whether the requested topic should be treated as sensitive.
        return "normal"


class GapAnalysisStage:
    def __init__(self, character: Any):
        self.character = character

    def run(self, perception: PerceptionResult) -> GapAnalysisResult:
        logger.verbose("Running gap analysis with %s tool calls", len(perception.tool_calls))
        retrieval_actions = self.extract_retrieval_actions(perception.tool_calls)
        needs_memory = self.needs_memory(perception, retrieval_actions)
        needs_relationship = self.needs_relationship_context(perception, retrieval_actions)
        needs_knowledge = self.needs_knowledge(perception, retrieval_actions)
        needs_social = self.needs_social_context(perception, retrieval_actions)
        needs_threat = self.needs_threat_analysis(perception)
        needs_manipulation = self.needs_manipulation_analysis(perception)
        needs_sensitivity = self.needs_sensitivity_analysis(perception)

        return GapAnalysisResult(
            needs_retrieval=any([
                needs_memory,
                needs_relationship,
                needs_knowledge,
                needs_social,
                needs_threat,
                needs_manipulation,
                needs_sensitivity,
            ]),
            needs_memory=needs_memory,
            needs_relationship_context=needs_relationship,
            needs_knowledge=needs_knowledge,
            needs_social_context=needs_social,
            needs_threat_analysis=needs_threat,
            needs_manipulation_analysis=needs_manipulation,
            needs_sensitivity_analysis=needs_sensitivity,
            retrieval_actions=retrieval_actions,
        )

    def extract_retrieval_actions(self, tool_calls: list[Any]) -> list[str]:
        for tool_call in tool_calls:
            if tool_call.function.name != "cognitive_action":
                continue

            actions = tool_call.function.arguments.get("actions")
            if isinstance(actions, list):
                return [str(action) for action in actions]

        return []

    def needs_memory(self, perception: PerceptionResult, retrieval_actions: list[str]) -> bool:
        return CognitiveAction.REMEMBER.value in retrieval_actions

    def needs_relationship_context(self, perception: PerceptionResult, retrieval_actions: list[str]) -> bool:
        return CognitiveAction.SOCIAL.value in retrieval_actions

    def needs_knowledge(self, perception: PerceptionResult, retrieval_actions: list[str]) -> bool:
        return any(action in retrieval_actions for action in [
            CognitiveAction.RESEARCH.value,
            CognitiveAction.RECALLKNOWLEDGE.value,
        ])

    def needs_social_context(self, perception: PerceptionResult, retrieval_actions: list[str]) -> bool:
        return CognitiveAction.SOCIAL.value in retrieval_actions

    def needs_threat_analysis(self, perception: PerceptionResult) -> bool:
        # TODO: Drive threat retrieval from explicit threat perception or model output.
        return False

    def needs_manipulation_analysis(self, perception: PerceptionResult) -> bool:
        # TODO: Drive manipulation analysis from perception once implemented.
        return False

    def needs_sensitivity_analysis(self, perception: PerceptionResult) -> bool:
        # TODO: Drive sensitivity analysis from perception once implemented.
        return False


class RetrievalStage:
    def __init__(self, character: Any):
        self.character = character

    def create_plan(self, perception: PerceptionResult, gap_analysis: GapAnalysisResult) -> RetrievalPlan:
        logger.verbose("Creating retrieval plan")
        filters: list[dict[str, Any]] = []

        for tool_call in perception.tool_calls:
            if tool_call.function.name != "cognitive_action":
                continue

            args = tool_call.function.arguments
            filter_value = self.character.cognitive_action(**args)
            if filter_value is not None:
                filters.append(filter_value)

        return RetrievalPlan(
            requires_retrieval=gap_analysis.needs_retrieval and len(filters) > 0,
            prompt=perception.normalized_prompt,
            filters=filters[:1],
            retrieval_actions=gap_analysis.retrieval_actions,
        )

    def run(self, plan: RetrievalPlan, gap_analysis: GapAnalysisResult) -> RetrievedContext:
        logger.verbose("Running retrieval stage, requires_retrieval=%s", plan.requires_retrieval)
        if not plan.requires_retrieval:
            return RetrievedContext(
                threat_analysis=self.analyze_threat(plan),
                manipulation_analysis=self.evaluate_manipulation(plan),
                sensitivity_analysis=self.evaluate_topic_sensitivity(plan),
                trust_respect_impact=self.evaluate_trust_respect_impact(plan),
                fear_suspicion=self.evaluate_fear_suspicion(plan),
                internal_conflict=self.evaluate_internal_conflict(plan),
                belief_updates=self.propose_belief_updates(plan),
                goal_updates=self.propose_goal_updates(plan),
                next_action_plan=self.plan_next_action(plan),
            )

        filter_value = plan.filters[0] if len(plan.filters) > 0 else None

        memory_context = self.recall_memory(plan) if gap_analysis.needs_memory else ""
        relationship_context = self.recall_relationship(plan) if gap_analysis.needs_relationship_context else ""
        knowledge_context = self.recall_knowledge(plan) if gap_analysis.needs_knowledge else ""
        social_context = self.evaluate_social_context(plan) if gap_analysis.needs_social_context else ""
        general_context = self.character.db.query_text(
            prompt=plan.prompt,
            filter=filter_value,
            stage_name="RetrievalStage.run",
        )

        combined_context = "\n".join([text for text in [
            general_context,
            memory_context,
            relationship_context,
            knowledge_context,
            social_context,
        ] if text != ""])

        return RetrievedContext(
            combined_context=combined_context,
            memory_context=memory_context,
            relationship_context=relationship_context,
            knowledge_context=knowledge_context,
            social_context=social_context,
            threat_analysis=self.analyze_threat(plan),
            manipulation_analysis=self.evaluate_manipulation(plan),
            sensitivity_analysis=self.evaluate_topic_sensitivity(plan),
            trust_respect_impact=self.evaluate_trust_respect_impact(plan),
            fear_suspicion=self.evaluate_fear_suspicion(plan),
            internal_conflict=self.evaluate_internal_conflict(plan),
            belief_updates=self.propose_belief_updates(plan),
            goal_updates=self.propose_goal_updates(plan),
            next_action_plan=self.plan_next_action(plan),
        )

    def recall_memory(self, plan: RetrievalPlan) -> str:
        # TODO: Split memory retrieval from generic retrieval into its own indexed source.
        return ""

    def recall_relationship(self, plan: RetrievalPlan) -> str:
        # TODO: Retrieve structured relationship context independently from raw lore.
        return ""

    def recall_knowledge(self, plan: RetrievalPlan) -> str:
        # TODO: Retrieve topic-specific knowledge from dedicated knowledge stores.
        return ""

    def evaluate_social_context(self, plan: RetrievalPlan) -> str:
        # TODO: Compute social framing guidance from prior relationship history.
        return ""

    def analyze_threat(self, plan: RetrievalPlan) -> str:
        # TODO: Analyze threatening content to inform future strategy selection.
        return ""

    def evaluate_manipulation(self, plan: RetrievalPlan) -> str:
        # TODO: Evaluate manipulation risk and expose it to downstream stages.
        return ""

    def evaluate_topic_sensitivity(self, plan: RetrievalPlan) -> str:
        # TODO: Evaluate topic sensitivity and produce guidance for disclosure.
        return ""

    def evaluate_trust_respect_impact(self, plan: RetrievalPlan) -> str:
        # TODO: Model the trust and respect implications of the current turn.
        return ""

    def evaluate_fear_suspicion(self, plan: RetrievalPlan) -> str:
        # TODO: Estimate fear or suspicion changes caused by the player prompt.
        return ""

    def evaluate_internal_conflict(self, plan: RetrievalPlan) -> str:
        # TODO: Surface internal conflicts that should influence the reply.
        return ""

    def propose_belief_updates(self, plan: RetrievalPlan) -> list[str]:
        # TODO: Suggest belief updates based on the interpreted turn.
        return []

    def propose_goal_updates(self, plan: RetrievalPlan) -> list[str]:
        # TODO: Suggest goal updates based on the interpreted turn.
        return []

    def plan_next_action(self, plan: RetrievalPlan) -> str:
        # TODO: Plan follow-up actions or quest-level intent outside the reply.
        return ""


class StrategyStage:
    def __init__(self, character: Any):
        self.character = character

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> StrategyResult:
        logger.verbose("Running strategy stage")
        intention = ""
        immediate_action = "keep_talking"
        new_sentiment = None
        sentiment_reasoning = ""

        for tool_call in perception.tool_calls:
            args = tool_call.function.arguments
            tool = tool_call.function.name

            if tool == "generate_npc_intention":
                intention += self.character.generate_npc_intention(**args)
            elif tool == "immediate_action":
                immediate_action = str(args.get("action", "keep_talking"))
            elif tool == "change_sentiment":
                new_sentiment = args.get("new_sentiment")
                sentiment_reasoning = str(args.get("reasoning", ""))

        return StrategyResult(
            intention=intention,
            conversation_goal=self.select_conversation_goal(initial_context, perception, retrieved_context),
            risk_level=self.assess_risk(initial_context, perception, retrieved_context),
            disclosure_level=self.decide_disclosure(initial_context, perception, retrieved_context),
            social_strategy=self.select_social_strategy(initial_context, perception, retrieved_context),
            tone=self.select_tone(initial_context, perception, retrieved_context),
            verbosity=self.select_verbosity(initial_context, perception, retrieved_context),
            conversation_move=self.select_conversation_move(initial_context, perception, retrieved_context),
            immediate_action=immediate_action,
            new_sentiment=str(new_sentiment) if new_sentiment is not None else None,
            sentiment_reasoning=sentiment_reasoning,
        )

    def select_conversation_goal(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Choose the concrete conversation goal from structured reasoning.
        return "answer_plainly"

    def assess_risk(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Compute risk level from threat, sensitivity, and social context.
        return "low"

    def decide_disclosure(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Decide how much information the NPC should reveal.
        return "normal"

    def select_social_strategy(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Select a social strategy such as trust-building or deception.
        return "neutral"

    def select_tone(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Select the response tone from emotion and social strategy.
        return "in_character"

    def select_verbosity(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Choose how concise or expansive the reply should be.
        return "normal"

    def select_conversation_move(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
    ) -> str:
        # TODO: Choose the next conversational move from the strategy result.
        return "answer"


class ResponseStage:
    def __init__(self, character: Any):
        self.character = character

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
    ) -> ResponseResult:
        final_prompt = self.character.create_answer_prompt(
            prompt=perception.normalized_prompt,
            sentiment=initial_context.sentiment,
            intention=self.compose_intention(strategy),
            context=self.compose_context(initial_context, retrieved_context, strategy),
        )

        logger.verbose("Response stage assembled final prompt")
        logger.debug("Final prompt: %s", final_prompt)
        response = self.character.db.generate_text(final_prompt, stage_name="ResponseStage")

        return ResponseResult(reply=response, final_prompt=final_prompt)

    def compose_intention(self, strategy: StrategyResult) -> str:
        parts = [strategy.intention]
        parts.append(f"goal={strategy.conversation_goal}")
        parts.append(f"risk={strategy.risk_level}")
        parts.append(f"disclosure={strategy.disclosure_level}")
        parts.append(f"social_strategy={strategy.social_strategy}")
        parts.append(f"tone={strategy.tone}")
        parts.append(f"verbosity={strategy.verbosity}")
        parts.append(f"move={strategy.conversation_move}")
        return " | ".join([part for part in parts if part != ""])

    def compose_context(
        self,
        initial_context: InitialContext,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
    ) -> str:
        sections = [
            retrieved_context.combined_context,
            f"Relationship Summary: {initial_context.relationship_summary}" if initial_context.relationship_summary != "" else "",
            f"Active Goals: {', '.join(initial_context.active_goals)}" if len(initial_context.active_goals) > 0 else "",
            f"Recent Turns: {' | '.join(initial_context.recent_turns)}" if len(initial_context.recent_turns) > 0 else "",
            f"Beliefs: {', '.join(initial_context.belief_state)}" if len(initial_context.belief_state) > 0 else "",
            f"Threat Analysis: {retrieved_context.threat_analysis}" if retrieved_context.threat_analysis != "" else "",
            f"Manipulation Analysis: {retrieved_context.manipulation_analysis}" if retrieved_context.manipulation_analysis != "" else "",
            f"Sensitivity Analysis: {retrieved_context.sensitivity_analysis}" if retrieved_context.sensitivity_analysis != "" else "",
            f"Trust/Respect Impact: {retrieved_context.trust_respect_impact}" if retrieved_context.trust_respect_impact != "" else "",
            f"Fear/Suspicion: {retrieved_context.fear_suspicion}" if retrieved_context.fear_suspicion != "" else "",
            f"Internal Conflict: {retrieved_context.internal_conflict}" if retrieved_context.internal_conflict != "" else "",
            f"Planned Next Action: {retrieved_context.next_action_plan}" if retrieved_context.next_action_plan != "" else "",
        ]

        return "\n".join([section for section in sections if section != ""])


class TerminalUpdateStage:
    def __init__(self, character: Any):
        self.character = character

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        strategy: StrategyResult,
        response: ResponseResult,
    ) -> TerminalUpdateResult:
        logger.verbose("Running terminal update stage")
        return TerminalUpdateResult(
            sentiment=strategy.new_sentiment,
            sentiment_reasoning=strategy.sentiment_reasoning,
            immediate_action=strategy.immediate_action,
            relationship_update=self.update_relationship(initial_context, perception, response),
            belief_update=self.update_beliefs(initial_context, perception, retrieved_context, response),
            goal_update=self.update_goals(initial_context, perception, retrieved_context, response),
            store_memory=self.store_memory(initial_context, perception, response),
            external_actions=self.trigger_external_actions(initial_context, perception, response),
        )

    def update_relationship(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        response: ResponseResult,
    ) -> StateUpdate:
        # TODO: Persist relationship changes for future turns.
        return StateUpdate()

    def update_beliefs(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        response: ResponseResult,
    ) -> StateUpdate:
        # TODO: Persist or enqueue belief changes for the NPC state model.
        return StateUpdate(values=list(retrieved_context.belief_updates))

    def update_goals(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        response: ResponseResult,
    ) -> StateUpdate:
        # TODO: Persist or enqueue goal changes for future planning.
        return StateUpdate(values=list(retrieved_context.goal_updates))

    def store_memory(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        response: ResponseResult,
    ) -> bool:
        # TODO: Store notable memories for future retrieval.
        return False

    def trigger_external_actions(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        response: ResponseResult,
    ) -> list[str]:
        # TODO: Trigger world actions that happen outside the spoken reply.
        return []
