from logger import get_logger
from workflow.models import (
    AppraisalResult,
    EmotionResult,
    InitialContext,
    PerceptionResult,
    ResponseResult,
    RetrievedContext,
    StateUpdate,
    StrategyResult,
    TerminalUpdateResult,
)
from workflow.stages.base import LLMStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class TerminalUpdateStage(LLMStage):
    def get_prompt(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        appraisal: AppraisalResult | None,
        emotion: EmotionResult | None,
        strategy: StrategyResult,
        response: ResponseResult,
    ) -> str:
        return format_prompt(
            "Determine which durable NPC state updates should happen after the response has been produced.",
            [
                ("Player input", perception.raw_prompt),
                ("Response", response.reply),
                ("Current sentiment", initial_context.sentiment),
                ("Appraisal", self.describe_appraisal(appraisal)),
                ("Emotional reaction", self.describe_emotion(emotion)),
                ("Strategy", f"actions={', '.join(strategy.immediate_actions)}, new_sentiment={strategy.new_sentiment}"),
                ("Expected result", "Terminal updates covering sentiment, relationship changes, belief updates, goal updates, memory storage, and external actions."),
            ],
        )

    def run(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        appraisal: AppraisalResult | None,
        emotion: EmotionResult | None,
        strategy: StrategyResult,
        response: ResponseResult,
    ) -> TerminalUpdateResult:
        logger.verbose("Running terminal update stage")
        return TerminalUpdateResult(
            sentiment=strategy.new_sentiment,
            sentiment_reasoning=strategy.sentiment_reasoning,
            sentiment_tags=self.build_sentiment_tags(initial_context, perception, appraisal, emotion, strategy),
            immediate_actions=list(strategy.immediate_actions),
            relationship_update=self.update_relationship(
                initial_context,
                perception,
                response,
                tags=self.build_relationship_tags(initial_context, perception, appraisal, emotion, response),
            ),
            belief_update=self.update_beliefs(
                initial_context,
                perception,
                retrieved_context,
                response,
                tags=self.build_belief_tags(initial_context, perception, retrieved_context, appraisal, emotion, response),
            ),
            goal_update=self.update_goals(
                initial_context,
                perception,
                retrieved_context,
                response,
                tags=self.build_goal_tags(initial_context, perception, retrieved_context, appraisal, emotion, response),
            ),
            store_memory=self.store_memory(initial_context, perception, response),
            memory_tags=self.build_memory_tags(initial_context, perception, appraisal, emotion, response, strategy),
            external_actions=self.trigger_external_actions(initial_context, perception, response, strategy),
        )

    def update_relationship(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        response: ResponseResult,
        tags: list[str] | None = None,
    ) -> StateUpdate:
        # TODO: Persist relationship changes for future turns.
        return StateUpdate(tags=tags or [])

    def update_beliefs(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        response: ResponseResult,
        tags: list[str] | None = None,
    ) -> StateUpdate:
        # TODO: Persist or enqueue belief changes for the NPC state model.
        return StateUpdate(tags=tags or [])

    def update_goals(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        response: ResponseResult,
        tags: list[str] | None = None,
    ) -> StateUpdate:
        # TODO: Persist or enqueue goal changes for future planning.
        return StateUpdate(tags=tags or [])

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
        strategy: StrategyResult,
    ) -> list[str]:
        # TODO: Replace this placeholder passthrough with real game-side effect dispatch.
        return list(strategy.immediate_actions)

    def build_relationship_tags(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        appraisal: AppraisalResult | None,
        emotion: EmotionResult | None,
        response: ResponseResult,
    ) -> list[str]:
        return self.combine_tags(
            ["relationship", "social"],
            self.build_perception_tags(perception),
            self.build_appraisal_tags(appraisal),
            self.build_emotion_tags(emotion),
            [initial_context.sentiment],
        )

    def build_belief_tags(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        appraisal: AppraisalResult | None,
        emotion: EmotionResult | None,
        response: ResponseResult,
    ) -> list[str]:
        return self.combine_tags(
            ["belief", "inference"],
            self.build_perception_tags(perception),
            self.build_retrieved_context_tags(retrieved_context),
            self.build_appraisal_tags(appraisal),
            self.build_emotion_tags(emotion),
        )

    def build_goal_tags(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        retrieved_context: RetrievedContext,
        appraisal: AppraisalResult | None,
        emotion: EmotionResult | None,
        response: ResponseResult,
    ) -> list[str]:
        return self.combine_tags(
            ["goal", "planning"],
            self.build_perception_tags(perception),
            self.build_retrieved_context_tags(retrieved_context),
            self.build_appraisal_tags(appraisal),
            self.build_emotion_tags(emotion),
        )

    def build_sentiment_tags(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        appraisal: AppraisalResult | None,
        emotion: EmotionResult | None,
        strategy: StrategyResult,
    ) -> list[str]:
        return self.combine_tags(
            ["sentiment"],
            [strategy.new_sentiment or "", initial_context.sentiment, perception.player_emotion],
            self.build_perception_tags(perception),
            self.build_appraisal_tags(appraisal),
            self.build_emotion_tags(emotion),
        )

    def build_memory_tags(
        self,
        initial_context: InitialContext,
        perception: PerceptionResult,
        appraisal: AppraisalResult | None,
        emotion: EmotionResult | None,
        response: ResponseResult,
        strategy: StrategyResult,
    ) -> list[str]:
        return self.combine_tags(
            ["memory", "conversation"],
            self.build_perception_tags(perception),
            self.build_appraisal_tags(appraisal),
            self.build_emotion_tags(emotion),
            [strategy.conversation_move, strategy.conversation_goal],
        )

    def build_perception_tags(self, perception: PerceptionResult) -> list[str]:
        return self.combine_tags([
            perception.request_type,
            perception.player_intent,
            perception.player_emotion,
            perception.topic,
            "ambiguous" if perception.is_ambiguous else "",
            perception.threat_signal if perception.threat_signal != "none" else "",
            perception.manipulation_signal if perception.manipulation_signal != "none" else "",
            perception.topic_sensitivity if perception.topic_sensitivity != "normal" else "",
        ])

    def build_retrieved_context_tags(self, retrieved_context: RetrievedContext) -> list[str]:
        tags: list[str] = []
        if retrieved_context.memory_context.strip() != "":
            tags.append("memory_context")
        if retrieved_context.relationship_context.strip() != "":
            tags.append("relationship_context")
        if retrieved_context.knowledge_context.strip() != "":
            tags.append("knowledge_context")
        if retrieved_context.social_context.strip() != "":
            tags.append("social_context")
        return self.combine_tags(tags)

    def build_appraisal_tags(self, appraisal: AppraisalResult | None) -> list[str]:
        if appraisal is None:
            return []
        tags = ["appraisal", appraisal.attribution_source]
        if appraisal.relevance >= 0.7:
            tags.append("high relevance")
        if appraisal.threat >= 0.5:
            tags.append("threat")
        if appraisal.valence < -0.25:
            tags.append("negative valence")
        elif appraisal.valence > 0.25:
            tags.append("positive valence")
        if appraisal.social_self_impact < -0.25:
            tags.append("social injury")
        elif appraisal.social_self_impact > 0.25:
            tags.append("social boost")
        return self.combine_tags(tags)

    def build_emotion_tags(self, emotion: EmotionResult | None) -> list[str]:
        if emotion is None:
            return []
        return self.combine_tags(["emotion", emotion.primary], emotion.secondary)

    def describe_appraisal(self, appraisal: AppraisalResult | None) -> str:
        if appraisal is None:
            return "no appraisal available"
        return "\n".join([
            f"relevance={appraisal.relevance}",
            f"valence={appraisal.valence}",
            f"goal_impact={appraisal.goal_impact}",
            f"social_self_impact={appraisal.social_self_impact}",
            f"threat={appraisal.threat}",
            f"control={appraisal.control}",
            f"attribution_source={appraisal.attribution_source}",
            f"attribution_responsibility={appraisal.attribution_responsibility}",
            f"summary={appraisal.summary}",
        ])

    def describe_emotion(self, emotion: EmotionResult | None) -> str:
        if emotion is None:
            return "no emotional reaction available"
        return "\n".join([
            f"primary={emotion.primary}",
            f"secondary={', '.join(emotion.secondary)}",
            f"intensity={emotion.intensity}",
        ])

    def combine_tags(self, *tag_groups: list[str]) -> list[str]:
        combined_tags: list[str] = []
        seen_tags: set[str] = set()
        for group in tag_groups:
            for tag in group:
                normalized = self.normalize_tag(tag)
                if normalized == "" or normalized in seen_tags:
                    continue
                seen_tags.add(normalized)
                combined_tags.append(normalized)
        return combined_tags

    def normalize_tag(self, tag: str) -> str:
        normalized = str(tag).strip().lower()
        if normalized == "":
            return ""
        return " ".join(normalized.split())
