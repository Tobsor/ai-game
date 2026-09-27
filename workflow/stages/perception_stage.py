import json

from workflow.perception_contract import validate_perception
from workflow.enrichment import GapAnalysis
from logger import get_logger
from workflow.models import InitialContext, PerceptionResult, RetrievedContext, TurnInput
from workflow.stages.base import EnrichedStage
from workflow.stages.prompting import format_prompt

logger = get_logger(__name__)


class PerceptionStage(EnrichedStage):
    def get_prompt(
        self,
        turn_input: TurnInput,
        initial_context: InitialContext,
        retrieved_context: RetrievedContext | None = None,
    ) -> str:
        return GapAnalysis.prompt(format_prompt(
            f"You are simulating how the NPC {initial_context.character_name} perceives the player's latest message. Do not generate dialogue. Analyze the player's prompt from this NPC's perspective so downstream stages can act on that perception. Evaluate the message through this character's personality, knowledge, sentiment, and social awareness, and only infer signals this character would realistically notice or understand. Reflect on whether this NPC is perceptive enough to recognize subtle manipulation, threat, ambiguity, or emotional subtext rather than assuming perfect insight.",
            [
                ("Situation", initial_context.situation),
                ("Current sentiment towards player", initial_context.sentiment),
                ("Character definition", initial_context.character_definition),
                ("Belief state", "\n".join(initial_context.belief_state)),
                ("Example dialogues", initial_context.example_dialogues),
                (
                    "Relevant relationship and goals",
                    "\n".join([
                        initial_context.relationship_summary,
                        "Active goals: " + ", ".join(initial_context.active_goals) if len(initial_context.active_goals) > 0 else "",
                    ]).strip(),
                ),
                (
                    "Recent conversation state",
                    "\n".join(initial_context.recent_turns),
                ),
                (
                    "Decision rubric",
                    "\n".join([
                        "Evaluate every interpretive field strictly from this NPC's viewpoint, consistent with their profile, knowledge, biases, current state, and perceptiveness; report what they would notice, believe, or suspect, not an objective analyst's assessment or the player's hidden truth.",
                        "Apply this perspective to all core fields and context requests: identify only context needs this NPC would recognize, and preserve plausible misunderstandings or missed signals rather than correcting them with outside knowledge.",
                        "Return raw_prompt and core plus the two enrichment fields specified below as a JSON object; copy raw_prompt verbatim from Player input.",
                        "core contains situation_summary, interaction_type, topic, perceived_intent, perceived_player_attitude, perceived_player_emotion, perceived_threat, and perceived_manipulation.",
                        "Use one concise sentence each for situation_summary, perceived_intent, and perceived_player_attitude; acknowledge uncertainty and relevant personal sensitivity without inventing facts.",
                        "situation_summary describes what the NPC believes is happening in this interaction; perceived_intent describes what the player seems to want to achieve; perceived_player_attitude describes the player's apparent stance toward the NPC, such as respectful, dismissive, or cooperative.",
                        "interaction_type identifies the player's communicative act as one categorical string (such as question, greeting, threat, negotiation, or social_bid); topic names the main subject as one string, not a list or multiple categories.",
                        "perceived_player_emotion must be a nested JSON object, never a string or array, with exactly primary (an emotion label), intensity (low, medium, or high), and summary (one sentence grounding the inference); describe the player's inferred emotional state, distinct from their attitude and the NPC's reaction.",
                        "perceived_threat grades perceived danger or harm; perceived_manipulation grades attempted influence through deception, coercion, or exploitation. Each uses none, low, medium, or high, reflecting the NPC's assessment rather than objective certainty.",
                        "Use neutral with low intensity when the player shows no particular emotion; describe uncertainty in the summary when evidence is unclear.",
                        "Your entire output must be one valid JSON object strictly matching the specified structure and enrichment fields: exact names, nesting, types, and allowed intensity values. All cognitive fields are required and nonblank; do not rename or flatten fields, use null, or include commentary, markdown, or code fences. Only context_request may be empty.",
                        "Do not produce NPC emotion, appraisal, strategy, or dialogue.",
                    ]),
                ),
                ("Player input", turn_input.prompt),
                (
                    "Additional retrieved context",
                    retrieved_context.combined_context if retrieved_context is not None else "",
                ),
                (
                    "Required JSON structure",
                    "Use this cognitive structure plus the enrichment fields below, replacing placeholders with your assessment and choosing intensity values from the rubric; retain raw_prompt exactly as shown.\n"
                    + json.dumps({
                        "raw_prompt": turn_input.prompt,
                        "core": {
                            "situation_summary": "<one sentence>",
                            "interaction_type": "<one category>",
                            "topic": "<one topic>",
                            "perceived_intent": "<one sentence>",
                            "perceived_player_attitude": "<one sentence>",
                            "perceived_player_emotion": {
                                "primary": "<emotion>",
                                "intensity": "<low|medium|high>",
                                "summary": "<one sentence>",
                            },
                            "perceived_threat": "<none|low|medium|high>",
                            "perceived_manipulation": "<none|low|medium|high>",
                        },
                    }, ensure_ascii=False, indent=2),
                ),
            ],
        ), retrieved_context or RetrievedContext())

    def run(
        self,
        turn_input: TurnInput,
        initial_context: InitialContext,
        retrieved_context: RetrievedContext | None = None,
    ) -> PerceptionResult:
        context = retrieved_context if retrieved_context is not None else RetrievedContext()
        return self.enrichment.run(
            stage_name="PerceptionStage", initial_context=initial_context, context=context,
            build_prompt=lambda: self.get_prompt(turn_input, initial_context, context),
            parse_result=lambda payload, response, prompt: validate_perception(payload, turn_input.prompt),
        )
