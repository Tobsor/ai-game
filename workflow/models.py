from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TurnInput:
    prompt: str


@dataclass
class InitialContext:
    character_name: str
    situation: str
    sentiment: str
    character_definition: str
    example_dialogues: str
    relationship_summary: str = ""
    active_goals: list[str] = field(default_factory=list)
    recent_turns: list[str] = field(default_factory=list)
    belief_state: list[str] = field(default_factory=list)


@dataclass
class PerceivedPlayerEmotion:
    primary: str
    intensity: Literal["low", "medium", "high"]
    summary: str


@dataclass
class PerceptionCore:
    situation_summary: str
    interaction_type: str
    topic: str
    perceived_intent: str
    perceived_player_attitude: str
    perceived_player_emotion: PerceivedPlayerEmotion
    perceived_threat: Literal["none", "low", "medium", "high"]
    perceived_manipulation: Literal["none", "low", "medium", "high"]


@dataclass
class PerceptionResult:
    raw_prompt: str
    core: PerceptionCore


@dataclass
class ContextEvidence:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float | None = None
    stage: str = ""


@dataclass
class EnrichmentDecision:
    stage: str
    request: str = ""
    reason: str = ""


@dataclass
class RetrievedContext:
    combined_context: str = ""
    memory_context: str = ""
    relationship_context: str = ""
    knowledge_context: str = ""
    social_context: str = ""
    initial_context: InitialContext | None = None
    evidence: list[ContextEvidence] = field(default_factory=list)
    history: list[EnrichmentDecision] = field(default_factory=list)


@dataclass
class AppraisalResult:
    relevance: float = 0.0
    valence: float = 0.0
    goal_impact: float = 0.0
    social_self_impact: float = 0.0
    threat: float = 0.0
    control: float = 0.5
    attribution_source: str = "unknown"
    attribution_responsibility: float = 0.0
    summary: str = ""
    stage_prompt: str = ""


@dataclass
class EmotionResult:
    primary: str = "neutral"
    secondary: list[str] = field(default_factory=list)
    intensity: float = 0.0
    stage_prompt: str = ""


@dataclass
class StrategyResult:
    intention: str = ""
    conversation_goal: str = "answer_plainly"
    risk_level: str = "low"
    disclosure_level: str = "normal"
    social_strategy: str = "neutral"
    tone: str = "in_character"
    verbosity: str = "normal"
    conversation_move: str = "answer"
    immediate_actions: list[str] = field(default_factory=list)
    new_sentiment: str | None = None
    sentiment_reasoning: str = ""


@dataclass
class ResponseResult:
    reply: str
    turn_prompt: str


@dataclass
class StateUpdate:
    changed: bool = False
    value: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class TerminalUpdateResult:
    sentiment: str | None = None
    sentiment_reasoning: str = ""
    sentiment_tags: list[str] = field(default_factory=list)
    immediate_actions: list[str] = field(default_factory=list)
    relationship_update: StateUpdate = field(default_factory=StateUpdate)
    belief_update: StateUpdate = field(default_factory=StateUpdate)
    goal_update: StateUpdate = field(default_factory=StateUpdate)
    store_memory: bool = False
    memory_tags: list[str] = field(default_factory=list)
    external_actions: list[str] = field(default_factory=list)


@dataclass
class TurnResult:
    initial_context: InitialContext
    perception: PerceptionResult
    retrieved_context: RetrievedContext
    appraisal: AppraisalResult
    emotion: EmotionResult
    strategy: StrategyResult
    response: ResponseResult
    terminal_update: TerminalUpdateResult
