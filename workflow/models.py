from dataclasses import dataclass, field
from typing import Any


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
class PerceptionResult:
    raw_prompt: str
    normalized_prompt: str
    jailbreak_detected: bool = False
    player_intent: str = "unknown"
    player_emotion: str = "neutral"
    request_type: str = "general"
    topic: str = ""
    is_ambiguous: bool = False
    threat_signal: str = "none"
    manipulation_signal: str = "none"
    topic_sensitivity: str = "normal"
    tool_calls: list[Any] = field(default_factory=list)
    retrieval_reasoning: str = ""


@dataclass
class GapAnalysisResult:
    needs_retrieval: bool
    needs_memory: bool = False
    needs_relationship_context: bool = False
    needs_knowledge: bool = False
    needs_social_context: bool = False
    needs_threat_analysis: bool = False
    needs_manipulation_analysis: bool = False
    needs_sensitivity_analysis: bool = False
    retrieval_actions: list[str] = field(default_factory=list)


@dataclass
class RetrievalPlan:
    requires_retrieval: bool
    prompt: str
    filters: list[dict[str, Any]] = field(default_factory=list)
    retrieval_actions: list[str] = field(default_factory=list)


@dataclass
class RetrievedContext:
    combined_context: str = ""
    memory_context: str = ""
    relationship_context: str = ""
    knowledge_context: str = ""
    social_context: str = ""
    threat_analysis: str = ""
    manipulation_analysis: str = ""
    sensitivity_analysis: str = ""
    trust_respect_impact: str = ""
    fear_suspicion: str = ""
    internal_conflict: str = ""
    belief_updates: list[str] = field(default_factory=list)
    goal_updates: list[str] = field(default_factory=list)
    next_action_plan: str = ""


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
    immediate_action: str = "keep_talking"
    new_sentiment: str | None = None
    sentiment_reasoning: str = ""


@dataclass
class ResponseResult:
    reply: str
    final_prompt: str


@dataclass
class StateUpdate:
    changed: bool = False
    summary: str = ""
    values: list[str] = field(default_factory=list)


@dataclass
class TerminalUpdateResult:
    sentiment: str | None = None
    sentiment_reasoning: str = ""
    immediate_action: str = "keep_talking"
    relationship_update: StateUpdate = field(default_factory=StateUpdate)
    belief_update: StateUpdate = field(default_factory=StateUpdate)
    goal_update: StateUpdate = field(default_factory=StateUpdate)
    store_memory: bool = False
    external_actions: list[str] = field(default_factory=list)


@dataclass
class TurnResult:
    initial_context: InitialContext
    perception: PerceptionResult
    gap_analysis: GapAnalysisResult
    retrieval_plan: RetrievalPlan
    retrieved_context: RetrievedContext
    strategy: StrategyResult
    response: ResponseResult
    terminal_update: TerminalUpdateResult
