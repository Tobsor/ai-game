from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Sequence, Any
from ollama._types import Message

ToolCall = Message.ToolCall

class Faction(Enum):
    RACCOON = "raccoon"
    JELLIFISH = "jellyfish"
    ANTS = "ants"
    APES = "apes"
    CHAMELEON = "chameleon"
    FENNEK = "fennek"
    WORLD = "1"

class MetadataType(Enum):
    FACTION = "faction"
    CHARACTER = "character"

class PromptCategory(Enum):
    GENERAL = "general"
    SENTIMENT = "sentiment"
    LORE = "lore"
    MANIPULATION = "manipulation"
    PAST = "past"
    JAILBREAK = "jailbreak"

class MetadataCategory(Enum):
    KNOWLEDGE = "knowledge"
    PAST = "past"
    SENTIMENT = "sentiment"
    MEMORY = "memory"
    RELATIONS = "relations"
    LORE = "lore"

class CognitiveAction(Enum):
    REMEMBER = "remember"
    RESEARCH = "research"
    RECALLKNOWLEDGE = "recall_knowledge"
    SOCIAL = "social_interaction"
    INTROSPECT = "introspect"
    PLAN = "plan"

class NPCAction(Enum):
    KEEP_TALKING = "keep_talking"
    END_CONVERSATION = "end_conversation"

class Sentiment(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SHOCKED = "shocked"
    GRATEFUL = "grateful"
    CONFUSED = "confused"
    STIMLUATED = "stimulated"
    INSULTED = "insulted"
    SKEPTICAL = "skeptical"
    DISAPPOINTED = "disappointed"
    ANGRY = "angry"
    INTERESTED = "interested"
    DISINTERESTED = "disinterested"
    AGITATED = "agitated"
    NERVOUS = "nervous"

class FactionData(BaseModel):
    faction: Faction
    lore: str

class Character(BaseModel):
    name: str
    faction: Faction
    pl_list: str
    ali_chat: str
    knowledge: str
    past: str
    relations: str
    sentiment: str

class Metadata(BaseModel):
    faction: Optional[Faction] = None
    name: Optional[str] = None
    type: Optional[MetadataType] = None
    category: Optional[MetadataCategory] = None

class TestPrompt(BaseModel):
    user_query: str
    npc_response: Optional[str] = None
    category: PromptCategory
    authors_note: str

class ExpectedResult(BaseModel):
    is_invoked: bool
    args: Any

class ExpectedToolArgs(BaseModel):
    cognitive_action: ExpectedResult
    generate_npc_intention: ExpectedResult
    change_sentiment: ExpectedResult
    immediate_actions: ExpectedResult

class AgentTestPrompt(BaseModel):
    user_query: str
    npc_response: Optional[Sequence[ToolCall] | None] = None
    category: PromptCategory
    expected_args: ExpectedToolArgs

class NPCAgentDecision(BaseModel):
    cognitive_actions: list[CognitiveAction]
    npc_intent: str
    action: NPCAction
    sentiment: Sentiment

class AgentJudgeResult(BaseModel):
    tool: str
    args: Any
    expected_invoked: bool
    expected_args: Any
    user_prompt: str
    raw_response: Any
    invoked_pass: bool
    args_pass: float


class StageExpectationMode(str, Enum):
    DETERMINISTIC = "deterministic"
    JUDGE = "judge"


class StageName(str, Enum):
    PERCEPTION = "PerceptionStage"
    GAP_ANALYSIS = "GapAnalysisStage"
    RETRIEVAL_SUMMARIZE = "RetrievalStage.summarize"
    STRATEGY = "StrategyStage"
    RESPONSE = "ResponseStage"


class StageDeterministicCheck(BaseModel):
    metric_name: str
    path: str
    operator: str = "equals"
    value: Any = None


class StageJudgeMetric(BaseModel):
    metric_name: str
    guidance: str = ""


class StageTestPrompt(BaseModel):
    user_query: str
    source_category: PromptCategory
    target_stage: StageName
    expectation_mode: StageExpectationMode
    deterministic_checks: list[StageDeterministicCheck] = Field(default_factory=list)
    judge_metrics: list[StageJudgeMetric] = Field(default_factory=list)
    stage_inputs: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class StageJudgeMetricResult(BaseModel):
    metric_name: str
    score: float
    passed: bool
    explanation: str


class StageEvaluationResult(BaseModel):
    user_query: str
    source_category: PromptCategory
    target_stage: StageName
    expectation_mode: StageExpectationMode
    metric_name: str
    passed: bool
    score: float
    explanation: str
    expected_value: Any = None
    actual_value: Any = None
    stage_output: str
    notes: str = ""
