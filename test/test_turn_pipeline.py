import unittest
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai import ChatCompletionResult
from logger import configure_logging, get_logger
from workflow.models import TurnInput
from workflow import pipeline as workflow_pipeline
from workflow.pipeline import TurnPipeline
from workflow.stages import LLMStage, Stage

test_logger = get_logger(__name__)


@dataclass
class FakeFunction:
    name: str
    arguments: dict


@dataclass
class FakeToolCall:
    function: FakeFunction


class FakeAgent:
    def __init__(
        self,
        tool_calls=None,
        gap_content: str | None = None,
        gap_tool_calls=None,
        perception_content: str | None = None,
        retrieval_summary_content: str | None = None,
    ):
        self.tool_calls = tool_calls or []
        self.gap_content = gap_content
        self.gap_tool_calls = gap_tool_calls or []
        self.perception_content = perception_content
        self.retrieval_summary_content = retrieval_summary_content or "summarized retrieved lore"
        self.strategy_tool_calls = []
        self.prompts: list[str] = []

    def run_prompt(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        self.prompts.append(prompt)
        stage_name = kwargs.get("stage_name", "PerceptionStage")
        if stage_name == "GapAnalysisStage":
            content = self.gap_content
            if content is None:
                content = self.default_gap_content()
            tool_calls = list(self.gap_tool_calls)
        elif stage_name == "RetrievalStage.summarize":
            content = self.retrieval_summary_content
            tool_calls = []
        elif stage_name == "StrategyStage":
            content = ""
            tool_calls = list(self.strategy_tool_calls)
        else:
            content = self.perception_content
            if content is None:
                content = self.default_perception_content()
            tool_calls = list(self.tool_calls)

        test_logger.conversation_event(
            stage_name=stage_name,
            event="decision_model",
            payload={"prompt": prompt},
            ai_request={"messages": ["agent prompt"]},
            ai_response={"content": content, "tool_calls": tool_calls},
            result={"content": content, "tool_calls": tool_calls},
        )
        return ChatCompletionResult(content=content, tool_calls=tool_calls)

    def parse_output(self, raw_output: str, fallback=None):
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return fallback or {}

    def default_gap_content(self) -> str:
        tool_names = [tool_call.function.name for tool_call in self.gap_tool_calls]
        return json.dumps({"tool_names": tool_names})

    def default_perception_content(self) -> str:
        return json.dumps({
            "player_intent": "unknown",
            "player_emotion": "neutral",
            "request_type": "general",
            "topic": "",
            "is_ambiguous": False,
            "threat_signal": "none",
            "manipulation_signal": "none",
            "topic_sensitivity": "normal",
        })


class FakeDB:
    def __init__(self):
        self.query_calls = 0
        self.stage_query_calls: dict[str, int] = {}
        self.prompts: list[str] = []
        self.messages: list[dict[str, str]] = []
        self.response_context_initialized = False

    def seed_response_context(self, system_prompt: str, seed_context_prompt: str):
        if self.response_context_initialized:
            return

        seed_messages = []
        if system_prompt.strip() != "":
            seed_messages.append({"role": "system", "content": system_prompt})
        if seed_context_prompt.strip() != "":
            seed_messages.append({"role": "user", "content": seed_context_prompt})

        self.messages = seed_messages + self.messages
        self.response_context_initialized = True

    def query_text(self, prompt: str, filter=None, stage_name: str = "RetrievalStage"):
        self.query_calls += 1
        self.stage_query_calls[stage_name] = self.stage_query_calls.get(stage_name, 0) + 1
        test_logger.conversation_event(
            stage_name=stage_name,
            event="query_text",
            payload={"prompt": prompt, "filter": filter},
            result={"text": "retrieved lore"},
        )
        return "retrieved lore"

    def generate_text(self, prompt: str, stage_name: str = "ResponseStage") -> str:
        request_messages = list(self.messages)
        request_messages.append({"role": "user", "content": prompt})

        self.prompts.append(prompt)
        self.messages = list(request_messages)
        self.messages.append({"role": "assistant", "content": "npc reply"})
        test_logger.conversation_event(
            stage_name=stage_name,
            event="generate_text",
            payload={"prompt": prompt},
            ai_request={"messages": request_messages},
            ai_response={"content": "npc reply", "tool_calls": []},
            result={"reply": "npc reply"},
        )
        return "npc reply"


class FakeCharacter:
    def __init__(
        self,
        tool_calls=None,
        gap_content: str | None = None,
        gap_tool_calls=None,
        perception_content: str | None = None,
        retrieval_summary_content: str | None = None,
    ):
        self.name = "Mira"
        self.situation = "At the market"
        self.sentiment = "neutral"
        self.pl_list = "Helpful trader"
        self.ali_chat = "Welcome, traveler."
        self.agent = FakeAgent(
            tool_calls=tool_calls,
            gap_content=gap_content,
            gap_tool_calls=gap_tool_calls,
            perception_content=perception_content,
            retrieval_summary_content=retrieval_summary_content,
        )
        self.db = FakeDB()
        self.talk_ongoing = True
        self.applied_sentiment = None

    def cognitive_action(self, actions, reasoning):
        return {"$or": [{"category": "memory"}]}

    def get_relations(self):
        return {"category": "relations"}

    def get_sentiment_filter(self):
        return {"category": "sentiment"}

    def get_memories(self):
        return {"category": "memory"}

    def get_past(self):
        return {"category": "past"}

    def get_faction_knowledge(self):
        return {"category": "faction_knowledge"}

    def get_world_knowledge(self):
        return {"category": "world_knowledge"}

    def generate_npc_intention(self, intention, reasoning):
        return str(intention) + ": " + reasoning

    def immediate_actions(self, action):
        self.talk_ongoing = action != "end_conversation"

    def change_sentiment(self, new_sentiment, reasoning):
        self.applied_sentiment = f"{new_sentiment}: {reasoning}"

class TurnPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_logging("VERBOSE")

    def test_pipeline_runs_with_placeholder_defaults(self):
        character = FakeCharacter()
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="Hello there"))

        self.assertEqual(result.response.reply, "npc reply")
        self.assertEqual(result.gap_analysis.tool_calls, [])
        self.assertEqual(result.perception.raw_prompt, "Hello there")
        self.assertEqual(result.strategy.conversation_goal, "answer_plainly")
        self.assertEqual(result.strategy.immediate_actions, ["keep_talking"])
        self.assertFalse(result.terminal_update.store_memory)

    def test_pipeline_performs_at_most_one_retrieval_pass(self):
        gap_tool_calls = [
            FakeToolCall(FakeFunction("recall_memory", {"reasoning": "Need memory context"})),
            FakeToolCall(FakeFunction("evaluate_social_context", {"reasoning": "Need social context"})),
        ]
        character = FakeCharacter(
            gap_content='{"tool_names": ["recall_memory", "evaluate_social_context"]}',
            gap_tool_calls=gap_tool_calls,
        )
        pipeline = TurnPipeline(character)

        pipeline.run(TurnInput(prompt="Do you remember me?"))
        self.assertEqual(character.db.stage_query_calls.get("RetrievalStage.run"), 2)
        self.assertEqual(len(character.agent.prompts), 5)

    def test_original_prompt_flows_into_response(self):
        character = FakeCharacter()
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="Tell me about the weather."))

        self.assertEqual(result.perception.raw_prompt, "Tell me about the weather.")
        self.assertIn("Tell me about the weather.", result.response.turn_prompt)
        self.assertIn("Tell me about the weather.", result.perception.stage_prompt)
        self.assertIn("Current sentiment towards player", result.perception.stage_prompt)
        self.assertIn("Player input", result.response.turn_prompt)
        self.assertIn("Response strategy", result.response.turn_prompt)
        self.assertEqual(character.db.prompts[-1], result.response.turn_prompt)

    def test_every_stage_exposes_prompt_getter(self):
        character = FakeCharacter()
        pipeline = TurnPipeline(character)

        self.assertIsInstance(pipeline.initial_context_stage, Stage)
        self.assertNotIsInstance(pipeline.initial_context_stage, LLMStage)
        self.assertFalse(hasattr(pipeline.initial_context_stage, "get_prompt"))

        self.assertIsInstance(pipeline.perception_stage, LLMStage)
        self.assertIsInstance(pipeline.gap_analysis_stage, LLMStage)
        self.assertIsInstance(pipeline.retrieval_stage, LLMStage)
        self.assertIsInstance(pipeline.strategy_stage, LLMStage)
        self.assertIsInstance(pipeline.response_stage, LLMStage)
        self.assertIsInstance(pipeline.terminal_update_stage, LLMStage)

        self.assertTrue(callable(pipeline.perception_stage.get_prompt))
        self.assertTrue(callable(pipeline.gap_analysis_stage.get_prompt))
        self.assertTrue(callable(pipeline.retrieval_stage.get_prompt))
        self.assertTrue(callable(pipeline.strategy_stage.get_prompt))
        self.assertTrue(callable(pipeline.response_stage.get_prompt))
        self.assertTrue(callable(pipeline.terminal_update_stage.get_prompt))

    def test_perception_stage_constructs_its_own_prompt(self):
        character = FakeCharacter()
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="Do you know this town?"))

        self.assertEqual(len(character.agent.prompts), 3)
        self.assertEqual(character.agent.prompts[0], result.perception.stage_prompt)
        self.assertIn("Do you know this town?", result.perception.stage_prompt)
        self.assertIn("Character definition", result.perception.stage_prompt)
        self.assertIn("Decision rubric", result.perception.stage_prompt)
        self.assertIn("Return strictly valid JSON", result.perception.stage_prompt)

    def test_perception_stage_parses_strict_json_into_result(self):
        character = FakeCharacter(
            perception_content=json.dumps({
                "player_intent": "seek_information",
                "player_emotion": "curious",
                "request_type": "question",
                "topic": "local history",
                "is_ambiguous": False,
                "threat_signal": "none",
                "manipulation_signal": "subtle_flattery",
                "topic_sensitivity": "normal",
            })
        )
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="Tell me about this town's history."))

        self.assertEqual(result.perception.player_intent, "seek_information")
        self.assertEqual(result.perception.player_emotion, "curious")
        self.assertEqual(result.perception.request_type, "question")
        self.assertEqual(result.perception.topic, "local history")
        self.assertFalse(result.perception.is_ambiguous)
        self.assertEqual(result.perception.threat_signal, "none")
        self.assertEqual(result.perception.manipulation_signal, "subtle_flattery")
        self.assertEqual(result.perception.topic_sensitivity, "normal")
        self.assertEqual(result.perception.tool_calls, [])

    def test_strategy_stage_uses_immediate_action_tools(self):
        character = FakeCharacter()
        character.agent.strategy_tool_calls = [
            FakeToolCall(FakeFunction("open_trade", {"reasoning": "The player is engaging the NPC as a merchant."})),
            FakeToolCall(FakeFunction("keep_talking", {"reasoning": "The conversation should continue during the trade."})),
        ]
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="Show me what you have for sale."))

        self.assertEqual(result.strategy.immediate_actions, ["open_trade", "keep_talking"])
        self.assertEqual(result.terminal_update.external_actions, ["open_trade", "keep_talking"])
        self.assertIn("The strategy is not limited to dialogue alone. If carrying out the strategy would naturally involve an immediate in-world action, the NPC may use the provided action tools.", character.agent.prompts[2])

    def test_strategy_stage_can_end_conversation_via_tool(self):
        character = FakeCharacter()
        character.agent.strategy_tool_calls = [
            FakeToolCall(FakeFunction("alert_guards", {"reasoning": "The player is making a dangerous threat."})),
            FakeToolCall(FakeFunction("end_conversation", {"reasoning": "The NPC refuses further engagement."})),
        ]
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="Tell me or I'll burn this place down."))

        self.assertEqual(result.strategy.immediate_actions, ["alert_guards", "end_conversation"])
        self.assertEqual(result.terminal_update.external_actions, ["alert_guards", "end_conversation"])

    def test_gap_analysis_stage_requests_strict_json(self):
        character = FakeCharacter(
            gap_content='{"tool_names": ["recall_memory"]}'
        )
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="What did I tell you yesterday?"))

        self.assertEqual(result.gap_analysis.tool_calls, [])
        self.assertIn("If more context is needed, call the relevant retrieval tools directly.", character.agent.prompts[1])
        self.assertIn("The tool calls are the decision payload", character.agent.prompts[1])

    def test_gap_analysis_tool_calls_drive_retrieval_collection(self):
        gap_tool_calls = [
            FakeToolCall(FakeFunction("recall_memory", {"reasoning": "Need memory context before answering"}))
        ]
        character = FakeCharacter(
            gap_content='{"tool_names": []}',
            gap_tool_calls=gap_tool_calls,
        )
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="What happened last time?"))

        self.assertEqual(result.gap_analysis.tool_calls, gap_tool_calls)
        self.assertEqual(character.db.stage_query_calls.get("RetrievalStage.run"), 1)
        self.assertEqual(result.retrieved_context.memory_context, "retrieved lore")
        self.assertEqual(result.retrieved_context.combined_context, "summarized retrieved lore")
        self.assertEqual(result.retrieved_context.relationship_context, "no information")
        self.assertEqual(result.retrieved_context.knowledge_context, "no information")
        self.assertEqual(result.retrieved_context.social_context, "no information")

    def test_retrieval_loops_back_to_perception_once_when_gap_tools_exist(self):
        gap_tool_calls = [
            FakeToolCall(FakeFunction("recall_memory", {"reasoning": "Need memory context before answering"}))
        ]
        character = FakeCharacter(
            gap_content='{"tool_names": ["recall_memory"]}',
            gap_tool_calls=gap_tool_calls,
        )
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="What happened here before?"))

        self.assertEqual(len(character.agent.prompts), 5)
        self.assertIn("Additional retrieved context", result.perception.stage_prompt)
        self.assertIn("summarized retrieved lore", result.perception.stage_prompt)

    def test_retrieval_summary_prompt_filters_for_evidently_relevant_context(self):
        gap_tool_calls = [
            FakeToolCall(FakeFunction("recall_knowledge", {"reasoning": "Need knowledge context before answering"}))
        ]
        character = FakeCharacter(
            gap_content='{"tool_names": ["recall_knowledge"]}',
            gap_tool_calls=gap_tool_calls,
        )
        pipeline = TurnPipeline(character)

        pipeline.run(TurnInput(prompt="What do you know about the old temple?"))

        summary_prompt = character.agent.prompts[2]
        self.assertIn("Include only evidently relevant context snippets.", summary_prompt)
        self.assertIn("Refer every included context snippet directly to the player's prompt input.", summary_prompt)
        self.assertIn("Player input: What do you know about the old temple?", summary_prompt)

    def test_retrieval_defaults_unrequested_context_to_no_information(self):
        character = FakeCharacter()
        pipeline = TurnPipeline(character)

        result = pipeline.run(TurnInput(prompt="Hello there"))

        self.assertEqual(result.retrieved_context.combined_context, "no information")
        self.assertEqual(result.retrieved_context.memory_context, "no information")
        self.assertEqual(result.retrieved_context.relationship_context, "no information")
        self.assertEqual(result.retrieved_context.knowledge_context, "no information")
        self.assertEqual(result.retrieved_context.social_context, "no information")

    def test_pipeline_emits_verbose_logs_in_stage_order(self):
        character = FakeCharacter()
        pipeline = TurnPipeline(character)

        with self.assertLogs("workflow.pipeline", level="VERBOSE") as captured:
            pipeline.run(TurnInput(prompt="Hello there"))

        log_output = "\n".join(captured.output)
        expected_sequence = [
            "InitialContextStage started",
            "InitialContextStage completed successfully",
            "PerceptionStage started",
            "PerceptionStage completed successfully",
            "GapAnalysisStage started",
            "GapAnalysisStage completed successfully",
            "RetrievalStage.run started",
            "RetrievalStage.run completed successfully",
            "StrategyStage started",
            "StrategyStage completed successfully",
            "ResponseStage started",
            "ResponseStage completed successfully",
            "TerminalUpdateStage started",
            "TerminalUpdateStage completed successfully",
        ]

        last_index = -1
        for item in expected_sequence:
            current_index = log_output.find(item)
            self.assertGreater(current_index, last_index, item)
            last_index = current_index

    def test_pipeline_persists_stage_trace_to_text_file(self):
        with TemporaryDirectory() as temp_dir:
            token = test_logger.start_conversation_trace(
                root_dir=Path(temp_dir),
                character_name="Mira",
                profile="test-profile",
                providers={"response": "fake:model"},
                persist_enabled=True,
            )
            conversation_id = test_logger.get_conversation_id()
            try:
                character = FakeCharacter()
                pipeline = TurnPipeline(character)
                pipeline.run(TurnInput(prompt="Hello there"))
            finally:
                test_logger.reset_conversation_id(token)

            trace_path = test_logger.get_conversation_trace_path(conversation_id)
            self.assertIsNotNone(trace_path)
            assert trace_path is not None
            self.assertTrue(trace_path.exists())
            content = trace_path.read_text(encoding="utf-8")
            self.assertIn("conversation_id:", content)
            self.assertIn("=== Stage: InitialContextStage ===", content)
            self.assertIn("=== Stage: PerceptionStage ===", content)
            self.assertIn("=== Stage: RetrievalStage.run ===", content)
            self.assertIn("=== Stage: ResponseStage ===", content)
            self.assertIn("=== Stage: TerminalUpdateStage ===", content)
            self.assertIn("=== Stage: TurnPipeline ===", content)
            self.assertIn("event: final_response_output", content)
            self.assertIn("ai_request:", content)
            self.assertIn("ai_response:", content)
            self.assertIn("Hello there", content)
            self.assertIn("npc reply", content)

    def test_pipeline_failure_logs_stage_and_persists_error_marker(self):
        recorded_events = []
        with TemporaryDirectory() as temp_dir:
            token = test_logger.start_conversation_trace(
                root_dir=Path(temp_dir),
                character_name="Mira",
                profile="test-profile",
                providers={"response": "fake:model"},
                persist_enabled=True,
            )
            conversation_id = test_logger.get_conversation_id()
            try:
                character = FakeCharacter()
                pipeline = TurnPipeline(character)

                def raise_failure(*args, **kwargs):
                    raise RuntimeError("stage boom")

                pipeline.perception_stage.run = raise_failure

                original_conversation_event = workflow_pipeline.logger.conversation_event

                def capture_event(**kwargs):
                    recorded_events.append(kwargs)
                    original_conversation_event(**kwargs)

                with patch("workflow.pipeline.logger.conversation_event", side_effect=capture_event):
                    with self.assertLogs("workflow.pipeline", level="ERROR") as captured:
                        with self.assertRaises(RuntimeError):
                            pipeline.run(TurnInput(prompt="Hello there"))
            finally:
                test_logger.reset_conversation_id(token)

            self.assertIn("PerceptionStage failed: stage boom", "\n".join(captured.output))
            self.assertTrue(any(
                event.get("stage_name") == "PerceptionStage"
                and event.get("event") == "stage_failed"
                and event.get("payload") == {"prompt": "Hello there"}
                for event in recorded_events
            ))
            trace_path = test_logger.get_conversation_trace_path(conversation_id)
            self.assertIsNotNone(trace_path)
            assert trace_path is not None
            self.assertTrue(trace_path.exists())
            content = trace_path.read_text(encoding="utf-8")
            self.assertIn("conversation_id:", content)
            self.assertIn("event: stage_failed", content)
            self.assertIn('"prompt": "Hello there"', content)
            self.assertIn('"error": "stage boom"', content)


if __name__ == "__main__":
    unittest.main()
