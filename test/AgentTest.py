from models import AgentJudgeResult, AgentTestPrompt
from classes.Character import Character
from dataclasses import asdict
import os
import csv
from typing import Any, Callable, Sequence
from logger import configure_logging, get_logger

script_dir = os.path.dirname(__file__)
configure_logging()
logger = get_logger(__name__)

class AgentTest:
    def export_data(self, data: list[Any], columns: Sequence[str], path: str) -> None:
         with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=";")
            writer.writeheader()
            rows = [record.model_dump() for record in data]
            writer.writerows(rows)

    def find_tool(self, name: str, prompt: AgentTestPrompt) -> Any | None:
        if prompt.npc_response == None:
            return None
        
        tool_call = next((x for x in prompt.npc_response if x.function.name == name), None)

        return tool_call
    
    def evaluate_cognitive_action(self, prompt: AgentTestPrompt) -> AgentJudgeResult:
        cognitive_action_tool = self.find_tool("cognitive_action", prompt=prompt)

        if cognitive_action_tool == None:
            return AgentJudgeResult(
                tool="cognitive_action",
                args=None,
                expected_invoked=prompt.expected_args.cognitive_action.is_invoked,
                expected_args=prompt.expected_args.cognitive_action.args,
                user_prompt=prompt.user_query,
                raw_response=prompt.npc_response,
                invoked_pass=prompt.expected_args.cognitive_action.is_invoked == False,
                args_pass= 1 if prompt.expected_args.cognitive_action.is_invoked == False else 0
            )
        
        match = 0
        for arg in prompt.expected_args.cognitive_action.args:
            if arg in cognitive_action_tool.function.arguments.get("actions"):
                match += 1

        return AgentJudgeResult(
            tool="cognitive_action",
            args=cognitive_action_tool.function.arguments,
            expected_invoked=prompt.expected_args.cognitive_action.is_invoked,
            expected_args=prompt.expected_args.cognitive_action.args,
            user_prompt=prompt.user_query,
            raw_response=prompt.npc_response,
            invoked_pass=prompt.expected_args.cognitive_action.is_invoked == True,
            args_pass=match / len(prompt.expected_args.cognitive_action.args)
        )
    
    def evaluate_generate_npc_intention(self, prompt: AgentTestPrompt) -> AgentJudgeResult:
        generate_npc_intention_tool = self.find_tool("generate_npc_intention", prompt=prompt)

        if generate_npc_intention_tool == None:
            return AgentJudgeResult(
                tool="generate_npc_intention",
                args=None,
                expected_invoked=prompt.expected_args.generate_npc_intention.is_invoked,
                expected_args=prompt.expected_args.generate_npc_intention.args,
                user_prompt=prompt.user_query,
                raw_response=prompt.npc_response,
                invoked_pass=prompt.expected_args.generate_npc_intention.is_invoked == False,
                args_pass= 1 if prompt.expected_args.generate_npc_intention.is_invoked == False else 0
            )
        
        intention_matches = generate_npc_intention_tool.function.arguments.get("intention") in prompt.expected_args.generate_npc_intention.args

        return AgentJudgeResult(
            tool="generate_npc_intention",
            args=generate_npc_intention_tool.function.arguments,
            expected_invoked=prompt.expected_args.generate_npc_intention.is_invoked,
            expected_args=prompt.expected_args.generate_npc_intention.args,
            user_prompt=prompt.user_query,
            raw_response=prompt.npc_response,
            invoked_pass=prompt.expected_args.generate_npc_intention.is_invoked == True,
            args_pass=intention_matches
        )
    
    def evaluate_immediate_action(self, prompt: AgentTestPrompt) -> AgentJudgeResult:
        immediate_action_tool = self.find_tool("immediate_action", prompt=prompt)

        if immediate_action_tool == None:
            return AgentJudgeResult(
                tool="immediate_action",
                args=None,
                expected_invoked=prompt.expected_args.immediate_action.is_invoked,
                expected_args=prompt.expected_args.immediate_action.args,
                user_prompt=prompt.user_query,
                raw_response=prompt.npc_response,
                invoked_pass=prompt.expected_args.immediate_action.is_invoked == False,
                args_pass= 1 if prompt.expected_args.immediate_action.is_invoked == False else 0
            )
        
        action_matches = immediate_action_tool.function.arguments.get("action") == prompt.expected_args.immediate_action.args

        return AgentJudgeResult(
            tool="immediate_action",
            args=immediate_action_tool.function.arguments,
            expected_invoked=prompt.expected_args.immediate_action.is_invoked,
            expected_args=prompt.expected_args.immediate_action.args,
            user_prompt=prompt.user_query,
            raw_response=prompt.npc_response,
            invoked_pass=prompt.expected_args.immediate_action.is_invoked == True,
            args_pass=action_matches if prompt.expected_args.immediate_action.is_invoked == True else True
        )
    
    def evaluate_change_sentiment(self, prompt: AgentTestPrompt) -> AgentJudgeResult:
        change_sentiment_tool = self.find_tool("change_sentiment", prompt=prompt)

        if change_sentiment_tool == None:
            return AgentJudgeResult(
                tool="change_sentiment",
                args=None,
                expected_invoked=prompt.expected_args.change_sentiment.is_invoked,
                expected_args=prompt.expected_args.change_sentiment.args,
                user_prompt=prompt.user_query,
                raw_response=prompt.npc_response,
                invoked_pass=prompt.expected_args.change_sentiment.is_invoked == False,
                args_pass= 1 if prompt.expected_args.change_sentiment.is_invoked == False else 0
            )
        
        sentiment_matches = change_sentiment_tool.function.arguments.get("new_sentiment") in prompt.expected_args.change_sentiment.args

        return AgentJudgeResult(
            tool="change_sentiment",
            args=change_sentiment_tool.function.arguments,
            expected_invoked=prompt.expected_args.change_sentiment.is_invoked,
            expected_args=prompt.expected_args.change_sentiment.args,
            user_prompt=prompt.user_query,
            raw_response=prompt.npc_response,
            invoked_pass=prompt.expected_args.change_sentiment.is_invoked == True,
            args_pass=sentiment_matches if prompt.expected_args.change_sentiment.is_invoked == True else True
        )

    def evaluate(self, prompt: AgentTestPrompt) -> list[AgentJudgeResult]:
        cognitive_action_res = self.evaluate_cognitive_action(prompt)
        generate_npc_intention_res = self.evaluate_generate_npc_intention(prompt)
        immediate_action_res = self.evaluate_immediate_action(prompt)
        change_sentiment_res = self.evaluate_change_sentiment(prompt)

        return list([cognitive_action_res, generate_npc_intention_res, immediate_action_res, change_sentiment_res])

    def evaluate_prompts(self, character: Character, prompts: list[AgentTestPrompt], tools: list[Callable]) -> None:
        all_results: list[AgentJudgeResult] = []
        all_prompts: list[AgentTestPrompt] = []

        prompts_path = os.path.join(script_dir, "results/" + character.name + "_agent_prompts.csv")
        results_path = os.path.join(script_dir, "results/" + character.name + "_agent_results.csv")

        prompts_columns = list(AgentTestPrompt.model_fields.keys())
        results_columns = list(AgentJudgeResult.model_fields.keys())

        logger.info("Start generating NPC answers")
        for (i, prompt) in enumerate(prompts):
            prompt.npc_response = character.agent.prompt_agent(
                name=character.name,
                pl_list=character.pl_list,
                situation=character.situation,
                prompt=prompt.user_query,
                sentiment=character.sentiment,
                tools=tools
            )

            all_prompts.append(prompt)
            logger.info("Generated NPC answers: %s / %s", i, len(prompts))
        
        self.export_data(all_prompts, prompts_columns, prompts_path)

        for(i, testPrompt) in enumerate(all_prompts):
            result = self.evaluate(prompt=testPrompt)

            if result != None:
                all_results.extend(result)

            logger.info("Evaluated %s / %s", i, len(prompts))

        self.export_data(all_results, results_columns, results_path)
