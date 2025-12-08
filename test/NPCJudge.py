from dataclasses import dataclass, fields, asdict
from typing import Dict, Any, Type
import csv
import json
import re
import ollama
from classes.Character import Character
import os
from models import TestPrompt, PromptCategory

@dataclass
class JudgeResult:
    prompt: str
    response: str
    category: PromptCategory

    character_consistency: int
    voice_consistency: int
    sentiment_match: int
    lore_consistency: int
    context_usage: int
    breaks_character: bool

    explanation_character_consistency: str
    explanation_voice_consistency: str
    explanation_sentiment_match: str
    explanation_lore_consistency: str
    explanation_context_usage: str

    # rule-based extras
    mentions_ooc: bool
    length_tokens: int

# judge_model = "mixtral:8x7b-instruct-v0.1-q5_K_M"
judge_model = "mistral:7b-instruct-v0.3-q8_0"

script_dir = os.path.dirname(__file__)

class NpcJudge:
    """
    Judges NPC replies against:
    - character sheet (PList)
    - example dialogues (Ali:Chat style)
    - sentiment, situation, context
    """

    # -------------- public API --------------

    def evaluate(
        self,
        character: Character,
        context: str,
        prompt: TestPrompt,
    ) -> JudgeResult | None:
        if prompt.npc_response == None:
            return None
        
        npc_reply = prompt.npc_response
        """Evaluate a single NPC reply and return metrics."""
        judge_prompt = self.build_judge_prompt(
            character=character,
            context=context,
            user_message=prompt.user_query,
            npc_reply=prompt.npc_response, # type: ignore
        )

        raw_output = self.call_model(judge_prompt)
        parsed = self.parse_judge_output(raw_output)

        # rule-based checks
        mentions_ooc = self.check_ooc_markers(npc_reply)
        approx_tokens = self.approx_token_count(npc_reply)

        # If the judge didn't return something, set safe defaults
        result = JudgeResult(
            prompt=prompt.user_query,
            category=prompt.category,
            response=npc_reply,
            character_consistency=parsed.get("character_consistency", "none"),
            voice_consistency=parsed.get("voice_consistency", "none"),
            sentiment_match=parsed.get("sentiment_match", "none"),
            lore_consistency=parsed.get("lore_consistency", "none"),
            context_usage=parsed.get("context_usage", "none"),
            breaks_character=parsed.get("breaks_character", "none") or mentions_ooc,
            mentions_ooc=mentions_ooc,
            length_tokens=approx_tokens,
            explanation_character_consistency=parsed.get("explanation_character_consistency", "no explanation"),
            explanation_voice_consistency=parsed.get("explanation_voice_consistency", "no explanation"),
            explanation_sentiment_match=parsed.get("explanation_sentiment_match", "no explanation"),
            explanation_lore_consistency=parsed.get("explanation_lore_consistency", "no explanation"),
            explanation_context_usage=parsed.get("explanation_context_usage", "no explanation")
        )
        return result

    # -------------- prompt building --------------

    def build_judge_prompt(
        self,
        character: Character,
        context: str,
        user_message: str,
        npc_reply: str,
    ) -> str:
        return f"""
        You are an evaluation assistant. You will judge whether an NPC reply matches its character sheet, sentiment, and world context.

        Return ONLY valid JSON with no extra text.

        Here is the information:

        [Character Definition]
        {character.pl_list}

        [Example Dialogues]
        {character.ali_chat}

        [Current Situation]
        {character.situation}

        [Sentiment Towards Player]
        {character.init_sentiment}

        [Relevant Context]
        {context}

        [User Message]
        {user_message}

        [NPC Reply]
        {npc_reply}

        Now evaluate the NPC reply on these metrics, each from 1 to 10 (integers only):

        - character_consistency: How well does the reply match the character's personality, motivations, and behavior?
        - voice_consistency: How well does the reply match the character's voice, speaking style, and POV?
        - sentiment_match: How well does the reply match the described sentiment towards the player?
        - lore_consistency: How well does the reply stay consistent with the world and context facts given?
        - context_usage: How well does the reply make use of the relevant context (memories, world facts, etc.)?

        If a property cannot be evaluated due to lack of evidence, explain that clearly in your reasoning.

        When evaluating lore consistency, distinguish between:

        (1) Allowed environmental embellishment:
            - Adding descriptive or atmospheric details (e.g., lighting, weather, generic objects, smells, crowds).
            - These do NOT alter world facts or imply new canonical information.
            - Do NOT penalize these.

        (2) Forbidden lore invention:
            - Introducing new world facts not present in the character sheet or context.
            - Inventing new locations, factions, NPCs, or events with canonical significance.
            - Giving directions, instructions, or knowledge that contradicts or extends the known world.
            - Inventing player-relevant information (e.g., quest hints, important geography).
            - Claiming knowledge the character should not have.

        If the reply includes forbidden lore invention, lower the `lore_consistency` score to 1 or 2.
        You MUST explain which element of the reply was invented in your reasoning.

        If the reply only includes harmless environmental decoration, do NOT penalize it.

        Also report:

        - breaks_character: true/false – whether the reply breaks character, mentions being an AI or language model, or goes out-of-world.

        Return JSON with this exact schema:

        {{
        "character_consistency": 1,
        "voice_consistency": 1,
        "sentiment_match": 1,
        "lore_consistency": 1,
        "context_usage": 1,
        "breaks_character": false,
        "explanation_character_consistency": "Short explanation of why you gave this score to character_consistency."
        "explanation_voice_consistency": "Short explanation of why you gave this score to voice_consistency."
        "explanation_sentiment_match": "Short explanation of why you gave this score to sentiment_match."
        "explanation_lore_consistency": "Short explanation of why you gave this score to lore_consistency."
        "explanation_context_usage": "Short explanation of why you gave this score to context_usage."
        }}
        """.strip()

    # -------------- LLM call (to be implemented by you) --------------

    def call_model(self, prompt: str) -> str:
        """
        Call your judge model here.

        For example, with Ollama's /api/generate or /api/chat.
        This is intentionally left abstract so you can plug in what you use.
        """
        return ollama.generate(model=judge_model, prompt=prompt)["response"]

    def parse_judge_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Attempt to parse JSON from the model output.
        Falls back to safe defaults on error.
        """
        # Sometimes models wrap JSON in markdown fences; strip them
        cleaned = raw_output.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
            return data
        except json.JSONDecodeError:
            # Fallback: safe defaults with explanation
            return {
                "character_consistency": 1,
                "voice_consistency": 1,
                "sentiment_match": 1,
                "lore_consistency": 1,
                "context_usage": 1,
                "breaks_character": False,
                "explanation": f"Failed to parse JSON from judge output: {cleaned[:200]}",
            }

    def check_ooc_markers(self, text: str) -> bool:
        """Check for obvious out-of-character / AI markers."""
        lower = text.lower()
        patterns = [
            "as an ai",
            "as a language model",
            "i am an ai",
            "i am a language model",
            "out of character",
            "ooc:",
            "<debug>",
            "<|user|>",
            "<|assistant|>",
        ]
        return any(p in lower for p in patterns)

    def approx_token_count(self, text: str) -> int:
        """
        Rough token approximation: 1 token ≈ 3/4 words.
        Replace with a real tokenizer if you like.
        """
        words = text.split()
        return int(len(words) * 0.75)
    
    def export_data(self, data, columns, path):
         with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=";")
            writer.writeheader()
            writer.writerows([asdict(r) for r in data])

    def evaluate_prompts(self, character: Character, prompts: list[TestPrompt]):
        all_results: list[JudgeResult] = []
        all_prompts: list[TestPrompt] = []

        prompts_path = os.path.join(script_dir, "results/" + character.name + "_prompts.csv")
        results_path = os.path.join(script_dir, "results/" + character.name + "_results.csv")

        prompts_columns = [f.name for f in fields(TestPrompt)]
        results_columns = [f.name for f in fields(JudgeResult)]

        print("Start generating NPC answers")
        for (i, prompt) in enumerate(prompts):
            prompt.npc_response = character.prompt(prompt.user_query)

            all_prompts.append(prompt)
            print("Generated NPC answers: " + str(i) + " / " + str(len(prompts)))
        
        self.export_data(all_prompts, prompts_columns, prompts_path)

        for(i, testPrompt) in enumerate(all_prompts):
            result = self.evaluate(
                character=character,
                context="no context",
                prompt=testPrompt
            )

            if result != None:
                all_results.append(result)

            print("Evaluated " + str(i) + " / " + str(len(prompts)))

        self.export_data(all_results, results_columns, results_path)