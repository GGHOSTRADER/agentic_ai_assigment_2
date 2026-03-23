import json
import os
import re
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from tools import search


load_dotenv()


class Agent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        self.model = "gemini-2.5-flash-lite"
        self.max_loops = 5

        # Explicit state
        self.history = []
        self.has_searched = False
        self.loop_count = 0

        # System prompt (ReAct + strict format)
        self.system_prompt = """
You are a ReAct agent.

You MUST follow this loop strictly:

Thought → Action → Observation → Thought → Action ...

Rules:
- NEVER skip Action after Thought
- NEVER output multiple Thoughts in a row
- NEVER give Final answer before at least one Action
- If Observation is empty or bad → REFLECT and try a better query
- If ambiguity exists → refine search

FORMAT STRICTLY:

Thought: <reasoning>
Action: Search["query"]

When enough information is gathered:

Final: <answer>

If uncertain:

Final_Uncertain: <answer>

DO NOT output anything else.
"""

        # Few-shot examples (critical)
        self.few_shot = """
User: What is the population of Japan?

Thought: The user asks for population. I should search for current data.
Action: Search["Japan current population"]

Observation: Japan population is approximately 125 million.

Thought: I have the answer.
Final: Japan's population is approximately 125 million.

---

User: What fraction of Japan's population is Taiwan's population?

Thought: I need both populations. I will search Japan first.
Action: Search["Japan current population"]

Observation: Japan population is 125 million.

Thought: Now I need Taiwan population.
Action: Search["Taiwan current population"]

Observation: Taiwan population is 23 million.

Thought: I can compute the fraction.
Final: Taiwan's population is about 23/125 ≈ 0.18 of Japan's population.

---

User: Japan population 2025 exact?

Thought: The query may be too specific. I will search broadly.
Action: Search["Japan population 2025"]

Observation: No results found.

Thought: The search returned nothing. I was too specific. I will broaden it.
Action: Search["Japan current population"]

Observation: Japan population is 125 million.

Thought: I now have a reasonable estimate.
Final: Japan's population is approximately 125 million.
"""

    def run(self, user_query):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.few_shot},
            {"role": "user", "content": user_query},
        ]

        while self.loop_count < self.max_loops:
            self.loop_count += 1
            print(f"\n--- Iteration {self.loop_count} ---")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                stop=["Observation:"],
            )

            output = response.choices[0].message.content.strip()
            print(output)

            # --- PARSE OUTPUT ---
            if "Final:" in output or "Final_Uncertain:" in output:
                return output

            if "Action:" not in output:
                # malformed → force correction
                messages.append(
                    {
                        "role": "user",
                        "content": "You must follow Thought → Action format. Provide an Action.",
                    }
                )
                continue

            # Extract query
            try:
                action_line = [l for l in output.split("\n") if "Action:" in l][0]
                query = action_line.split("Search[")[1].split("]")[0].strip('"')
            except Exception:
                messages.append(
                    {
                        "role": "user",
                        "content": 'Malformed Action. Use: Action: Search["..."]',
                    }
                )
                continue

            # --- TOOL EXECUTION ---
            observation = search(query)

            if not observation:
                observation = "No results found."

            print(f"Observation: {observation}")

            # --- APPEND BACK TO LLM ---
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        print("Limit of reasoning 5 loops complete")
        return "Final_Uncertain: Could not complete reasoning within limits."
