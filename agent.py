import os
import openai
from dotenv import load_dotenv
from tools import search

load_dotenv()


class Agent:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = "gemini-2.5-flash-lite"
        self.max_iterations = 5

    def run(self, query):
        system_prompt = """
You are a ReAct agent that reasons step by step, uses tools to gather information, reflects on results, and plans next actions.

Always follow this format in your responses:

Thought: [Your reasoning here]

Action: search(query="your search query here")

After receiving an observation, reflect and continue with:

Thought: [Reflection and new reasoning]

Action: search(query="next query") or Final Answer: [answer]

Available tool:
- search(query): Searches the web for current information using Tavily API. Use this to find facts, statistics, etc.

Few-shot examples:

Example 1: Simple question
User: What is the capital of France?
Thought: The user asks for the capital of France. I know it's Paris, but to demonstrate tool use, I should search.
Action: search(query="capital of France")

(Observation would be provided, then continue)

Thought: The search confirms Paris is the capital.
Final Answer: Paris

Example 2: Complex question requiring decomposition
User: What fraction of Japan's population is Taiwan's population as of 2025?
Thought: To calculate the fraction, I need the populations of Japan and Taiwan in 2025. I should search for each separately.
Action: search(query="Japan population 2025")

(Observation: Japan population 2025 is approximately 124.9 million)

Thought: Got Japan's population. Now need Taiwan's.
Action: search(query="Taiwan population 2025")

(Observation: Taiwan population 2025 is approximately 23.6 million)

Thought: Now I can calculate: 23.6 / 124.9 ≈ 0.189
Final Answer: Approximately 0.189

Example 3: Handling failed search
Thought: I need information on X.
Action: search(query="very specific query")

(Observation: No information found or empty results)

Thought: The search returned nothing. I might have been too specific. I will try searching for 'Japan current population' instead.
Action: search(query="general query")

Remember: Always decompose complex tasks, reflect on failures, and use tools appropriately.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        has_used_search = False

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=1000
            )
            content = response.choices[0].message.content
            print(content)
            messages.append({"role": "assistant", "content": content})

            if "Action:" in content:
                # Parse the action
                lines = content.split("\n")
                action_line = next(
                    (line for line in lines if line.strip().startswith("Action:")), None
                )
                if action_line:
                    action_str = action_line.split("Action:", 1)[1].strip()
                    if action_str.startswith("search(query="):
                        has_used_search = True
                        # Extract query
                        start = action_str.find('"')
                        end = action_str.find('"', start + 1)
                        if start != -1 and end != -1:
                            query_search = action_str[start + 1 : end]
                            observation = search(query_search)
                            print(f"Observation: {observation}")
                            messages.append(
                                {
                                    "role": "user",
                                    "content": f"Observation: {observation}",
                                }
                            )
                        else:
                            observation = "Observation: Invalid query format"
                            print(observation)
                            messages.append(
                                {
                                    "role": "user",
                                    "content": observation,
                                }
                            )
                    elif action_str.startswith("Final Answer:"):
                        final_answer = action_str.split("Final Answer:", 1)[1].strip()
                        if not has_used_search:
                            print("Enforcing tool use before final answer.")
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "Observation: Final Answer issued before any search. Please perform a search action and reflect first."
                                }
                            )
                            continue
                        print(f"Final Answer: {final_answer}")
                        return final_answer
                    else:
                        observation = "Observation: Unknown action"
                        print(observation)
                        messages.append({"role": "user", "content": observation})
                else:
                    observation = "Observation: No action found"
                    print(observation)
                    messages.append({"role": "user", "content": observation})
            elif "Final Answer:" in content:
                final_answer = content.split("Final Answer:", 1)[1].strip()
                if not has_used_search:
                    print("Enforcing tool use before final answer.")
                    messages.append(
                        {
                            "role": "user",
                            "content": "Observation: Final Answer issued without prior search action. Please produce a search and reflection first."
                        }
                    )
                    continue
                print(f"Final Answer: {final_answer}")
                return final_answer
            else:
                # If no action or final answer, prompt to continue
                prompt = "Observation: No final answer yet. Continue with reflection and Action or Final Answer."
                print(prompt)
                messages.append({"role": "user", "content": prompt})

        print("Limit of reasoning 5 loops complete")
        return "Max iterations reached without final answer"
