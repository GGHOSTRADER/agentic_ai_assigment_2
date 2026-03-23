
# Goal : Create Agent implementation using ReACT (reasoning and acting). 

## Tools  and instruccions to Build the agent with
  - Using tavily python library and key for Research API 
  - Use Python for script
  - Build using Clean architecture,separated concerns. Build the following files
    - agent.py: Your Agent class. 
    - tools.py: Search tool wrapper. **tavily**
    - main.py: Execution script. 
    - .env: Config template. **Dont build, is already present** , 
    - requirements.txt: Dependencies.
  - Use openai SDK for calling LLM model but will use "gemini-2.5-flash-lite" as LLM model. 
    - Use openai tool schema for functions
  - For LLM key and tavily key variables, check .env
  - Create .gitignore that does not uploads .env and the other useless python files that are not the script
  - use dotenv for the keys
  - Messages must contain the initial prompt where roll and React is induced


## Invariants
  - Architecture must be ReAct and only 1 agent is allowed
  - React must include Throught, Action, Observation , reflection and Action
  - Implement Stop sequence that stops the LLM from keep generating after the word "Action" been generated
  - Can Max iterate of the loop is 5 times
  - Use Few few shot learning prompting to teach the agent how to reason and behave. of a user asking a question and the agent going through the Thought/Action/Observation steps.
  - Also include few shot examples for handling ambiguity.
  - If result of query has multiple different answer which makes it ambious, must continue the query.

## Success Criteria
- Agent decomposed task in "What fraction of Japan's population is Taiwan's population as of 2025?" question
- agent asked for "Action" after reading the question "What fraction of Japan's population is Taiwan's population as of 2025?" 
- Agent prints in console "Limit of reasoning 5 loops complete" after reasoning for 5 times and stop the loop.
- Agent must reason after failed search query due to No information or wrong information. Reflection (New Thought): "The search returned nothing. I might have been too 
specific. I will try searching for 'Japan current population' instead." 

