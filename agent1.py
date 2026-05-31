from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate

import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

#Load your llm
llm = ChatOllama(model="llama3.2:1b")

#Tools
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word"""
    return len(word)

tools = [get_word_length]

prompt = PromptTemplate.from_template("""Answer the following questions as best as you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
...(this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

executor.invoke({"input": "How many letters are there in the word 'intelligence'?"})