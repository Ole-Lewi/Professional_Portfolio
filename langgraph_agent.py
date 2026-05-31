from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator

import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

from langchain_core.tools import tool

llm = ChatOllama(model="llama3.2:3b")

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word"""
    return len(word)


tools = [get_word_length]
llm_with_tools = llm.bind_tools(tools)

#Define the nodes.
def llm_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]

    #Find and run the right tool
    tool_map = {t.name: t for t in tools}
    result = tool_map[tool_call["name"]].invoke(tool_call["args"])

    return {"messages": [ToolMessage(
        content = str(result),
        tool_call_id=tool_call["id"]
    )]}

#The Decision Function
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "use_tool"
    return END

#Build And Run the Graph
graph = StateGraph(AgentState)

#Add nodes
graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)

#Add edges
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue, {"use_tool": "tool", END: END})
graph.add_edge("tool", "llm") #after tool go back to llm

app = graph.compile()

#Run it
result = app.invoke({
    "messages": [HumanMessage(content="How many letters in the word 'elephant'?")]
})
print(result["messages"][-1].content)