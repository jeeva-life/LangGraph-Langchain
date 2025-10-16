from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # reducer

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def make_default_graph(state: State) :
    graph_workflow = StateGraph(State)

    def call_model(state: State) :
        return {"messages": [model.invoke(state['messages'])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    agent = graph_workflow.compile()
    return agent

agent = make_default_graph()