from pydantic import BaseModel, Field
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

class State(TypedDict):
    """State definition for the agentic chatbot."""
    messages: Annotated[List, add_messages]
    

