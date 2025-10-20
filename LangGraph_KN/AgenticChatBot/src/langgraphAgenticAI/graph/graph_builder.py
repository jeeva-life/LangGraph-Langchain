from langgraph.graph import StateGraph, START, END
from ..state.state import State
from ..nodes.basic_chatbot_node import BasicChatbotNode

class GraphBuilder:
    def __init__(self, model):
        self.llm = model
        self.graph_builder = StateGraph(State)

    def basic_chatbot_build_graph(self):
        """Build a basic chatbot graph.
        This method initializes the graph builder and adds the nodes and edges for the basic chatbot."""
        try:

            self.basic_chatbot_node = BasicChatbotNode(self.llm)

            self.graph_builder.add_node("chatbot", self.basic_chatbot_node.process)
            self.graph_builder.add_edge(START, "chatbot")
            self.graph_builder.add_edge("chatbot", END)
        except Exception as e:
            raise ValueError(f"Error: Failed to build the basic chatbot graph: {e}")

    def setup_graph(self, usecase: str):
        """Setup the graph based on the use case.
        """
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()

        return self.graph_builder.compile()

    