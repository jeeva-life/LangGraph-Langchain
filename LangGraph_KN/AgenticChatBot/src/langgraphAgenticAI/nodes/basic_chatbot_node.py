from ..state.state import State

class BasicChatbotNode:
    """A basic chatbot node that uses the LLM to generate a response.
    """
    def __init__(self, model):
        self.llm = model


    def process(self, state: State) -> dict:
        """Process the state and return a new state.
        This method uses the LLM to generate a response and updates the state with the response.
        """
        return {"messages": self.llm.invoke(state["messages"])}