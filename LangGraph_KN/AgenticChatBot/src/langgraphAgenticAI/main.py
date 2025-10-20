import streamlit as st
from .ui.streamlitui.loadui import LoadStreamlitUI
from .LLMs.groqllm import GroqLLM
from .graph.graph_builder import GraphBuilder
from .ui.streamlitui.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    """Load and runs the langgraph agentic AI app with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while
    implementingexception handling for robustness.
    """

    ## Load UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from UI")
        return

    user_message = st.chat_input("Enter your message here...")

    if user_message:
        try:
            # configure LLM
            obj_llm_config = GroqLLM(usercontrols_input = user_input)
            model = obj_llm_config.get_llm_model()

            if not model:
                st.error("Error: LLM model could not be initialized.")
                return

            # Initialize and set up the graph based on use case
            usecase = user_input.get("selected_usecase")
            if not usecase:
                st.error("Error: No use case selected.")
                return
            
            # Graph Builder
            obj_graph_builder = GraphBuilder(model)
            try:
                graph = obj_graph_builder.setup_graph(usecase)
                obj_display_result = DisplayResultStreamlit(usecase, graph, user_message)
                obj_display_result.display_result_on_ui()
            except Exception as e:
                st.error(f"Error: Failed to build the graph: {e}")
                return
        except Exception as e:
            st.error(f"Error: Failed to run the app: {e}")
            return
    return