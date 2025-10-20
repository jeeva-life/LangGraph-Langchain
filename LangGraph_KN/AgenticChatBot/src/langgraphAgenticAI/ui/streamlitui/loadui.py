import streamlit as st
import os

from ..uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        st.set_page_config(page_title=self.config.get_page_title(), page_icon=":robot_face:", layout="wide")
        st.header(self.config.get_page_title())
        

        with st.sidebar:
            # Get options from config
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            # LLM Selection
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options, index=0, help="Choose the language model for the chatbot")

            if self.user_controls["selected_llm"] == "Groq":
                # Model Selection
                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_Groq_model"] = st.selectbox("Select Model", model_options, index=0, help="Choose the model for the chatbot")
                self.user_controls["GROQ_API_KEY"] =st.session_state["GROQ_API_KEY"] = st.text_input("API KEY", type="password", help="Enter your Groq API key")
                # Validate the API Key
                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning("Please enter your GROQ API key to proceed. Don't have? refer: https://groq.com/api-key")
            
            ## usecase selection
            self.user_controls["selected_usecase"] = st.selectbox("Select Usecase", usecase_options, index=0, help="Choose the usecase for the chatbot")

        return self.user_controls