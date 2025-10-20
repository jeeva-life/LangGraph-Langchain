import os
import streamlit as st
from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        try:
            groq_api_key = self.user_controls_input.get("GROQ_API_KEY")
            selected_groq_model = self.user_controls_input.get("selected_Groq_model")
            if not groq_api_key and not os.environ.get("GROQ_API_KEY"):
                st.error("Error: GROQ_API_KEY not found in environment variables")
            
            llm = ChatGroq(api_key=groq_api_key, model_name=selected_groq_model)

        except Exception as e:
            raise ValueError(f"Error: Failed to initialize Groq LLM: {e}")
        return llm