from configparser import ConfigParser # used to read .ini file
import os


class Config:
    def __init__(self, config_file=None):
        if config_file is None:
            # Get the directory of this file and construct the path to the config file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(current_dir, "uiconfigfile.ini")
        
        self.config = ConfigParser()
        self.config.read(config_file)

    def get_llm_options(self):
        llm_options = self.config["DEFAULT"].get("LLM_OPTIONS")
        if llm_options:
            return llm_options.split(",")
        return ["Groq"]  # Default fallback

    def get_usecase_options(self):
        usecase_options = self.config["DEFAULT"].get("USECASE_OPTIONS")
        if usecase_options:
            return usecase_options.split(",")
        return ["Basic Chatbot"]  # Default fallback

    def get_groq_model_options(self):
        groq_options = self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS")
        if groq_options:
            return groq_options.split(",")
        return ["mistral-7b-instruct", "gemma2-9b-it", "deepseek-r1-distill-llama-70b"]  # Default fallback

    def get_page_title(self):
        title = self.config["DEFAULT"].get("PAGE_TITLE")
        return title if title else "LangGraph: Build Stateful Agentic AI Graph"