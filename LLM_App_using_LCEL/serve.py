from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
from langserve import add_routes # used to create api's in fastapi
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any

groq_api_key = os.getenv("GROQ_API_KEY")

model_llm = ChatGroq(
    model_name="gemma2-9b-it",
    api_key=groq_api_key,
    temperature=0.5,
    max_tokens=1000,
    #top_p=0.9,
    #top_k=40,
    max_retries=3,
    timeout=10,
    #max_completion_tokens=1000
)

system_template = "Translate the following text to {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",system_template),
        ("user", "{text}")
    ]
)

parser = StrOutputParser()

## create chain

chain = prompt_template | model_llm | parser

# Define input/output models for better API documentation
class TranslationRequest(BaseModel):
    text: str
    language: str

class TranslationResponse(BaseModel):
    result: str

##App definition
app = FastAPI(title="LangGraph LCEL App",
              description="This is a simple LangGraph LCEL App server",
              version="1.0.0")

## add routes with explicit input/output types
add_routes(
    app,
    chain,
    path="/chain",
    input_type=TranslationRequest,
    output_type=TranslationResponse
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)