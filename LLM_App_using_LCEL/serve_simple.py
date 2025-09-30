"""
Simple FastAPI server for LangChain LCEL without LangServe
This version avoids Pydantic compatibility issues by using direct FastAPI endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

load_dotenv()

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

model_llm = ChatGroq(
    model_name="gemma2-9b-it",
    api_key=groq_api_key,
    temperature=0.5,
    max_tokens=1000,
    max_retries=3,
    timeout=10,
)

# Create prompt template
system_template = "Translate the following text to {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

# Create parser
parser = StrOutputParser()

# Create chain
chain = prompt_template | model_llm | parser

# Pydantic models for request/response
class TranslationRequest(BaseModel):
    text: str
    language: str

class TranslationResponse(BaseModel):
    result: str
    success: bool = True

class HealthResponse(BaseModel):
    status: str = "healthy"
    message: str = "Service is running"

# FastAPI app
app = FastAPI(
    title="LangChain LCEL Translation Service",
    description="A simple translation service using LangChain LCEL and Groq",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse()

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text to the specified language
    
    Args:
        request: TranslationRequest containing text and target language
        
    Returns:
        TranslationResponse with translated text
    """
    try:
        # Invoke the chain with the request data
        result = chain.invoke({
            "text": request.text,
            "language": request.language
        })
        
        return TranslationResponse(result=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/chain/invoke", response_model=TranslationResponse)
async def chain_invoke(request: TranslationRequest):
    """
    Alternative endpoint that matches LangServe's invoke pattern
    
    Args:
        request: TranslationRequest containing text and target language
        
    Returns:
        TranslationResponse with translated text
    """
    try:
        # Invoke the chain with the request data
        result = chain.invoke({
            "text": request.text,
            "language": request.language
        })
        
        return TranslationResponse(result=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chain invocation failed: {str(e)}")

@app.get("/docs")
async def get_docs():
    """Redirect to FastAPI docs"""
    return {"message": "Visit /docs for API documentation"}

if __name__ == "__main__":
    print("Starting LangChain LCEL Translation Service...")
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("Health Check: http://127.0.0.1:8000/health")
    print("Translation Endpoint: http://127.0.0.1:8000/translate")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        log_level="info"
    )
