"""
Object-Oriented FastAPI server for LangChain LCEL Translation Service
This version uses OOP principles with proper class structure and separation of concerns
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class TranslationRequest(BaseModel):
    """Request model for translation"""
    text: str
    language: str


class TranslationResponse(BaseModel):
    """Response model for translation"""
    result: str
    success: bool = True


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = "healthy"
    message: str = "Service is running"


class LLMService:
    """Service class for managing the Language Model and chain"""
    
    def __init__(self, model_name: str = "gemma2-9b-it", temperature: float = 0.5):
        """
        Initialize the LLM service
        
        Args:
            model_name: Name of the Groq model to use
            temperature: Temperature for model generation
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.chain = None
        self._setup_llm()
        self._setup_chain()
    
    def _setup_llm(self) -> None:
        """Setup the Groq LLM"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        try:
            self.llm = ChatGroq(
                model_name=self.model_name,
                api_key=groq_api_key,
                temperature=self.temperature,
                max_tokens=1000,
                max_retries=3,
                timeout=10,
            )
            logger.info(f"LLM initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _setup_chain(self) -> None:
        """Setup the LangChain LCEL chain"""
        try:
            # Create prompt template
            system_template = "Translate the following text to {language}"
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_template),
                ("user", "{text}")
            ])
            
            # Create parser
            parser = StrOutputParser()
            
            # Create chain
            self.chain = prompt_template | self.llm | parser
            logger.info("LangChain LCEL chain created successfully")
        except Exception as e:
            logger.error(f"Failed to setup chain: {e}")
            raise
    
    def translate(self, text: str, language: str) -> str:
        """
        Translate text to specified language
        
        Args:
            text: Text to translate
            language: Target language
            
        Returns:
            Translated text
        """
        try:
            result = self.chain.invoke({
                "text": text,
                "language": language
            })
            logger.info(f"Translation completed: {text[:50]}... -> {language}")
            return result
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise


class TranslationService:
    """Main service class for handling translation operations"""
    
    def __init__(self):
        """Initialize the translation service"""
        self.llm_service = LLMService()
        logger.info("Translation service initialized")
    
    def process_translation(self, request: TranslationRequest) -> TranslationResponse:
        """
        Process a translation request
        
        Args:
            request: Translation request
            
        Returns:
            Translation response
        """
        try:
            result = self.llm_service.translate(request.text, request.language)
            return TranslationResponse(result=result)
        except Exception as e:
            logger.error(f"Translation processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    
    def health_check(self) -> HealthResponse:
        """
        Perform health check
        
        Returns:
            Health status
        """
        try:
            # Test if LLM service is working
            test_result = self.llm_service.translate("test", "Spanish")
            return HealthResponse()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(status="unhealthy", message=f"Service error: {str(e)}")


class FastAPIServer:
    """FastAPI server class for the translation service"""
    
    def __init__(self, title: str = "LangChain LCEL Translation Service", 
                 description: str = "A translation service using LangChain LCEL and Groq",
                 version: str = "1.0.0"):
        """
        Initialize the FastAPI server
        
        Args:
            title: API title
            description: API description
            version: API version
        """
        self.title = title
        self.description = description
        self.version = version
        self.app = None
        self.translation_service = None
        self._setup_app()
        self._setup_middleware()
        self._setup_routes()
        logger.info("FastAPI server initialized")
    
    def _setup_app(self) -> None:
        """Setup the FastAPI application"""
        self.app = FastAPI(
            title=self.title,
            description=self.description,
            version=self.version
        )
    
    def _setup_middleware(self) -> None:
        """Setup middleware for the FastAPI app"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self) -> None:
        """Setup API routes"""
        self.translation_service = TranslationService()
        
        @self.app.get("/", response_model=HealthResponse)
        async def root():
            """Root endpoint"""
            return self.translation_service.health_check()
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return self.translation_service.health_check()
        
        @self.app.post("/translate", response_model=TranslationResponse)
        async def translate_text(request: TranslationRequest):
            """
            Translate text to the specified language
            
            Args:
                request: TranslationRequest containing text and target language
                
            Returns:
                TranslationResponse with translated text
            """
            return self.translation_service.process_translation(request)
        
        @self.app.post("/chain/invoke", response_model=TranslationResponse)
        async def chain_invoke(request: TranslationRequest):
            """
            Alternative endpoint that matches LangServe's invoke pattern
            
            Args:
                request: TranslationRequest containing text and target language
                
            Returns:
                TranslationResponse with translated text
            """
            return self.translation_service.process_translation(request)
        
        @self.app.get("/docs")
        async def get_docs():
            """Redirect to FastAPI docs"""
            return {"message": "Visit /docs for API documentation"}
    
    def run(self, host: str = "127.0.0.1", port: int = 8000, log_level: str = "info") -> None:
        """
        Run the FastAPI server
        
        Args:
            host: Host to run on
            port: Port to run on
            log_level: Log level for uvicorn
        """
        logger.info(f"Starting server on {host}:{port}")
        print(f"Starting {self.title}...")
        print(f"API Documentation: http://{host}:{port}/docs")
        print(f"Health Check: http://{host}:{port}/health")
        print(f"Translation Endpoint: http://{host}:{port}/translate")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=log_level
        )


class ServerManager:
    """Manager class for the entire server application"""
    
    def __init__(self):
        """Initialize the server manager"""
        self.server = None
        logger.info("Server manager initialized")
    
    def create_server(self) -> FastAPIServer:
        """
        Create a new FastAPI server instance
        
        Returns:
            FastAPIServer instance
        """
        self.server = FastAPIServer()
        return self.server
    
    def start_server(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """
        Start the server
        
        Args:
            host: Host to run on
            port: Port to run on
        """
        if not self.server:
            self.create_server()
        
        try:
            self.server.run(host=host, port=port)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise


def main():
    """Main function to run the server"""
    try:
        # Create server manager
        manager = ServerManager()
        
        # Create and start server
        manager.create_server()
        manager.start_server()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
