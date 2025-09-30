"""
Object-Oriented Test Client for the LangChain LCEL Translation Service
Demonstrates OOP principles with proper class structure and error handling
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Enum for request status"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ClientConfig:
    """Configuration class for the client"""
    base_url: str = "http://127.0.0.1:8000"
    timeout: int = 30
    max_retries: int = 3
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Type": "application/json"}


@dataclass
class TranslationRequest:
    """Data class for translation request"""
    text: str
    language: str


@dataclass
class TranslationResponse:
    """Data class for translation response"""
    result: str
    success: bool
    status: RequestStatus
    error_message: Optional[str] = None


class HTTPClient:
    """Base HTTP client class for making requests"""
    
    def __init__(self, config: ClientConfig):
        """
        Initialize the HTTP client
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.headers)
        logger.info("HTTP client initialized")
    
    def get(self, endpoint: str) -> Dict[str, Any]:
        """
        Make a GET request
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Response data
        """
        try:
            url = f"{self.config.base_url}{endpoint}"
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GET request failed: {e}")
            return {"error": str(e), "status": RequestStatus.ERROR}
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data
        """
        try:
            url = f"{self.config.base_url}{endpoint}"
            response = self.session.post(
                url, 
                json=data, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request failed: {e}")
            return {"error": str(e), "status": RequestStatus.ERROR}


class TranslationClient:
    """Client for interacting with the translation service"""
    
    def __init__(self, config: Optional[ClientConfig] = None):
        """
        Initialize the translation client
        
        Args:
            config: Client configuration
        """
        self.config = config or ClientConfig()
        self.http_client = HTTPClient(self.config)
        logger.info("Translation client initialized")
    
    def health_check(self) -> TranslationResponse:
        """
        Check if the service is healthy
        
        Returns:
            Health check response
        """
        try:
            result = self.http_client.get("/health")
            if "error" in result:
                return TranslationResponse(
                    result="",
                    success=False,
                    status=RequestStatus.ERROR,
                    error_message=result["error"]
                )
            
            return TranslationResponse(
                result=result.get("message", ""),
                success=True,
                status=RequestStatus.SUCCESS
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return TranslationResponse(
                result="",
                success=False,
                status=RequestStatus.ERROR,
                error_message=str(e)
            )
    
    def translate(self, text: str, language: str) -> TranslationResponse:
        """
        Translate text to specified language
        
        Args:
            text: Text to translate
            language: Target language
            
        Returns:
            Translation response
        """
        try:
            request_data = {
                "text": text,
                "language": language
            }
            
            result = self.http_client.post("/translate", request_data)
            
            if "error" in result:
                return TranslationResponse(
                    result="",
                    success=False,
                    status=RequestStatus.ERROR,
                    error_message=result["error"]
                )
            
            return TranslationResponse(
                result=result.get("result", ""),
                success=result.get("success", True),
                status=RequestStatus.SUCCESS
            )
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResponse(
                result="",
                success=False,
                status=RequestStatus.ERROR,
                error_message=str(e)
            )
    
    def chain_invoke(self, text: str, language: str) -> TranslationResponse:
        """
        Alternative translation endpoint
        
        Args:
            text: Text to translate
            language: Target language
            
        Returns:
            Translation response
        """
        try:
            request_data = {
                "text": text,
                "language": language
            }
            
            result = self.http_client.post("/chain/invoke", request_data)
            
            if "error" in result:
                return TranslationResponse(
                    result="",
                    success=False,
                    status=RequestStatus.ERROR,
                    error_message=result["error"]
                )
            
            return TranslationResponse(
                result=result.get("result", ""),
                success=result.get("success", True),
                status=RequestStatus.SUCCESS
            )
        except Exception as e:
            logger.error(f"Chain invoke failed: {e}")
            return TranslationResponse(
                result="",
                success=False,
                status=RequestStatus.ERROR,
                error_message=str(e)
            )


class TestSuite:
    """Test suite class for running various tests"""
    
    def __init__(self, client: TranslationClient):
        """
        Initialize the test suite
        
        Args:
            client: Translation client instance
        """
        self.client = client
        self.test_cases = [
            {
                "text": "Hello, how are you?",
                "language": "Spanish",
                "description": "English to Spanish"
            },
            {
                "text": "Good morning, have a great day!",
                "language": "French",
                "description": "English to French"
            },
            {
                "text": "Thank you very much",
                "language": "German",
                "description": "English to German"
            },
            {
                "text": "What is the weather like today?",
                "language": "Italian",
                "description": "English to Italian"
            }
        ]
        logger.info("Test suite initialized")
    
    def run_health_check(self) -> bool:
        """
        Run health check test
        
        Returns:
            True if health check passes
        """
        print("\n1. Testing health check...")
        health = self.client.health_check()
        print(f"Health status: {health.result if health.success else health.error_message}")
        return health.success
    
    def run_translation_tests(self) -> None:
        """Run translation tests"""
        print("\n2. Testing translations...")
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Original: {test_case['text']}")
            print(f"Target Language: {test_case['language']}")
            
            # Test main translate endpoint
            result = self.client.translate(test_case['text'], test_case['language'])
            if result.success:
                print(f"Translation: {result.result}")
            else:
                print(f"Error: {result.error_message}")
            
            # Test chain invoke endpoint
            chain_result = self.client.chain_invoke(test_case['text'], test_case['language'])
            if chain_result.success:
                print(f"Chain Translation: {chain_result.result}")
            else:
                print(f"Chain Error: {chain_result.error_message}")
            
            print("-" * 30)
    
    def run_all_tests(self) -> None:
        """Run all tests"""
        print("Testing LangChain LCEL Translation Service")
        print("=" * 50)
        
        # Run health check
        health_ok = self.run_health_check()
        
        if health_ok:
            # Run translation tests
            self.run_translation_tests()
        else:
            print("Health check failed. Skipping translation tests.")


class InteractiveMode:
    """Interactive mode class for user interaction"""
    
    def __init__(self, client: TranslationClient):
        """
        Initialize interactive mode
        
        Args:
            client: Translation client instance
        """
        self.client = client
        logger.info("Interactive mode initialized")
    
    def start(self) -> None:
        """Start interactive mode"""
        print("\nInteractive Translation Mode")
        print("=" * 30)
        print("Type 'quit' to exit")
        
        while True:
            try:
                text = input("\nEnter text to translate: ").strip()
                if text.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif not text:
                    continue
                
                language = input("Enter target language: ").strip()
                if not language:
                    language = "Spanish"
                
                result = self.client.translate(text, language)
                if result.success:
                    print(f"Translation: {result.result}")
                else:
                    print(f"Error: {result.error_message}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


class Application:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.client = None
        self.test_suite = None
        self.interactive_mode = None
        logger.info("Application initialized")
    
    def setup(self) -> None:
        """Setup the application components"""
        # Create client with custom config
        config = ClientConfig(
            base_url="http://127.0.0.1:8000",
            timeout=30,
            max_retries=3
        )
        self.client = TranslationClient(config)
        self.test_suite = TestSuite(self.client)
        self.interactive_mode = InteractiveMode(self.client)
        logger.info("Application setup completed")
    
    def run_tests(self) -> None:
        """Run test suite"""
        if not self.test_suite:
            self.setup()
        self.test_suite.run_all_tests()
    
    def run_interactive(self) -> None:
        """Run interactive mode"""
        if not self.interactive_mode:
            self.setup()
        self.interactive_mode.start()
    
    def run(self) -> None:
        """Run the main application"""
        self.setup()
        self.run_tests()
        
        # Ask if user wants interactive mode
        try:
            interactive = input("\nWould you like to start interactive mode? (y/n): ").strip().lower()
            if interactive in ['y', 'yes']:
                self.run_interactive()
        except KeyboardInterrupt:
            print("\nGoodbye!")


def main():
    """Main function"""
    try:
        app = Application()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
