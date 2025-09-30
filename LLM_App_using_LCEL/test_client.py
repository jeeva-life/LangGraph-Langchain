"""
Test client for the LangChain LCEL Translation Service
Demonstrates how to interact with the FastAPI endpoints
"""

import requests
import json
from typing import Dict, Any

class TranslationClient:
    """Client for interacting with the translation service"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def translate(self, text: str, language: str) -> Dict[str, Any]:
        """Translate text to specified language"""
        try:
            payload = {
                "text": text,
                "language": language
            }
            response = requests.post(
                f"{self.base_url}/translate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def chain_invoke(self, text: str, language: str) -> Dict[str, Any]:
        """Alternative translation endpoint"""
        try:
            payload = {
                "text": text,
                "language": language
            }
            response = requests.post(
                f"{self.base_url}/chain/invoke",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def test_translation_service():
    """Test the translation service with various examples"""
    client = TranslationClient()
    
    print("Testing LangChain LCEL Translation Service")
    print("=" * 50)
    
    # Test health check
    print("\n1. Testing health check...")
    health = client.health_check()
    print(f"Health status: {health}")
    
    # Test translations
    test_cases = [
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
    
    print("\n2. Testing translations...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Original: {test_case['text']}")
        print(f"Target Language: {test_case['language']}")
        
        # Test main translate endpoint
        result = client.translate(test_case['text'], test_case['language'])
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Translation: {result['result']}")
        
        # Test chain invoke endpoint
        chain_result = client.chain_invoke(test_case['text'], test_case['language'])
        if 'error' in chain_result:
            print(f"Chain Error: {chain_result['error']}")
        else:
            print(f"Chain Translation: {chain_result['result']}")
        
        print("-" * 30)

def interactive_mode():
    """Interactive mode for testing translations"""
    client = TranslationClient()
    
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
            
            result = client.translate(text, language)
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Translation: {result['result']}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        test_translation_service()
        
        # Ask if user wants interactive mode
        try:
            interactive = input("\nWould you like to start interactive mode? (y/n): ").strip().lower()
            if interactive in ['y', 'yes']:
                interactive_mode()
        except KeyboardInterrupt:
            print("\nGoodbye!")
