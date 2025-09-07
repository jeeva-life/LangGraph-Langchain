"""
Ollama Embeddings with LangChain

This script demonstrates how to use Ollama embeddings with LangChain
without Jupyter notebook environment issues.
"""

import requests
import json
from langchain_community.embeddings import OllamaEmbeddings

def check_ollama_status():
    """Check if Ollama is running and list available models."""
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✓ Ollama is running!")
            print("Available models:")
            for model in models.get('models', []):
                print(f"  - {model['name']}")
            return models.get('models', [])
        else:
            print("✗ Ollama is not responding properly")
            return []
    except requests.exceptions.ConnectionError:
        print("✗ Ollama is not running. Please start Ollama first.")
        print("  You can start it by running: ollama serve")
        return []
    except Exception as e:
        print(f"✗ Error checking Ollama status: {e}")
        return []

def create_ollama_embeddings(model_name=None):
    """Create Ollama embeddings with the specified model or find a suitable one."""
    available_models = check_ollama_status()
    
    if not available_models:
        print("No models available. Please install a model first.")
        print("Try: ollama pull nomic-embed-text")
        return None
    
    # If no model specified, try to find a suitable embedding model
    if not model_name:
        # Look for common embedding models
        embedding_models = ['nomic-embed-text', 'all-minilm', 'mxbai-embed-large']
        for model in available_models:
            model_name = model['name']
            if any(embed_model in model_name.lower() for embed_model in embedding_models):
                print(f"Using available embedding model: {model_name}")
                break
        else:
            # If no embedding model found, use the first available model
            model_name = available_models[0]['name']
            print(f"No specific embedding model found. Using: {model_name}")
    
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        print(f"✓ Successfully created embeddings with model: {model_name}")
        return embeddings
    except Exception as e:
        print(f"✗ Error creating embeddings with {model_name}: {e}")
        return None

def test_document_embeddings(embeddings):
    """Test embeddings with multiple documents."""
    if not embeddings:
        print("No embeddings object available for testing")
        return None
    
    print("\n📄 Testing document embeddings...")
    try:
        sample_texts = [
            "This is a sample sentence for testing embeddings.",
            "Machine learning is fascinating and powerful.",
            "Natural language processing helps computers understand text."
        ]
        
        embeddings_result = embeddings.embed_documents(sample_texts)
        print(f"✓ Successfully created {len(embeddings_result)} embeddings")
        print(f"  Each embedding has {len(embeddings_result[0])} dimensions")
        print(f"  First embedding (first 5 values): {embeddings_result[0][:5]}")
        
        return embeddings_result
        
    except Exception as e:
        print(f"✗ Error creating document embeddings: {e}")
        print("This might be because:")
        print("  1. The model doesn't support embeddings")
        print("  2. Ollama is not running")
        print("  3. The model needs to be pulled first")
        return None

def test_query_embedding(embeddings):
    """Test embedding for a single query."""
    if not embeddings:
        print("No embeddings object available for testing")
        return None
    
    print("\n🔍 Testing query embedding...")
    try:
        query_text = "This is a single query for embedding."
        query_embedding = embeddings.embed_query(query_text)
        print(f"✓ Query embedding created successfully!")
        print(f"  Query embedding dimensions: {len(query_embedding)}")
        print(f"  Query embedding (first 5 values): {query_embedding[:5]}")
        
        return query_embedding
        
    except Exception as e:
        print(f"✗ Error creating query embedding: {e}")
        return None

def display_embeddings_info(embeddings):
    """Display embeddings configuration information."""
    if embeddings:
        print("\n📋 Embeddings Configuration:")
        print(f"  Model: {embeddings.model}")
        print(f"  Base URL: {embeddings.base_url}")
        print(f"  Embed Instruction: {embeddings.embed_instruction}")
        print(f"  Query Instruction: {embeddings.query_instruction}")
    else:
        print("No embeddings object created. Please check Ollama setup.")

def main():
    """Main function to run the Ollama embeddings demo."""
    print("🚀 Starting Ollama Embeddings Demo")
    print("=" * 50)
    
    # Create embeddings
    embeddings = create_ollama_embeddings()
    
    # Display configuration
    display_embeddings_info(embeddings)
    
    if embeddings:
        # Test document embeddings
        doc_embeddings = test_document_embeddings(embeddings)
        
        # Test query embedding
        query_embedding = test_query_embedding(embeddings)
        
        if doc_embeddings and query_embedding:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n⚠️ Demo completed with some issues.")
            print("Please check the error messages above.")
    else:
        print("\n❌ Demo failed - no embeddings object created.")
        print("Please ensure:")
        print("  1. Ollama is installed and running (ollama serve)")
        print("  2. You have pulled an embedding model (ollama pull nomic-embed-text)")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
