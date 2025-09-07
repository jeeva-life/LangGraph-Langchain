"""
HuggingFace Embeddings with LangChain

This script demonstrates how to use HuggingFace embeddings with LangChain
without Jupyter notebook environment issues.
"""

import os
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

def setup_environment():
    """Set up environment variables and load configuration."""
    load_dotenv()
    
    # Set up HuggingFace API key (optional for local models)
    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if huggingface_api_key:
        os.environ["HUGGINGFACE_API_KEY"] = huggingface_api_key
        print("✓ HuggingFace API key loaded successfully")
    else:
        print("ℹ No HuggingFace API key found - using local models only")
    
    return True

def create_embeddings_model():
    """Create and initialize the HuggingFace embeddings model."""
    try:
        # Create HuggingFace embeddings instance
        # Using a lightweight model that works well locally
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Use CPU to avoid GPU issues
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print("✓ HuggingFace embeddings model loaded successfully!")
        print(f"  Model: {embeddings.model_name}")
        print(f"  Model dimensions: {embeddings.client.get_sentence_embedding_dimension()}")
        
        return embeddings
        
    except Exception as e:
        print(f"✗ Error loading embeddings model: {e}")
        return None

def test_document_embeddings(embeddings):
    """Test embeddings with multiple documents."""
    sample_texts = [
        "This is a sample sentence for testing embeddings.",
        "Machine learning is fascinating and powerful.",
        "Natural language processing helps computers understand text."
    ]
    
    print("\n📄 Testing document embeddings...")
    try:
        # Generate embeddings
        sample_embeddings = embeddings.embed_documents(sample_texts)
        
        print(f"✓ Successfully created {len(sample_embeddings)} embeddings")
        print(f"  Each embedding has {len(sample_embeddings[0])} dimensions")
        
        # Show similarity between first two texts
        similarity = np.dot(sample_embeddings[0], sample_embeddings[1])
        print(f"  Similarity between first two texts: {similarity:.4f}")
        
        return sample_embeddings
        
    except Exception as e:
        print(f"✗ Error creating document embeddings: {e}")
        return None

def test_query_embedding(embeddings):
    """Test embedding for a single query."""
    single_text = "This is a single document for embedding."
    
    print("\n🔍 Testing query embedding...")
    try:
        single_embedding = embeddings.embed_query(single_text)
        print(f"✓ Single embedding created successfully!")
        print(f"  Embedding dimensions: {len(single_embedding)}")
        print(f"  First 5 values: {single_embedding[:5]}")
        
        return single_embedding
        
    except Exception as e:
        print(f"✗ Error creating query embedding: {e}")
        return None

def calculate_similarity(embeddings, text1, text2):
    """Calculate similarity between two texts."""
    try:
        emb1 = embeddings.embed_query(text1)
        emb2 = embeddings.embed_query(text2)
        similarity = np.dot(emb1, emb2)
        return similarity
    except Exception as e:
        print(f"✗ Error calculating similarity: {e}")
        return None

def main():
    """Main function to run the embeddings demo."""
    print("🚀 Starting HuggingFace Embeddings Demo")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Create embeddings model
    embeddings = create_embeddings_model()
    if not embeddings:
        print("❌ Failed to create embeddings model. Exiting.")
        return
    
    # Test document embeddings
    doc_embeddings = test_document_embeddings(embeddings)
    
    # Test query embedding
    query_embedding = test_query_embedding(embeddings)
    
    # Test similarity calculation
    print("\n🔗 Testing similarity calculation...")
    text1 = "The weather is nice today."
    text2 = "It's a beautiful sunny day."
    text3 = "I love programming in Python."
    
    sim1_2 = calculate_similarity(embeddings, text1, text2)
    sim1_3 = calculate_similarity(embeddings, text1, text3)
    
    if sim1_2 is not None and sim1_3 is not None:
        print(f"✓ Similarity between weather texts: {sim1_2:.4f}")
        print(f"✓ Similarity between weather and programming: {sim1_3:.4f}")
        print(f"  (Higher values indicate more similarity)")
    
    print("\n✅ Demo completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
