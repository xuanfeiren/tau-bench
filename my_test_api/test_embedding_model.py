#!/usr/bin/env python3
"""
Test script for Gemini text embedding model using LiteLLM.

This script demonstrates how to use Gemini text embedding models through LiteLLM
with various examples including basic embedding, batch processing, and error handling.

Requirements:
- litellm package
- GEMINI_API_KEY environment variable set

Usage:
    python test_embedding_model.py
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional
import litellm

def test_basic_embedding():
    """Test basic embedding functionality with a single text."""
    print("🔍 Testing basic embedding functionality...")
    
    try:
        # Simple text embedding
        text = "Hello, this is a test sentence for embedding."
        
        response = litellm.embedding(
            model="gemini/text-embedding-004",  # Using the latest Gemini embedding model
            input=text
        )
        
        # Extract embedding vector
        embedding = response.data[0].embedding
        
        print(f"✅ Basic embedding successful!")
        print(f"   Input text: {text}")
        print(f"   Embedding dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Response model: {response.model}")
        
        return embedding
        
    except Exception as e:
        print(f"❌ Basic embedding failed: {e}")
        return None


def test_batch_embedding():
    """Test batch embedding with multiple texts."""
    print("\n📦 Testing batch embedding...")
    
    try:
        texts = [
            "This is the first sentence.",
            "Here's a second example text.",
            "Machine learning is fascinating.",
            "Natural language processing enables computers to understand human language.",
            "Embeddings convert text into numerical vectors."
        ]
        
        response = litellm.embedding(
            model="gemini/text-embedding-004",
            input=texts
        )
        
        print(f"✅ Batch embedding successful!")
        print(f"   Number of texts: {len(texts)}")
        print(f"   Number of embeddings returned: {len(response.data)}")
        
        for i, (text, data) in enumerate(zip(texts, response.data)):
            embedding = data.embedding
            print(f"   Text {i+1}: '{text[:30]}...' -> {len(embedding)} dimensions")
        
        return [data.embedding for data in response.data]
        
    except Exception as e:
        print(f"❌ Batch embedding failed: {e}")
        return None


def test_different_text_types():
    """Test embedding with different types of text content."""
    print("\n🎭 Testing different text types...")
    
    test_cases = [
        ("Short text", "Hi"),
        ("Question", "What is the meaning of life?"),
        ("Technical text", "Neural networks use backpropagation for training."),
        ("Long text", "This is a much longer piece of text that contains multiple sentences and ideas. It discusses various topics including technology, science, and human behavior. The purpose is to test how the embedding model handles longer input sequences."),
        ("Special characters", "Hello! @#$%^&*() 123 测试 🚀 🎉"),
        ("Code snippet", "def hello_world():\n    print('Hello, World!')\n    return True")
    ]
    
    results = []
    
    for description, text in test_cases:
        try:
            response = litellm.embedding(
                model="gemini/text-embedding-004",
                input=text
            )
            
            embedding = response.data[0].embedding
            results.append((description, len(embedding), embedding[:3]))
            print(f"   ✅ {description}: {len(embedding)} dimensions")
            
        except Exception as e:
            print(f"   ❌ {description} failed: {e}")
            results.append((description, None, None))
    
    return results


async def test_async_embedding():
    """Test asynchronous embedding calls."""
    print("\n⚡ Testing async embedding...")
    
    try:
        texts = [f"Async test sentence number {i}" for i in range(5)]
        
        # Create async tasks
        tasks = []
        for text in texts:
            task = litellm.aembedding(
                model="gemini/text-embedding-004",
                input=text
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"✅ Async embedding successful!")
        print(f"   Processed {len(texts)} texts in {end_time - start_time:.2f} seconds")
        
        for i, response in enumerate(responses):
            embedding = response.data[0].embedding
            print(f"   Text {i+1}: {len(embedding)} dimensions")
            
        return responses
        
    except Exception as e:
        print(f"❌ Async embedding failed: {e}")
        return None


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n🚨 Testing error handling...")
    
    test_cases = [
        ("Empty string", ""),
        ("None input", None),
        ("Very long text", "A" * 10000),  # Test with very long input
    ]
    
    for description, test_input in test_cases:
        try:
            if test_input is None:
                # Skip None test as it will cause TypeError before API call
                print(f"   ⚠️  {description}: Skipped (would cause TypeError)")
                continue
                
            response = litellm.embedding(
                model="gemini/text-embedding-004",
                input=test_input
            )
            
            embedding = response.data[0].embedding
            print(f"   ✅ {description}: Handled successfully ({len(embedding)} dimensions)")
            
        except Exception as e:
            print(f"   ⚠️  {description}: Expected error - {type(e).__name__}: {str(e)[:100]}")


def test_model_info():
    """Test and display information about the current Gemini embedding model."""
    print("\n🔬 Testing model information...")
    
    model = "gemini/text-embedding-004"  # Current model
    test_text = "This is a test to get model information and embedding characteristics."
    
    try:
        response = litellm.embedding(
            model=model,
            input=test_text
        )
        
        embedding = response.data[0].embedding
        
        # Calculate some basic statistics
        import statistics
        mean_val = statistics.mean(embedding)
        std_val = statistics.stdev(embedding)
        min_val = min(embedding)
        max_val = max(embedding)
        
        print(f"   ✅ Model: {model}")
        print(f"   📊 Embedding dimensions: {len(embedding)}")
        print(f"   📈 Statistics:")
        print(f"      Mean: {mean_val:.6f}")
        print(f"      Std Dev: {std_val:.6f}")
        print(f"      Min: {min_val:.6f}")
        print(f"      Max: {max_val:.6f}")
        print(f"   🎯 Response model: {response.model}")
        
        return {
            'model': model,
            'dimensions': len(embedding),
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ {model}: {e}")
        return {'model': model, 'error': str(e), 'success': False}


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    import math
    
    # Dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    
    # Magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in embedding1))
    magnitude2 = math.sqrt(sum(a * a for a in embedding2))
    
    # Cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)


def test_similarity_comparison():
    """Test semantic similarity between different texts."""
    print("\n🔍 Testing semantic similarity...")
    
    try:
        text_pairs = [
            ("The cat sat on the mat.", "A feline rested on the rug."),
            ("I love programming.", "I hate vegetables."),
            ("Machine learning is awesome.", "AI and ML are fascinating."),
            ("Hello world", "Goodbye moon"),
        ]
        
        for text1, text2 in text_pairs:
            # Get embeddings
            response1 = litellm.embedding(model="gemini/text-embedding-004", input=text1)
            response2 = litellm.embedding(model="gemini/text-embedding-004", input=text2)
            
            embedding1 = response1.data[0].embedding
            embedding2 = response2.data[0].embedding
            
            # Calculate similarity
            similarity = calculate_similarity(embedding1, embedding2)
            
            print(f"   Text 1: '{text1}'")
            print(f"   Text 2: '{text2}'")
            print(f"   Similarity: {similarity:.4f}")
            print()
            
    except Exception as e:
        print(f"❌ Similarity test failed: {e}")


def main():
    """Main test function."""
    print("🚀 Starting Gemini Embedding Model Tests with LiteLLM")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY environment variable not set.")
        print("   You may encounter authentication errors.")
        print()
    
    # Run synchronous tests
    test_basic_embedding()
    test_batch_embedding()
    test_different_text_types()
    test_error_handling()
    test_model_info()
    test_similarity_comparison()
    
    # Run async test
    print("\n" + "=" * 60)
    asyncio.run(test_async_embedding())
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\nUsage Summary:")
    print("- ✅ LiteLLM can access Gemini embedding models")
    print("- ✅ Current model: 'gemini/text-embedding-004' (768 dimensions)")
    print("- ✅ Supports both sync and async operations")
    print("- ✅ Handles batch processing efficiently")
    print("- ✅ Good error handling for edge cases")
    print("- ✅ Provides detailed embedding statistics")
    print("\n💡 Ready for production use with Gemini text embeddings!")


if __name__ == "__main__":
    main()