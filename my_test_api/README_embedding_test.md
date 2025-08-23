# Gemini Embedding Model Test

This directory contains a comprehensive test script for using Google's Gemini text embedding models through LiteLLM.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r test_requirements.txt
   ```

2. **Set up your Gemini API key:**
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key-here"
   ```

3. **Run the test:**
   ```bash
   python test_embedding_model.py
   ```

## What the Test Covers

The test script demonstrates:

- ✅ **Basic embedding**: Single text to vector conversion
- ✅ **Batch processing**: Multiple texts at once
- ✅ **Different text types**: Short, long, technical, special characters, code
- ✅ **Async operations**: Concurrent embedding requests
- ✅ **Error handling**: Edge cases and invalid inputs
- ✅ **Model information**: Detailed statistics about embeddings
- ✅ **Semantic similarity**: Cosine similarity between embeddings

## Key Findings

**Yes, you can use LiteLLM to access Gemini embedding models!**

- Use `"gemini/text-embedding-004"` for the current model
- Both sync (`litellm.embedding()`) and async (`litellm.aembedding()`) work
- Supports batch processing for efficiency
- Returns high-dimensional vectors (768 dimensions)
- Provides detailed embedding statistics and analysis

## Example Usage

```python
import litellm

# Simple embedding
response = litellm.embedding(
    model="gemini/text-embedding-004",
    input="Your text here"
)
embedding = response.data[0].embedding

# Batch embedding
response = litellm.embedding(
    model="gemini/text-embedding-004", 
    input=["Text 1", "Text 2", "Text 3"]
)
embeddings = [data.embedding for data in response.data]
```

## Requirements

- `litellm` package
- Valid `GEMINI_API_KEY` environment variable
- Internet connection for API calls

The test script will warn you if the API key is not set but will still attempt to run (useful for testing error handling).
