import requests
import os

# Disable proxy for this session
session = requests.Session()
session.proxies = {'http': None, 'https': None}

# Use with LiteLLM
import litellm

response = litellm.completion(
    model="openai/Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    custom_llm_provider="openai",
    # Pass the session without proxy
    http_client=session
)