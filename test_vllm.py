#!/usr/bin/env python3
"""Simple test for vLLM hosted model"""
import os
from litellm import completion

# Configuration - change these to match your setup
BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,0.0.0.0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1,0.0.0.0'

def test_vllm():
    print(f"Testing vLLM model: {MODEL_NAME} at {BASE_URL}")
    
    try:
        response = completion(
            model=MODEL_NAME,
            custom_llm_provider="hosted_vllm",
            api_base=BASE_URL,
            api_key="EMPTY",
            messages=[{"role": "user", "content": "Hello! Say hi back."}],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"✅ Success! Response: {result}")
        print(f"\n💡 Use with tau-bench:")
        print(f"python run.py --model '{MODEL_NAME}' --model-provider hosted_vllm")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_vllm()
