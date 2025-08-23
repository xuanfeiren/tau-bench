#!/usr/bin/env python3
"""
Gemini Model Discovery and Rate Limit Testing

This script discovers available Gemini models through LiteLLM and tests their rate limits.
It attempts to identify:
- Available text generation models
- Available embedding models  
- Rate limits for each model
- Model capabilities and specifications

Requirements:
- litellm package
- GEMINI_API_KEY environment variable set

Usage:
    python gemini_model_discovery.py
"""

import os
import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import litellm

# Suppress LiteLLM debug info for cleaner output
litellm.suppress_debug_info = True

# Known Gemini models to test (based on common naming patterns)
POTENTIAL_MODELS = {
    "text_generation": [
        "gemini/gemini-2.0-flash-exp",
        "gemini/gemini-2.0-flash", 
        "gemini/gemini-1.5-pro",
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-flash-8b",
        "gemini/gemini-pro",
        "gemini/gemini-pro-vision",
        "gemini/gemini-1.0-pro",
    ],
    "embedding": [
        "gemini/text-embedding-005",
        "gemini/text-embedding-004",
        "gemini/text-embedding-003", 
        "gemini/text-embedding-002",
        "gemini/text-embedding-001",
        "gemini/embedding-001",
        "gemini/textembedding-gecko",
        "gemini/textembedding-gecko-001",
    ]
}

class ModelTester:
    def __init__(self):
        self.available_models = {"text_generation": [], "embedding": []}
        self.rate_limits = {}
        self.model_info = {}
        
    def test_model_availability(self, model: str, model_type: str) -> Dict[str, Any]:
        """Test if a model is available and get basic info."""
        print(f"   Testing {model}...")
        
        try:
            if model_type == "text_generation":
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                
                return {
                    "available": True,
                    "response_model": getattr(response, 'model', 'unknown'),
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') else None,
                    "cost": response._hidden_params.get("response_cost") if hasattr(response, '_hidden_params') else None
                }
                
            elif model_type == "embedding":
                response = litellm.embedding(
                    model=model,
                    input="Test embedding"
                )
                
                embedding = response.data[0].embedding
                return {
                    "available": True,
                    "response_model": getattr(response, 'model', 'unknown'),
                    "dimensions": len(embedding),
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') else None
                }
                
        except Exception as e:
            error_msg = str(e).lower()
            if "404" in error_msg or "not found" in error_msg:
                return {"available": False, "error": "Model not found"}
            elif "403" in error_msg or "permission" in error_msg:
                return {"available": False, "error": "Permission denied"}
            elif "429" in error_msg or "rate limit" in error_msg:
                return {"available": True, "error": "Rate limited (model exists but hit limits)"}
            else:
                return {"available": False, "error": f"Unknown error: {str(e)[:100]}"}
    
    async def test_rate_limits(self, model: str, model_type: str, max_requests: int = 20) -> Dict[str, Any]:
        """Test rate limits by making multiple concurrent requests."""
        print(f"   🚀 Testing rate limits for {model} (max {max_requests} requests)...")
        
        start_time = time.time()
        successful_requests = 0
        rate_limited_requests = 0
        failed_requests = 0
        response_times = []
        
        async def make_request(request_id: int):
            nonlocal successful_requests, rate_limited_requests, failed_requests
            
            request_start = time.time()
            try:
                if model_type == "text_generation":
                    await litellm.acompletion(
                        model=model,
                        messages=[{"role": "user", "content": f"Request #{request_id}"}],
                        max_tokens=5
                    )
                elif model_type == "embedding":
                    await litellm.aembedding(
                        model=model,
                        input=f"Test embedding request {request_id}"
                    )
                
                request_time = time.time() - request_start
                response_times.append(request_time)
                successful_requests += 1
                print(f"      ✅ Request {request_id}: {request_time:.2f}s")
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                    rate_limited_requests += 1
                    print(f"      🚫 Request {request_id}: Rate limited")
                else:
                    failed_requests += 1
                    print(f"      ❌ Request {request_id}: {type(e).__name__}")
        
        # Create tasks for concurrent requests
        tasks = [make_request(i) for i in range(max_requests)]
        
        # Execute with some delay between batches to avoid overwhelming
        batch_size = 5
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            await asyncio.gather(*batch, return_exceptions=True)
            if i + batch_size < len(tasks):
                await asyncio.sleep(1)  # Brief pause between batches
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        requests_per_second = successful_requests / total_time if total_time > 0 else 0
        
        return {
            "total_requests": max_requests,
            "successful": successful_requests,
            "rate_limited": rate_limited_requests,
            "failed": failed_requests,
            "total_time": total_time,
            "avg_response_time": avg_response_time,
            "requests_per_second": requests_per_second,
            "estimated_rate_limit": f"~{successful_requests} requests in {total_time:.1f}s" if rate_limited_requests > 0 else "No rate limit hit"
        }
    
    def discover_models(self):
        """Discover available models by testing each potential model."""
        print("🔍 Discovering available Gemini models...")
        print("=" * 60)
        
        for model_type, models in POTENTIAL_MODELS.items():
            print(f"\n📋 Testing {model_type.replace('_', ' ').title()} Models:")
            
            for model in models:
                result = self.test_model_availability(model, model_type)
                
                if result["available"]:
                    self.available_models[model_type].append(model)
                    self.model_info[model] = result
                    
                    status = "✅ Available"
                    details = []
                    
                    if model_type == "embedding" and "dimensions" in result:
                        details.append(f"{result['dimensions']} dims")
                    
                    if "response_model" in result:
                        details.append(f"API model: {result['response_model']}")
                    
                    if "cost" in result and result["cost"]:
                        details.append(f"Cost: ${result['cost']:.6f}")
                    
                    detail_str = " | ".join(details) if details else ""
                    print(f"      {status} - {detail_str}")
                    
                else:
                    error_msg = result.get("error", "Unknown error")
                    if "not found" in error_msg:
                        print(f"      ❌ Not available - {error_msg}")
                    elif "Rate limited" in error_msg:
                        print(f"      ⚠️  Available but rate limited - {error_msg}")
                        self.available_models[model_type].append(model)
                    else:
                        print(f"      ⚠️  Error - {error_msg}")
    
    async def test_all_rate_limits(self):
        """Test rate limits for all available models."""
        print("\n" + "=" * 60)
        print("🚀 Testing Rate Limits for Available Models...")
        
        all_models = self.available_models["text_generation"] + self.available_models["embedding"]
        
        if not all_models:
            print("   No available models to test!")
            return
        
        for model in all_models:
            model_type = "text_generation" if model in self.available_models["text_generation"] else "embedding"
            
            print(f"\n📊 Rate limit testing: {model}")
            try:
                rate_limit_info = await self.test_rate_limits(model, model_type, max_requests=15)
                self.rate_limits[model] = rate_limit_info
                
                print(f"   Results:")
                print(f"      Successful: {rate_limit_info['successful']}/{rate_limit_info['total_requests']}")
                print(f"      Rate limited: {rate_limit_info['rate_limited']}")
                print(f"      Avg response time: {rate_limit_info['avg_response_time']:.2f}s")
                print(f"      Requests/second: {rate_limit_info['requests_per_second']:.2f}")
                print(f"      Estimated limit: {rate_limit_info['estimated_rate_limit']}")
                
            except Exception as e:
                print(f"   ❌ Rate limit test failed: {e}")
                self.rate_limits[model] = {"error": str(e)}
    
    def generate_report(self):
        """Generate a comprehensive report of findings."""
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE GEMINI MODEL REPORT")
        print("=" * 60)
        
        # Available models summary
        total_available = len(self.available_models["text_generation"]) + len(self.available_models["embedding"])
        print(f"\n🎯 Summary: Found {total_available} available models")
        
        if self.available_models["text_generation"]:
            print(f"\n💬 Text Generation Models ({len(self.available_models['text_generation'])}):")
            for model in self.available_models["text_generation"]:
                info = self.model_info.get(model, {})
                rate_info = self.rate_limits.get(model, {})
                
                print(f"   • {model}")
                if "response_model" in info:
                    print(f"     API Model: {info['response_model']}")
                if "successful" in rate_info:
                    print(f"     Rate Limit: ~{rate_info['successful']} req/{rate_info['total_time']:.1f}s")
                if "cost" in info and info["cost"]:
                    print(f"     Cost: ${info['cost']:.6f} per request")
        
        if self.available_models["embedding"]:
            print(f"\n🔤 Embedding Models ({len(self.available_models['embedding'])}):")
            for model in self.available_models["embedding"]:
                info = self.model_info.get(model, {})
                rate_info = self.rate_limits.get(model, {})
                
                print(f"   • {model}")
                if "dimensions" in info:
                    print(f"     Dimensions: {info['dimensions']}")
                if "response_model" in info:
                    print(f"     API Model: {info['response_model']}")
                if "successful" in rate_info:
                    print(f"     Rate Limit: ~{rate_info['successful']} req/{rate_info['total_time']:.1f}s")
        
        # Rate limit summary
        if self.rate_limits:
            print(f"\n⚡ Rate Limit Summary:")
            for model, rate_info in self.rate_limits.items():
                if "successful" in rate_info and rate_info["rate_limited"] > 0:
                    print(f"   • {model}: Hit limits after {rate_info['successful']} requests")
                elif "successful" in rate_info:
                    print(f"   • {model}: No limits hit ({rate_info['successful']} requests successful)")
        
        # Save detailed report to JSON
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "available_models": self.available_models,
            "model_info": self.model_info,
            "rate_limits": self.rate_limits
        }
        
        with open("/Users/xuanfeiren/Documents/tau-bench/my_test_api/gemini_model_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: gemini_model_report.json")
        
        # Usage recommendations
        print(f"\n💡 Recommendations:")
        
        if self.available_models["text_generation"]:
            best_text_model = self.available_models["text_generation"][0]  # First available
            print(f"   • For text generation: Use '{best_text_model}'")
        
        if self.available_models["embedding"]:
            best_embedding_model = self.available_models["embedding"][0]  # First available  
            print(f"   • For embeddings: Use '{best_embedding_model}'")
            
        print(f"   • Set GEMINI_API_KEY environment variable")
        print(f"   • Monitor rate limits in production")
        print(f"   • Use async operations for better throughput")


async def main():
    """Main function to run model discovery and rate limit testing."""
    print("🚀 Gemini Model Discovery and Rate Limit Analysis")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  Warning: GEMINI_API_KEY environment variable not set.")
        print("   You may encounter authentication errors.")
        print("   Set your API key: export GEMINI_API_KEY='your-key-here'")
        print()
    
    tester = ModelTester()
    
    # Step 1: Discover available models
    tester.discover_models()
    
    # Step 2: Test rate limits for available models
    await tester.test_all_rate_limits()
    
    # Step 3: Generate comprehensive report
    tester.generate_report()
    
    print("\n🎉 Model discovery and rate limit testing completed!")


if __name__ == "__main__":
    asyncio.run(main())
