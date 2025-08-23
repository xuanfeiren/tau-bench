import asyncio
import litellm
import json
import re

async def get_quota_info(model):
    """
    Trigger rate limit to extract quota information from error response
    """
    print(f"\n🔍 Checking quota limits for {model}...")
    
    quota_info = {
        "model": model,
        "quota_limit_value": None,
        "quota_metric": None,
        "quota_limit": None,
        "quota_unit": None,
        "quota_location": None,
        "service": None,
        "available": False,
        "error_details": None
    }
    
    try:
        # Fire many requests rapidly to trigger rate limit
        tasks = []
        for i in range(100):  # Start with 100 rapid requests
            task = litellm.aembedding(
                model=model,
                input=f"Quota test {i}"
            )
            tasks.append(task)
        
        # Execute all at once to quickly hit rate limit
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # If we get here without rate limit, the model works but we didn't hit limits
        quota_info["available"] = True
        print(f"   ✅ Model works, but didn't hit rate limit with 100 requests")
        
    except Exception as e:
        error_str = str(e)
        
        # Check if it's a rate limit error
        if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
            quota_info["available"] = True
            quota_info["error_details"] = error_str
            
            # Parse quota information from error message
            try:
                # Extract quota_limit_value
                quota_limit_match = re.search(r'"quota_limit_value":\s*"(\d+)"', error_str)
                if quota_limit_match:
                    quota_info["quota_limit_value"] = int(quota_limit_match.group(1))
                
                # Extract quota_metric
                quota_metric_match = re.search(r'"quota_metric":\s*"([^"]+)"', error_str)
                if quota_metric_match:
                    quota_info["quota_metric"] = quota_metric_match.group(1)
                
                # Extract quota_limit (the limit name)
                quota_limit_name_match = re.search(r'"quota_limit":\s*"([^"]+)"', error_str)
                if quota_limit_name_match:
                    quota_info["quota_limit"] = quota_limit_name_match.group(1)
                
                # Extract quota_unit
                quota_unit_match = re.search(r'"quota_unit":\s*"([^"]+)"', error_str)
                if quota_unit_match:
                    quota_info["quota_unit"] = quota_unit_match.group(1)
                
                # Extract quota_location
                quota_location_match = re.search(r'"quota_location":\s*"([^"]+)"', error_str)
                if quota_location_match:
                    quota_info["quota_location"] = quota_location_match.group(1)
                
                # Extract service
                service_match = re.search(r'"service":\s*"([^"]+)"', error_str)
                if service_match:
                    quota_info["service"] = service_match.group(1)
                
                print(f"   🚫 Hit rate limit - extracted quota info:")
                print(f"      Quota Limit Value: {quota_info['quota_limit_value']}")
                print(f"      Quota Metric: {quota_info['quota_metric']}")
                print(f"      Quota Limit Name: {quota_info['quota_limit']}")
                print(f"      Quota Unit: {quota_info['quota_unit']}")
                print(f"      Location: {quota_info['quota_location']}")
                print(f"      Service: {quota_info['service']}")
                
            except Exception as parse_error:
                print(f"   ⚠️  Could not parse quota details: {parse_error}")
                print(f"   Raw error: {error_str[:200]}...")
        
        elif "404" in error_str or "not found" in error_str.lower():
            quota_info["available"] = False
            quota_info["error_details"] = "Model not found"
            print(f"   ❌ Model not available: {error_str[:100]}...")
        
        else:
            quota_info["available"] = False
            quota_info["error_details"] = error_str[:200]
            print(f"   ❌ Other error: {error_str[:100]}...")
    
    return quota_info

async def check_multiple_models():
    """Check quota limits for multiple embedding models"""
    models_to_test = [
        "gemini/text-embedding-004",
        "gemini/gemini-embedding-001",
        "gemini/text-embedding-001",  # Different format
        "gemini/embedding-001",       # Another format
    ]
    
    all_quota_info = []
    
    for model in models_to_test:
        quota_info = await get_quota_info(model)
        all_quota_info.append(quota_info)
        
        # Brief pause between models
        await asyncio.sleep(2)
    
    return all_quota_info

def print_summary_report(all_quota_info):
    """Print a nice summary of all quota information"""
    print("\n" + "=" * 80)
    print("📋 GEMINI EMBEDDING MODELS - QUOTA LIMITS REPORT")
    print("=" * 80)
    
    available_models = [info for info in all_quota_info if info["available"]]
    unavailable_models = [info for info in all_quota_info if not info["available"]]
    
    if available_models:
        print(f"\n✅ Available Models ({len(available_models)}):")
        print(f"{'Model':<35} {'Quota Limit':<12} {'Unit':<25} {'Location':<15}")
        print("-" * 80)
        
        for info in available_models:
            quota_limit = info["quota_limit_value"] if info["quota_limit_value"] else "Unknown"
            quota_unit = info["quota_unit"] if info["quota_unit"] else "Unknown"
            location = info["quota_location"] if info["quota_location"] else "Unknown"
            
            print(f"{info['model']:<35} {quota_limit:<12} {quota_unit:<25} {location:<15}")
    
    if unavailable_models:
        print(f"\n❌ Unavailable Models ({len(unavailable_models)}):")
        for info in unavailable_models:
            print(f"   • {info['model']}: {info['error_details']}")
    
    # Detailed quota information
    models_with_quota = [info for info in available_models if info["quota_limit_value"]]
    if models_with_quota:
        print(f"\n📊 Detailed Quota Information:")
        for info in models_with_quota:
            print(f"\n🔹 {info['model']}:")
            print(f"   Limit Value: {info['quota_limit_value']} requests")
            print(f"   Metric: {info['quota_metric']}")
            print(f"   Limit Name: {info['quota_limit']}")
            print(f"   Unit: {info['quota_unit']}")
            print(f"   Location: {info['quota_location']}")
            print(f"   Service: {info['service']}")
    
    # Save to JSON
    with open("/Users/xuanfeiren/Documents/tau-bench/my_test_api/gemini_quota_limits.json", "w") as f:
        json.dump(all_quota_info, f, indent=2)
    
    print(f"\n💾 Detailed quota data saved to: gemini_quota_limits.json")
    
    # Summary
    print(f"\n💡 Summary:")
    for info in models_with_quota:
        if info["quota_limit_value"]:
            print(f"   • {info['model']}: {info['quota_limit_value']} requests per minute")

async def main():
    print("🚀 GEMINI EMBEDDING QUOTA LIMIT CHECKER")
    print("=" * 80)
    print("This script will trigger rate limits to extract quota information...")
    
    # Check quota limits for all models
    all_quota_info = await check_multiple_models()
    
    # Print comprehensive report
    print_summary_report(all_quota_info)
    
    print(f"\n🎉 Quota limit checking completed!")

if __name__ == "__main__":
    asyncio.run(main())
