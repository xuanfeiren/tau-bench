import asyncio
import litellm

async def call_embedding(model, i):
    try:
        await litellm.aembedding(
            model=model,
            input=f"Test embedding request #{i}"
        )
        print(f"{model} - {i}: ✅")
    except Exception as e:
        print(e)
        if "429" in str(e) or "rate limit" in str(e).lower():
            print(f"{model} - {i}: 🚫 rate limit")
        else:
            print(f"{model} - {i}: ❌ {type(e).__name__}")

async def test_model(model_name, num_requests=2000):
    print(f"\n🚀 Testing {model_name} with {num_requests} requests...")
    
    # Fire off requests in parallel
    for i in range(num_requests):
        asyncio.create_task(call_embedding(model_name, i))
    
    # Keep running for a bit to see what happens
    await asyncio.sleep(30)

async def main():
    models_to_test = [
        # "gemini/text-embedding-004",
        "gemini/gemini-embedding-001"  # Alternative name format
    ]
    
    for model in models_to_test:
        await test_model(model, 50)
        print(f"\n⏳ Waiting 10 seconds before next model...")
        await asyncio.sleep(10)  # Brief pause between model tests

asyncio.run(main())
