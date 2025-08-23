import asyncio
import litellm

async def call_llm(i):
    try:
        await litellm.acompletion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": f"Hello #{i}"}],
            max_tokens=50
        )
        print(f"{i}: ✅")
    except Exception as e:
        print(e)
        if "429" in str(e) or "rate limit" in str(e).lower():
            print(f"{i}: 🚫 rate limit")
        else:
            print(f"{i}: ❌ {type(e).__name__}")

async def main():
    # Fire off 100 requests in parallel directly
    for i in range(2000):
        asyncio.create_task(call_llm(i))
    
    # Keep running for a bit to see what happens
    await asyncio.sleep(30)

asyncio.run(main()) 