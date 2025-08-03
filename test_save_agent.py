# test_save_agent.py
# A simple test file to save and load an agent using trace repo's built-in methods
# An agent should be decorated by trace.model

from agents.tool_calling_agent import SimpleAgent

def main():
    # Create agent
    tools = [{"name": "test_tool", "description": "A test tool"}]
    agent = SimpleAgent(tools)
    # Try to save agent using trace repo's built-in save method
    print("\n--- Attempting to save agent ---")
    try:
        agent.save("agent.pkl")
        print("✅ Agent saved successfully using agent.save()")
        save_successful = True
    except Exception as e:
        print(f"❌ Agent save failed: {e}")
        print(f"Error type: {type(e).__name__}")
if __name__ == "__main__":
    main()
