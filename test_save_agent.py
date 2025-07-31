# test_save_agent.py
# A simple test file to save and load an agent using trace repo's built-in methods
# An agent should be decorated by trace.model

from typing import List, Dict, Any
from opto import trace
@trace.model
class SimpleAgent():
    """A simple test agent"""
    def __init__(self, tools_info: List[Dict[str, Any]]):
        self.tools_info = trace.node(tools_info, trainable=True)
        self.instructions = trace.node("Default instructions", trainable=True)
    @trace.bundle()
    def solve(self, tools_info, instructions, task):
        return f"Solved: {task} with {len(tools_info)} tools and instructions: {instructions}"
    def forward(self, task):
        return self.solve(self.tools_info, self.instructions, task)
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
