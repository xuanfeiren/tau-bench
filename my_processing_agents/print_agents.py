#!/usr/bin/env python3
"""
Simple script to load and print information about 10 agents from checkpoints.
This script only loads the agents and prints basic information about them.
"""

from agents.tool_calling_agent import ToolCallingAgent_v2 as ToolCallingAgent
from tau_bench.envs import get_env
from tau_bench.types import RunConfig

def main():
    """Load and print information about 10 agents from checkpoints."""
    
    # Configuration setup (minimal, just what's needed to initialize agents)
    provider = "gemini"
    config = RunConfig(
        model_provider=provider,
        user_model_provider=provider,
        model="gemini-2.0-flash",
        user_model="gemini-2.0-flash",
        num_trials=1,
        env="retail",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        task_ids=list(range(10)),
        log_dir="results",
        max_concurrency=1,
        seed=10,
        shuffle=0,
        user_strategy="llm",
        few_shot_displays_path=None
    )
    
    # Initialize environment (needed for agent setup)
    print("Initializing retail environment...")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
        task_index=0
    )
    
    # Load 10 agents from checkpoints folder
    print("\nLoading agents from checkpoints...")
    agents = []
    num_agents = 10
    
    for i in range(num_agents):
        try:
            agent = ToolCallingAgent(
                tools_info=env.tools_info,
                wiki=env.wiki,
                model=config.model,
                provider=config.model_provider,
                temperature=config.temperature
            )
            agent.load(f"checkpoints/myagent_{i}.pkl")
            agent.set_env(env)
            agents.append(agent)
            print(f"✓ Successfully loaded agent_{i}.pkl")
        except FileNotFoundError:
            print(f"✗ Failed to load agent_{i}.pkl - file not found")
        except Exception as e:
            print(f"✗ Failed to load agent_{i}.pkl - error: {str(e)}")
    
    print(f"\nSuccessfully loaded {len(agents)} agents out of {num_agents} attempted")
    
    # Print information about each loaded agent
    print("\n" + "="*60)
    print("AGENT INFORMATION")
    print("="*60)
    
    for i, agent in enumerate(agents):
        print(f"\nAgent {i}:")
        print(f"  Model: {agent.model}")
        print(f"  Provider: {agent.provider}")
        print(f"  Temperature: {agent.temperature}")
        
        # Print additional instructions if available
        if hasattr(agent, 'additional_instructions') and agent.additional_instructions:
            instructions = agent.additional_instructions.data if hasattr(agent.additional_instructions, 'data') else str(agent.additional_instructions)
            print(f"  Additional Instructions: {instructions}")
        else:
            print("  Additional Instructions: None")
        
        # Print tools info summary
        if hasattr(agent, 'tools_info') and agent.tools_info:
            tools_count = len(agent.tools_info) if hasattr(agent.tools_info, '__len__') else "Unknown"
            print(f"  Number of tools: {tools_count}")
        else:
            print("  Tools: None")

if __name__ == "__main__":
    main() 