from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from tau_bench.envs.user import UserStrategy
# os.environ["TRACE_LITELLM_MODEL"] = "gemini/gemini-2.0-flash" # optional

import opto 
from opto.trace import bundle, node , model
from opto.optimizers import OptoPrime 
from opto.trace.nodes import GRAPH 

# Copyright Sierra

import json
from litellm import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from tau_bench.model_utils.model.utils import trim_conversation_messages
import litellm 
litellm.drop_params = True

import sys
import os
from datetime import datetime

# Redirect all output to file
def setup_output_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    log_file = f"output/tau_bench_output_{timestamp}.txt"
    
    # Store original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create a custom stdout that writes to both console and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                try:
                    f.write(obj)
                    f.flush()
                except ValueError:
                    # Handle closed file gracefully
                    pass
        def flush(self):
            for f in self.files:
                try:
                    f.flush()
                except ValueError:
                    # Handle closed file gracefully
                    pass
    
    # Open log file
    log_file_handle = open(log_file, 'w')
    
    # Redirect stdout and stderr to both console and file
    sys.stdout = Tee(original_stdout, log_file_handle)
    sys.stderr = Tee(original_stderr, log_file_handle)
    
    print(f"Output logging started. Saving to: {log_file}")
    return log_file_handle, original_stdout, original_stderr

def create_run_config():
    """Create a RunConfig object with default parameters from run.py"""
    return RunConfig(
        model_provider="openai",
        user_model_provider="gemini",
        model="gpt-4.1-nano",
        user_model="gemini-2.0-flash",
        num_trials=1,
        env="retail",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        task_ids=[0],  # Select task with ID 1
        log_dir="results",
        max_concurrency=1,
        seed=10,
        shuffle=0,
        user_strategy="llm",
        few_shot_displays_path=None
    )
@model
class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = node(tools_info, trainable=True)
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    @bundle()
    def solve(self,tools_info, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
        """Agent solves the task with the given tools_info.
        Args:
            tools_info: The tools_info to use for the task. This is a node that contains information about the tools that the agent can use.
            env: The environment to use for the task.
            task_index: The index of the task to solve.
            max_num_steps: The maximum number of steps to take. This is the maximum number of steps the agent can take to solve the task.

        Returns:
            reward: The reward for the task. 1 if the task is solved correctly, 0 otherwise.
            messages: The messages for the task. If the task is solved correctly, the message will be Correct. Otherwise, the message will be the conversation history.
            info: The info for the task. If the task is solved correctly, the info will be Correct. Otherwise, the info will be the info of the task.
        """
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        for _ in range(max_num_steps):
            # Trim messages to prevent context window errors
            # trimmed_messages = trim_conversation_messages(messages, model=self.model)
            
            # Prepare completion arguments
            completion_kwargs = {
                "messages": messages,
                "model": self.model,
                "custom_llm_provider": self.provider,
                "tools": tools_info,
                "temperature": self.temperature,
            }
            
            # Add api_base only for local/hosted providers
            # if self.provider in ["hosted_vllm", "vllm"]:
            #     completion_kwargs["api_base"] = "http://127.0.0.1:8000/v1"
            
            res = completion(**completion_kwargs)
            next_message = res.choices[0].message.model_dump()
            cost = res._hidden_params.get("response_cost")
            if cost is not None:
                total_cost += cost
            # Skip cost tracking for vLLM/local models where cost is None
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        result = SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
        if result.reward == 1:
            return result.reward,"Correct","Correct"
        else:
            return result.reward,result.messages,result.info
        
    def forward(self, env, task_index):
        """Forward pass of the agent."""
        return self.solve(self.tools_info, env, task_index)
    
        


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})





def main():
    # Setup output logging
    log_file_handle, original_stdout, original_stderr = setup_output_logging()
    
    try:
        # Create configuration
        config = create_run_config()
        
        # Initialize environment
        print(f"Initializing retail environment with user strategy: {config.user_strategy}")
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
            task_index=config.task_ids[0]  # Use the first (and only) task ID
        )
        
        # Initialize agent
        print(f"Initializing {config.agent_strategy} agent with model: {config.model}")

        initial_tools_info = node(env.tools_info, trainable=True)
        # print(f"Tools info: {tools_info.data}")
        

        agent = ToolCallingAgent(
                tools_info=initial_tools_info,
                wiki=env.wiki,
                model=config.model,
                provider=config.model_provider,
                temperature=config.temperature
            )
        
        optimizer = OptoPrime(agent.parameters())
        optimizer.objective = """Optimize the tool descriptions in #Variables to improve agent performance based on #Feedback.

TASK: You are optimizing tool information for a retail customer service agent. The only thing you can change is the tool descriptions. You cannot do anything else to help the agent solve the task.

#Variables contains: Current tool schemas with function names, descriptions, and parameters
#Feedback contains: Either "Correct" (success) or conversation history (failure analysis needed)

INSTRUCTIONS:
1. If feedback is "Correct": Make minor refinements to successful tool descriptions
2. If feedback contains conversation history: Analyze the failure patterns to identify:
   - Which tools were used incorrectly or missed
   - Parameter confusion or formatting errors  
   - Workflow sequence problems

OPTIMIZATION RULES:
- ONLY modify the 'description' fields of tools
- NEVER change function names or parameter schemas
- MUST include ALL original tools in your output
- Focus on clarity, proper usage guidance, and error prevention

Make targeted improvements to tool descriptions that will help the agent make better decisions and avoid the observed failure patterns. 

OUTPUT FORMAT:
Your response must contain ONLY these two sections:
1. "reasoning": Explain your analysis of the feedback and what needs to be improved
2. "suggestion": Provide the complete optimized tool information with improved descriptions

Do not include any other text, explanations, or keywords like TERMINATE."""
        output = agent(env=env,task_index=config.task_ids[0]  )
        reward, messages, info = output
        
        print("\nTask Results:")
        print(f"✅ Success" if reward.data == 1 else "❌ Failed")
        # print(f"Info: {result.data.info}")
        if reward.data == 1:
            feedback = "Correct"
        else:
            # Include complete message information including tool calls
            conversation_parts = []
            for msg in messages.data:
                msg_str = f"{msg['role']}: {msg.get('content', '')}"
                
                # Add tool calls if present
                if 'tool_calls' in msg and msg['tool_calls']:
                    tool_calls_str = []
                    for tool_call in msg['tool_calls']:
                        if 'function' in tool_call:
                            func_name = tool_call['function'].get('name', '')
                            func_args = tool_call['function'].get('arguments', '')
                            tool_calls_str.append(f"Tool: {func_name}({func_args})")
                    if tool_calls_str:
                        msg_str += f" [Tool Calls: {'; '.join(tool_calls_str)}]"
                
                # Add tool call ID and name for tool messages
                if msg['role'] == 'tool':
                    tool_name = msg.get('name', '')
                    tool_call_id = msg.get('tool_call_id', '')
                    msg_str = f"tool ({tool_name}, ID: {tool_call_id}): {msg.get('content', '')}"
                
                conversation_parts.append(msg_str)
            
            feedback = "The agent failed to solve the task. Here is the conversation history: " + "\n".join(conversation_parts)
            # print(feedback)

        
        # feedback = result.data.messages
        optimizer.zero_feedback()
        optimizer.backward(output, feedback)
        optimizer.step(verbose=True)

    finally:
        # Restore original stdout and stderr before closing file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Close log file
        log_file_handle.close()
        print("Output logging completed.")


if __name__ == "__main__":
    main() 