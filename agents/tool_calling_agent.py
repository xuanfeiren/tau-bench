# agents/tool_calling_agent_opto.py
import json
from litellm import completion
from typing import List, Optional, Dict, Any
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from tau_bench.retry_utils import auto_retry_with_exponential_backoff
from opto import trace

@trace.model
class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        super().__init__()
        self.tools_info = trace.node(tools_info, trainable=True)
        self.wiki = wiki
        self.additional_instructions = trace.node("Here are the additional instructions to help the agent solve the task: ", trainable=True)
        self.model = model
        self.provider = provider
        self.temperature = temperature

    @trace.bundle()
    def solve(self, tools_info, additional_instructions, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
        """Agent solves the task with the given tools_info."""
        total_cost = 0.0
        
        # Wrap env.reset with retry logic
        def reset_env():
            return env.reset(task_index=task_index)
        
        env_reset_res = auto_retry_with_exponential_backoff(
            reset_env,
            operation_name="Environment reset"
        )
        
        if env_reset_res is None:
            # If reset failed after all retries, return failure
            print("Environment reset failed, return None reward")
            return None, [], {}
            
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "system", "content": additional_instructions},
            {"role": "user", "content": obs},
        ]
        
        for step in range(max_num_steps):
            completion_kwargs = {
                "messages": messages,
                "model": self.model,
                "custom_llm_provider": self.provider,
                "tools": tools_info,
                "temperature": self.temperature,
            }
            
            # Define the complete interaction as a single function to retry
            def step_interaction():
                # Step 1: Get completion from API
                res = completion(**completion_kwargs)
                
                # Step 2: Process the response
                next_message = res.choices[0].message.model_dump()
                cost = res._hidden_params.get("response_cost")
                if cost is not None:
                    nonlocal total_cost
                    total_cost += cost
                
                # Step 3: Convert message to action
                action = message_to_action(next_message)
                
                # Step 4: Execute action in environment
                env_response = env.step(action)
                
                return next_message, action, env_response
            
            # Use auto retry function
            step_result = auto_retry_with_exponential_backoff(
                step_interaction, 
                operation_name=f"Step {step}"
            )
            
            if step_result is None:
                print(f"Step {step}: Return None reward due to interaction failure")
                return None, [], {}
            
            if step_result == -1:
                print(f"Step {step}: Return 0 reward due to BadRequest error")
                return 0, [], "BadRequest"
            
            # Extract results
            next_message, action, env_response = step_result
            
            # Process results since step was successful
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend([
                    next_message,
                    {
                        "role": "tool",
                        "tool_call_id": next_message["tool_calls"][0]["id"],
                        "name": next_message["tool_calls"][0]["function"]["name"],
                        "content": env_response.observation,
                    },
                ])
            else:
                messages.extend([
                    next_message,
                    {"role": "user", "content": env_response.observation},
                ])
                
            if env_response.done:
                break
                
        result = SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)
        
        if result.reward == 1:
            return result.reward, "Correct", "Correct"
        else:
            return result.reward, result.messages, result.info
    
    def forward(self, task_input):
        """Forward pass of the agent for trainer compatibility."""
        env = getattr(self, '_env', None)
        if env is None:
            raise ValueError("Environment not set. Call set_env() before forward pass.")
        
        return self.solve(self.tools_info, self.additional_instructions, env, task_input)
    
    def set_env(self, env):
        """Set the environment for this agent."""
        self._env = env

def message_to_action(message: Dict[str, Any]) -> Action:
    """Convert message to action."""
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
    
@trace.model
class SimpleAgent(Agent):
    """A simple test agent"""
    def __init__(self, tools_info: List[Dict[str, Any]]):
        self.tools_info = trace.node(tools_info, trainable=True)
        self.instructions = trace.node("Default instructions", trainable=True)
    @trace.bundle()
    def solve(self, tools_info, instructions, task):
        return f"Solved: {task} with {len(tools_info)} tools and instructions: {instructions}"
    def forward(self, task):
        return self.solve(self.tools_info, self.instructions, task)
    