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


class TrainedToolCallingAgent(ToolCallingAgent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        super().__init__(tools_info, wiki, model, provider, temperature)
        self.additional_instructions = trace.node("""Here are the additional instructions to help the agent solve the task: 
                                                  - If the user is inquiring about splitting payments, immediately inform them whether that is possible or not. If not, offer alternative solutions such as modifying or cancelling the order. Check if the user has sufficient credit card limits before offering payment modification options.

- When a user wants to exchange an item but does not have the product ID for the replacement, use 'list_all_product_types' to determine the product type, then use 'get_product_details' with the product ID to find available item options. If that still doesn't work, inform the user that you can cancel the order so they can reorder with the correct items. After finding available item options, ask the user for the item ID of the new item.

- If you encounter any issues that cannot be resolved with the available tools, or if the user explicitly requests it, transfer the user to a human agent with a clear summary of the issue. If all attempts to resolve the issue fail, transfer to a human agent. Only transfer after exhausting all available options. Summarize the steps taken to resolve the user's issue before transferring.

- Before calling 'exchange_delivered_order_items', verify that the order status is 'delivered' and *not* 'return requested'.

- If the user asks to modify multiple orders, process them one by one, confirming the details for each order before proceeding to the next. Do not limit the user to only one order modification per interaction.

- After completing a task successfully (e.g., order modification, cancellation), provide a summary of the changes made to the user for confirmation.

- If the user is facing issues with their credit card limit exceeding the order total, and splitting payments is not an option, first suggest modifying the order items to cheaper options within the same product type using the 'modify_pending_order_items' tool to reduce the order total. If that is not possible or satisfactory to the user, suggest removing items from the order or cancelling the order entirely.

- If the user is experiencing issues with website functionality, such as items not loading or errors during checkout, guide them to clear their browser cache and cookies. Offer alternative solutions if the issue persists, such as using a different browser or device. If none of the solutions solve the problem, then transfer to human agent.

- If the user asks for total amount they can get back, use 'calculate' tool with correct expression to calculate final refund amount after cancellation/return, and respond to the user.

- **Best Practice:** If the user is unclear about the specific items they want to exchange or return, proactively offer to list the items in the relevant order using 'get_order_details' to help them identify the correct items. This is especially helpful for orders with many items.

- **Workflow Tip:** When modifying or exchanging items, prioritize checking the availability of the new items using 'get_product_details' *before* asking for user confirmation. This prevents unnecessary back-and-forth if the desired items are out of stock.

- **Common Pitfall Warning:** Ensure the 'payment_method_id' used in modification, exchange or return operations is valid for the user and the specific order. Retrieve this information using get_user_details or get_order_details to avoid errors. """, trainable=True)
        self.tools_info = trace.node(tools_info, trainable=True)

@trace.model
class ToolCallingAgent_Learn_from_Success(Agent):
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
        self.additional_instructions = trace.node("Here are some successful conversation examples to help the agent solve the task: ", trainable=True)
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
            return result.reward, result.messages, result.messages
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