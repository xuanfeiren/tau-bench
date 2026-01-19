# agents/tool_calling_agent_opto.py
import json
from litellm import completion
from typing import List, Optional, Dict, Any
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from tau_bench.retry_utils import auto_retry_with_exponential_backoff
from opto import trace
import dspy
from dspy.signatures import Signature, InputField, OutputField
from dspy.primitives import Module
import copy

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
        self.conversations = trace.node({}, trainable=True)  # Dict[int, str] mapping task_index to conversation
        self.model = model
        self.provider = provider
        self.temperature = temperature

    @trace.bundle()
    def solve(self, tools_info, conversations, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
        """Agent solves the task with the given tools_info and task-specific conversation examples."""
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
        
        # Get task-specific conversation if available
        task_conversation = ""
        if task_index is not None and task_index in conversations:
            task_conversation = f"\n\nHere is a successful conversation example for this task, you can use it to help you solve the task:\n{conversations[task_index]}\n"
        else:
            task_conversation = "You are a helpful assistant."
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "system", "content": f"You are a helpful assistant.{task_conversation}"},
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
        
        return self.solve(self.tools_info, self.conversations, env, task_input)
    
    def set_env(self, env):
        """Set the environment for this agent."""
        self._env = env


# List of 12 additional instructions (index 0 is the original, 1-12 are new ones)
ADDITIONAL_INSTRUCTIONS_LIST = [
    # Index 0: Original instruction
    "Here are the additional instructions to help the agent solve the task: ",
    
    # Index 1: Slot discipline
    """Maintain a checklist of required args {order_id, item_ids, new_item_ids, payment_method_id, address}. Ask exactly ONE targeted question at a time to fill the next missing slot.""",
    
    # Index 2: Freshen-before-write
    """Immediately before any write call, re-read the order/user to confirm status and IDs (e.g., get_order_details). If anything changed, repair your plan first.""",
    
    # Index 3: ID grounding hygiene
    """Never invent IDs. Resolve product/item_id from product type + option tuple via read tools; echo the tuple back to the user for confirmation wording only.""",
    
    # Index 4: Deterministic candidate pick
    """If multiple items match the user's description, present up to 3 candidates sorted by (best option match → lowest price → item_id) and ask the user to choose.""",
    
    # Index 5: "Once-only" staging buffer
    """For modify/exchange (one-shot actions), accumulate all item mappings in a staging list; call the tool only when the list is complete.""",
    
    # Index 6: Price-diff preview (use calculate)
    """Before a write, compute and show: old subtotal → new subtotal and Δ. State the required payment method if Δ>0, or refund path if Δ<0.""",
    
    # Index 7: Error-repair loop
    """On tool error (not found / insufficient balance / invalid args), switch to a focused fix: correct the field, propose an alternative (e.g., different payment method/variant), then retry once.""",
    
    # Index 8: Post-write verify & stop
    """After any write, re-read the affected record(s) to verify the intended state. If correct, announce completion concisely and stop; otherwise, repair.""",
    
    # Index 9: Minimal-calls planner
    """Plan the fewest tool calls to reach the goal (read→compute→single write). Avoid redundant reads and never mix talking to the user and tool calls in the same turn.""",
    
    # Index 10: Option normalization
    """Normalize free-text options to catalog keys (e.g., "sky blue"→"blue", "USB-C"→"usb"). If unsure, show exact catalog labels and ask the user to pick.""",
    
    # Index 11: Safe ordering for compound goals
    """For multi-step asks: (1) gather/verify → (2) check constraints → (3) compute cost → (4) act once → (5) verify → (6) next subgoal. Do not interleave writes.""",
    
    # Index 12: Write-once guardrail
    """Track whether a write action (modify/exchange/cancel/return) has been executed; if so, refuse additional writes on the same order and route to the next policy-compliant option."""
    ]

@trace.model
class ToolCallingAgent_v2(Agent):
    """Only use the additional instructions as the trainable parameter"""
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        additional_instructions_index: int = 0,
    ):
        super().__init__()
        self.tools_info = tools_info
        self.wiki = wiki
        self.ADDITIONAL_INSTRUCTIONS_LIST = ADDITIONAL_INSTRUCTIONS_LIST
        
        # Select the appropriate additional instructions based on index
        if 0 <= additional_instructions_index < len(self.ADDITIONAL_INSTRUCTIONS_LIST):
            selected_instruction = self.ADDITIONAL_INSTRUCTIONS_LIST[additional_instructions_index]
        else:
            # Default to index 0 if invalid index provided
            selected_instruction = self.ADDITIONAL_INSTRUCTIONS_LIST[0]
            
        self.additional_instructions = trace.node(selected_instruction, trainable=True)
        self.model = model
        self.provider = provider
        self.temperature = temperature

    @trace.bundle()
    def solve(self, additional_instructions, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
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
            return None, []
            
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
                "tools": self.tools_info,
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
                return None, messages
            
            if step_result == -1:
                print(f"Step {step}: Return 0 reward due to BadRequest error")
                return 0, messages
            
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
        # for debugging
        # print(result.info)
        # breakpoint()
        if result.reward == 1:
            return result.reward, result.messages
        else:
            return result.reward, result.messages
        
    def forward(self, task_input):
        """Forward pass of the agent for trainer compatibility."""
        env = getattr(self, '_env', None)
        if env is None:
            raise ValueError("Environment not set. Call set_env() before forward pass.")
        
        return self.solve(self.additional_instructions, env, task_input)
    
    def set_env(self, env):
        """Set the environment for this agent."""
        self._env = env

import random
class DummyToolCallingAgent(ToolCallingAgent_v2):
    """Dummy agent that does not call LLM"""
    @trace.bundle()
    def solve(self, additional_instructions, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
        """Agent solves the task with the given tools_info."""
        # return a random reward between 0 and 1
        reward = random.random()
        # construct a dummy conversation for messages
        messages = [
            {"role": "system", "content": self.wiki},
            {"role": "system", "content": additional_instructions},
            {"role": "user", "content": "This is a dummy conversation"},
        ]
        return reward, messages, "info"

# -----------------------------------------------------------------------------
# NEW DSPy Compatible Agent
# -----------------------------------------------------------------------------

class RetailTaskSignature(Signature):
    """Here are the additional instructions to help the agent solve the task: """
    x = InputField(desc="The task index/input")
    result = OutputField(desc="The agent execution result")

class ToolCallingAgentDSPy(Module):
    """
    A DSPy-compatible version of ToolCallingAgent_v2.
    It uses dspy.Predict to expose the additional_instructions as a trainable parameter (docstring).
    """
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        additional_instruction: Optional[str] = None,
    ):
        super().__init__()
        self.tools_info = tools_info
        self.wiki = wiki
        
        # Use provided instruction or default
        base_instruction = additional_instruction or "Here are the additional instructions to help the agent solve the task: "
            
        # Use dspy.Predict to hold the instruction
        self.prog = dspy.Predict(RetailTaskSignature)
        self.prog.signature.__doc__ = base_instruction
        
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self._env = None

    def solve(self, additional_instructions, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
        """
        Agent solves the task with the given tools_info.
        (Logic copied from ToolCallingAgent_v2.solve but without Opto traces)
        """
        total_cost = 0.0
        
        # Wrap env.reset with retry logic
        def reset_env():
            return env.reset(task_index=task_index)
        
        env_reset_res = auto_retry_with_exponential_backoff(
            reset_env,
            operation_name="Environment reset"
        )
        
        if env_reset_res is None:
            print("Environment reset failed, return None reward")
            return 0.0, []
            
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
                "tools": self.tools_info,
                "temperature": self.temperature,
            }
            
            def step_interaction():
                res = completion(**completion_kwargs)
                next_message = res.choices[0].message.model_dump()
                cost = res._hidden_params.get("response_cost")
                if cost is not None:
                    nonlocal total_cost
                    total_cost += cost
                
                action = message_to_action(next_message)
                env_response = env.step(action)
                return next_message, action, env_response
            
            step_result = auto_retry_with_exponential_backoff(
                step_interaction, 
                operation_name=f"Step {step}"
            )
            
            if step_result is None:
                print(f"Step {step}: Return None reward due to interaction failure")
                return 0.0, messages
            
            if step_result == -1:
                print(f"Step {step}: Return 0 reward due to BadRequest error")
                return 0.0, messages
            
            next_message, action, env_response = step_result
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
        
        return float(reward), messages

    def forward(self, x):
        """Forward pass required by DSPy."""
        if self._env is None:
            raise ValueError("Environment not set. Call set_env() before forward pass.")
            
        current_instruction = self.prog.signature.__doc__
        # from opto.optimizers.utils import print_color
        # print_color(f"Current instruction: {current_instruction}\n", "green")
        
        # Call the predictor to register it in the trace for GEPA
        # We ignore the result, but this ensures GEPA sees the module execution
        try:
            # Convert input to string to match signature expectation usually
            self.prog(x=str(x))
        except Exception:
            pass # Ignore errors in dummy call
        
        # Deepcopy environment to support parallel execution
        local_env = copy.deepcopy(self._env)
        reward, messages = self.solve(current_instruction, local_env, x)
        
        return dspy.Prediction(reward=reward, messages=messages)
    
    def set_env(self, env):
        self._env = env


class ToolCallingAgent_openevolve(Agent):
    """Only use the additional instructions as the trainable parameter"""
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        super().__init__()
        self.tools_info = tools_info
        self.wiki = wiki
        
        # Select the appropriate additional instructions based on index
        self.additional_instructions = "Here are the additional instructions to help the agent solve the task: "
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def solve(self, additional_instructions, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
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
            return None, []
            
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
                "tools": self.tools_info,
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
                return None, messages
            
            if step_result == -1:
                print(f"Step {step}: Return 0 reward due to BadRequest error")
                return 0, messages
            
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
        # for debugging
        # print(result.info)
        # breakpoint()
        if result.reward == 1:
            return result.reward, result.messages
        else:
            return result.reward, result.messages
        
    def forward(self, task_input):
        """Forward pass of the agent for trainer compatibility."""
        env = getattr(self, '_env', None)
        if env is None:
            raise ValueError("Environment not set. Call set_env() before forward pass.")
        
        return self.solve(self.additional_instructions, env, task_input)
    
    def set_env(self, env):
        """Set the environment for this agent."""
        self._env = env