# Copyright Sierra

import abc
import enum
import random
from litellm import completion

from typing import Optional, List, Dict, Any, Union
from tau_bench.model_utils.model.utils import trim_conversation_messages
from tau_bench.retry_utils import auto_retry_with_exponential_backoff


class BaseUserSimulationEnv(abc.ABC):
    metadata = {}

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class HumanUserSimulationEnv(BaseUserSimulationEnv):
    def reset(self, instruction: str) -> str:
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")

    def get_total_cost(self) -> float:
        return 0


class LLMUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.provider = provider
        self.total_cost = 0.0
        self.reset()

    def _make_completion_call(self, messages: List[Dict[str, Any]]) -> str:
        """Helper method to make a single completion call."""
        # Trim messages to prevent context window errors
        trimmed_messages = trim_conversation_messages(messages, model=self.model)
        
        # Prepare completion arguments
        completion_kwargs = {
            "model": self.model,
            "custom_llm_provider": self.provider,
            "messages": trimmed_messages,
        }
        
        # Add api_base only for local/hosted providers
        if self.provider in ["hosted_vllm", "vllm"]:
            completion_kwargs["api_base"] = "http://127.0.0.1:8000/v1"
        
        res = completion(**completion_kwargs)
        cur_message = res.choices[0].message
        cost = res._hidden_params.get("response_cost")
        self.total_cost = cost if cost is not None else 0.0
        self.messages.append(cur_message.model_dump())
        return cur_message.content

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """Generate next message using auto retry with exponential backoff."""
        def completion_call():
            return self._make_completion_call(messages)
        
        result = auto_retry_with_exponential_backoff(
            completion_call,
            max_retries=10,
            operation_name="LLM User Simulation"
        )
        
        if result is None:
            raise Exception("Failed to generate message after all retry attempts")
        
        return result

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class ReactUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__(model=model, provider=provider)
        self.reset()

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- First, generate a Thought about what to do next (this message will not be sent to the agent).
- Then, generate a one line User Response to simulate the user's message (this message will be sent to the agent).
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as the User Response without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.

Format:

Thought:
<the thought>

User Response:
<the user response (this will be parsed and sent to the agent)>"""

    def _make_completion_call(self, messages: List[Dict[str, Any]]) -> str:
        """Helper method to make a single completion call for React user."""
        # Trim messages to prevent context window errors
        trimmed_messages = trim_conversation_messages(messages, model=self.model)
        
        # Prepare completion arguments
        completion_kwargs = {
            "model": self.model,
            "custom_llm_provider": self.provider,
            "messages": trimmed_messages,
        }
        
        # Add api_base only for local/hosted providers
        if self.provider in ["hosted_vllm", "vllm"]:
            completion_kwargs["api_base"] = "http://127.0.0.1:8000/v1"
        
        res = completion(**completion_kwargs)
        cur_message = res.choices[0].message
        cost = res._hidden_params.get("response_cost")
        self.total_cost = cost if cost is not None else 0.0
        self.messages.append(cur_message.model_dump())
        return self.parse_response(cur_message.content)

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """Generate next message using auto retry with exponential backoff."""
        def completion_call():
            return self._make_completion_call(messages)
        
        result = auto_retry_with_exponential_backoff(
            completion_call,
            max_retries=10,
            operation_name="React User Simulation"
        )
        
        if result is None:
            raise Exception("Failed to generate message after all retry attempts: " + str(Exception))
        
        return result

    def parse_response(self, response: str) -> str:
        if "###STOP###" in response:
            return "###STOP###"
        elif "Thought:" in response:
            _, user_response = response.split("Thought:")
            return user_response.strip()
        elif "User Response:" in response:
            _, user_response = response.split("User Response:")
            return user_response.strip()
        else:
            raise ValueError(f"Invalid response format: {response}")

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class VerifyUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 3) -> None:
        self.model = model
        self.provider = provider
        self.max_attempts = max_attempts
        self.reset()

    def _make_single_completion_attempt(self, messages: List[Dict[str, Any]]) -> tuple:
        """Make a single completion attempt and return (message, success)."""
        # Trim messages to prevent context window errors
        trimmed_messages = trim_conversation_messages(messages, model=self.model)
        
        # Prepare completion arguments
        completion_kwargs = {
            "model": self.model,
            "custom_llm_provider": self.provider,
            "messages": trimmed_messages,
        }
        
        # Add api_base only for local/hosted providers
        if self.provider in ["hosted_vllm", "vllm"]:
            completion_kwargs["api_base"] = "http://127.0.0.1:8000/v1"
        
        res = completion(**completion_kwargs)
        cur_message = res.choices[0].message
        cost = res._hidden_params.get("response_cost")
        self.total_cost = cost if cost is not None else 0.0
        
        # Check if this message is verified
        is_verified = verify(self.model, self.provider, cur_message.content, messages)
        
        if is_verified:
            self.messages.append(cur_message.model_dump())
            return cur_message.content, True
        else:
            return cur_message.content, False

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """Generate next message with verification and retry logic."""
        attempts = 0
        last_message = None
        
        while attempts < self.max_attempts:
            def completion_call():
                return self._make_single_completion_attempt(messages)
            
            result = auto_retry_with_exponential_backoff(
                completion_call,
                max_retries=3,
                operation_name=f"Verify User Simulation (attempt {attempts + 1})"
            )
            
            if result is not None:
                message_content, is_verified = result
                last_message = message_content
                
                if is_verified:
                    return message_content
            
            attempts += 1
        
        # If we've exhausted all verification attempts, return the last message
        return last_message if last_message is not None else "Sorry, I couldn't generate a proper response."

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


def map_role_label(role: str) -> str:
    if role == "user":
        return "Customer"
    elif role == "assistant":
        return "Agent"
    else:
        return role.capitalize()


def verify(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> bool:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a Customer and an Agent. The Customer has generated a Response, and you need to verify if it is satisfactory (true) or not (false).
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

-----

Classification:"""
    
    def completion_call():
        # Prepare completion arguments
        completion_kwargs = {
            "model": model,
            "custom_llm_provider": provider,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # Add api_base only for local/hosted providers
        if provider in ["hosted_vllm", "vllm"]:
            completion_kwargs["api_base"] = "http://127.0.0.1:8000/v1"
        
        res = completion(**completion_kwargs)
        return "true" in res.choices[0].message.content.lower()
    
    result = auto_retry_with_exponential_backoff(
        completion_call,
        max_retries=3,
        operation_name="Verify user response"
    )
    
    if result is None:
        # Default to False if verification fails
        return False
    
    return result


def reflect(
    model: str, provider: str, response: str, messages: List[Dict[str, Any]]
) -> str:
    transcript = "\n".join(
        [
            f"{map_role_label(message['role'])}: {message['content']}"
            for message in messages
        ]
    )
    prompt = f"""You are a supervisor of the Agent in the conversation. You are given a Transcript of a conversation between a (simulated) Customer and an Agent. The Customer generated a Response that was marked as unsatisfactory by you.
You need to generate a Reflection on what went wrong in the conversation, and propose a new Response that should fix the issues.
Your answer will be parsed, so do not include any other text than the classification (true or false).
    
# Transcript:
{transcript}

# Response:
{response}

# Format:

Reflection:
<the reflection>

Response:
<the response (this will be parsed and sent to the agent)>"""
    
    def completion_call():
        # Prepare completion arguments
        completion_kwargs = {
            "model": model,
            "custom_llm_provider": provider,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # Add api_base only for local/hosted providers
        if provider in ["hosted_vllm", "vllm"]:
            completion_kwargs["api_base"] = "http://127.0.0.1:8000/v1"
        
        res = completion(**completion_kwargs)
        response_content = res.choices[0].message.content
        _, response = response_content.split("Response:")
        return response.strip()
    
    result = auto_retry_with_exponential_backoff(
        completion_call,
        max_retries=3,
        operation_name="Reflect on user response"
    )
    
    if result is None:
        # Return original response if reflection fails
        return response
    
    return result


class ReflectionUserSimulationEnv(LLMUserSimulationEnv):
    def __init__(self, model: str, provider: str, max_attempts: int = 2) -> None:
        self.model = model
        self.provider = provider
        self.max_attempts = max_attempts
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        cur_messages = messages.copy()
        initial_response = super().generate_next_message(cur_messages)
        if verify(self.model, self.provider, initial_response, cur_messages):
            return initial_response
        attempts = 1
        while attempts < self.max_attempts:
            new_message = reflect(
                self.model, self.provider, initial_response, cur_messages
            )
            cur_messages.append({"role": "user", "content": new_message})
            new_response = super().generate_next_message(cur_messages)
            if verify(self.model, self.provider, new_response, cur_messages):
                return new_response
            attempts += 1
        return initial_response

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_total_cost(self) -> float:
        return self.total_cost


class UserStrategy(enum.Enum):
    HUMAN = "human"
    LLM = "llm"
    REACT = "react"
    VERIFY = "verify"
    REFLECTION = "reflection"


def load_user(
    user_strategy: Union[str, UserStrategy],
    model: Optional[str] = "gpt-4o",
    provider: Optional[str] = None,
) -> BaseUserSimulationEnv:
    if isinstance(user_strategy, str):
        user_strategy = UserStrategy(user_strategy)
    if user_strategy == UserStrategy.HUMAN:
        return HumanUserSimulationEnv()
    elif user_strategy == UserStrategy.LLM:
        if model is None:
            raise ValueError("LLM user strategy requires a model")
        if provider is None:
            raise ValueError("LLM user strategy requires a model provider")
        return LLMUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REACT:
        if model is None:
            raise ValueError("React user strategy requires a model")
        if provider is None:
            raise ValueError("React user strategy requires a model provider")
        return ReactUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.VERIFY:
        if model is None:
            raise ValueError("Verify user strategy requires a model")
        if provider is None:
            raise ValueError("Verify user strategy requires a model provider")
        return VerifyUserSimulationEnv(model=model, provider=provider)
    elif user_strategy == UserStrategy.REFLECTION:
        if model is None:
            raise ValueError("Reflection user strategy requires a model")
        if provider is None:
            raise ValueError("Reflection user strategy requires a model provider")
        return ReflectionUserSimulationEnv(model=model, provider=provider)
    raise ValueError(f"Unknown user strategy {user_strategy}")
