from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from tau_bench.envs.user import UserStrategy
# os.environ["TRACE_LITELLM_MODEL"] = "gemini/gemini-2.0-flash" # optional

import opto 
from opto.trace import bundle, node 
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

class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
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
            trimmed_messages = trim_conversation_messages(messages, model=self.model)
            
            # Prepare completion arguments
            completion_kwargs = {
                "messages": trimmed_messages,
                "model": self.model,
                "custom_llm_provider": self.provider,
                "tools": self.tools_info,
                "temperature": self.temperature,
            }
            
            # Add api_base only for local/hosted providers
            if self.provider in ["hosted_vllm", "vllm"]:
                completion_kwargs["api_base"] = "http://127.0.0.1:8000/v1"
            
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
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


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



@bundle()
def get_result_for_tools_info(tools_info, env, config):
    "This function is used to get the result for an agent with a given tools_info. Specifically, tools_info is a node that contains information about the tools that the agent can use. It is used to create an agent that can use the tools to solve the task."
    agent = ToolCallingAgent(
        tools_info=tools_info,
        wiki=env.wiki,
        model=config.model,
        provider=config.model_provider,
        temperature=config.temperature
    )
    print(f"\nRunning task {config.task_ids[0]}")
    result = agent.solve(
        env=env,
        task_index=config.task_ids[0]  # Use the first (and only) task ID
    )
    if result.reward == 1:
        return result.reward,"Correct","Correct"
    else:
        return result.reward,result.messages,result.info

def main():
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

    tools_info = node(env.tools_info, trainable=True)
    # print(f"Tools info: {tools_info.data}")
    optimizer = OptoPrime([tools_info])
    optimizer.objective ="You need to change the <value> of the variables in #Variables to improve the output in accordance to #Feedback. In this task, please optimize tool information for a retail customer service agent. You only need to change the description part of the tool information, not the name or parameters."

    reward, messages, info = get_result_for_tools_info(tools_info, env, config)
    
    
    print("\nTask Results:")
    print(f"✅ Success" if reward.data == 1 else "❌ Failed")
    # print(f"Info: {result.data.info}")
    if reward.data == 1:
        feedback = "Correct"
    else:
        feedback = "The agent failed to solve the task. Here is the conversation history: " + "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages.data])
        # print(feedback)

    
    # feedback = result.data.messages
    optimizer.zero_feedback()
    optimizer.backward(info, feedback)
    optimizer.step(verbose=True)



if __name__ == "__main__":
    main() 