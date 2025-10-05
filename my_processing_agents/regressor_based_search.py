# Regressor-based search algorithm using pre-trained linear regressor weights
# This algorithm uses LLM to generate new candidates and evaluates them using a pre-trained regressor
# python my_processing_agents/regressor_based_search.py --test_frequency 5 --run_name regressor_based_search --num_steps 50
import numpy as np
import copy
import heapq
import time
import json
import os
import pickle
import argparse
from typing import Union, List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass

# Tau-bench imports
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from tau_bench.envs.user import UserStrategy
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

# Opto imports
import opto 
from opto import trace
from opto.trace.nodes import GRAPH, ParameterNode
from opto.trace.modules import Module 
from opto.optimizers.optimizer import Optimizer
from opto.trainer.loggers import WandbLogger, BaseLogger
from opto.trainer.utils import async_run, safe_mean
from opto.trainer.algorithms.basic_algorithms import batchify
from opto.trainer.evaluators import evaluate
from opto.utils.llm import LLM
from opto.optimizers.utils import extract_xml_like_data
from opto.utils.auto_retry import retry_with_exponential_backoff
from opto.optimizers.utils import print_color
# Agent imports
from agents.tool_calling_agent import ToolCallingAgent_v2 as ToolCallingAgent
from pretained_regressor import PretrainedLinearRegressor, get_parameter_text
# Priority search imports - using local definitions to avoid import issues
# from opto.features.priority_search.priority_search import ModuleCandidate
# from opto.features.priority_search.search_template import SearchTemplate, Samples, BatchRollout, save_train_config
# from opto.features.priority_search.utils import set_module_parameters, remap_update_dict, create_module_from_update_dict

# Import the ModuleCandidate from the attached priority_search.py
import sys
import os

# Try to import ModuleCandidate, fall back to a simple mock if not available

from opto.features.priority_search.priority_search import ModuleCandidate

import litellm 
litellm.drop_params = True
litellm.suppress_debug_info = True

# Provider configuration
provider = "gemini"
os.environ["TRACE_LITELLM_MODEL"] = f"{provider}/gemini-2.0-flash"



class LLMCandidateGenerator:
    """Generate new candidates using LLM with OptoPrimeV2-style prompts."""
    
    def __init__(self, model_name="gemini/gemini-2.0-flash", temperature=0.0):
        self.llm = LLM(model=model_name)
        self.temperature = temperature
    
    def generate_candidates(self, base_module, optimizer, memory, num_candidates=5):
        """Generate new candidates using LLM based on memory of past candidates."""
        
        # Create prompt based on OptoPrimeV2 structure
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(base_module, memory, num_candidates)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # print("system_prompt: ", system_prompt)
        # print("user_prompt: ", user_prompt)

        try:
            response = self.llm(messages=messages, temperature=self.temperature, max_tokens=4000)
            response_text = response.choices[0].message.content
            
            # Parse candidates from response
            candidates = self._parse_candidates(response_text, base_module, optimizer)
            return candidates
            
        except Exception as e:
            print(f"Error generating candidates: {e}")
            return []
    
    def _create_system_prompt(self):
        """Create system prompt for candidate generation."""
        return """You are an AI optimization assistant tasked with generating improved parameter configurations for a system.

Your goal is to analyze past performance data and generate new parameter configurations that are likely to achieve higher performance scores.

You will receive information about previous parameter configurations and their performance scores (0.0 to 1.0, where 1.0 is perfect)."""
    
    def _create_user_prompt(self, base_module, memory, num_candidates):
        """Create user prompt with memory context and generation request."""
        
        # Format memory examples
        memory_text = ""
        if memory:
            memory_text = "## Previous Configurations and Scores\n\n"
            # Sort memory by score (descending)
            sorted_memory = sorted(memory, key=lambda x: x[0], reverse=True)
            
            for i, (score, candidate) in enumerate(sorted_memory[:10]):  # Show top 10
                memory_text += f"### Configuration {i+1} (Score: {score:.3f})\n"
                # Format parameters with proper names
                params_display = get_parameter_text(candidate)
                memory_text += f"Parameters: {params_display}\n\n"
        
        # Get base parameters with proper names
        base_params = {p.py_name if hasattr(p, 'py_name') else str(p): p.data for p in base_module.parameters()}
        
        prompt = f"""## Current Task
Generate {num_candidates} new parameter configurations to improve system performance.

## Base Configuration
{base_params}

{memory_text}

## Task
Generate {num_candidates} new parameter configurations that improve upon the best previous results by:
1. Analyzing what made successful configurations work well
2. Identifying weaknesses in poor-performing configurations  
3. Creating variations that combine the best aspects while addressing weaknesses
4. Exploring promising new directions based on the patterns you observe

## Output Format
Provide your response in the following XML format:

<reasoning>
[Your analysis of what makes configurations successful and your strategy for improvement]
</reasoning>

<candidates>
<candidate index="1">
<reasoning>
[Brief explanation of why this configuration should perform well]
</reasoning>
<parameters>
[Complete parameter dictionary for this candidate]
</parameters>
</candidate>

<candidate index="2">
<reasoning>
[Brief explanation of why this configuration should perform well]
</reasoning>
<parameters>
[Complete parameter dictionary for this candidate]
</parameters>
</candidate>

... (continue for all {num_candidates} candidates)
</candidates>

Generate diverse candidates that explore different promising directions while building on successful patterns."""   
        
        return prompt
    
    def _parse_candidates(self, response_text, base_module, optimizer):
        """Parse candidate configurations from LLM response."""
        candidates = []
        
        try:
            # Extract candidates section
            import re
            candidates_match = re.search(r'<candidates>(.*?)</candidates>', response_text, re.DOTALL)
            if not candidates_match:
                print("No candidates section found in response")
                return candidates
            
            candidates_section = candidates_match.group(1)
            
            # Extract individual candidates
            candidate_pattern = r'<candidate[^>]*index=["\'](\d+)["\'][^>]*>(.*?)</candidate>'
            candidate_matches = re.finditer(candidate_pattern, candidates_section, re.DOTALL)
            
            for match in candidate_matches:
                candidate_content = match.group(2)
                
                # Extract parameters
                params_match = re.search(r'<parameters>(.*?)</parameters>', candidate_content, re.DOTALL)
                if params_match:
                    params_text = params_match.group(1).strip()
                    
                    try:
                        # Try to evaluate as Python dict with string keys
                        params_dict = eval(params_text)
                        
                        # Map string parameter names back to ParameterNode objects
                        update_dict = {}
                        param_name_to_node = {p.py_name if hasattr(p, 'py_name') else str(p): p for p in base_module.parameters()}
                        
                        for param_name, value in params_dict.items():
                            if param_name in param_name_to_node:
                                update_dict[param_name_to_node[param_name]] = value
                            else:
                                print(f"Warning: Parameter '{param_name}' not found in base module parameters")
                        
                        # Create ModuleCandidate with ParameterNode keys
                        candidate = ModuleCandidate(
                            base_module=base_module,
                            update_dict=update_dict,
                            optimizer=optimizer
                        )
                        candidates.append(candidate)
                        
                    except Exception as e:
                        print(f"Error parsing candidate parameters: {e}")
                        continue
        
        except Exception as e:
            print(f"Error parsing candidates: {e}")
        
        return candidates


class RegressorBasedSearch:
    """Search algorithm that uses LLM to generate candidates and regressor to evaluate them."""
    
    def __init__(self, 
                 agent: trace.Module,
                 optimizer: Optimizer,
                 regressor: PretrainedLinearRegressor,
                 generator: LLMCandidateGenerator,
                 logger: BaseLogger = None,
                 num_threads: int = None,
                 memory_size: int = 100):
        
        self.agent = agent
        self.optimizer = optimizer
        self.logger = logger
        self.num_threads = num_threads
        
        self.regressor = regressor
        self.generator = generator
        self.memory = []  # List of (score, candidate) tuples
        self.memory_size = memory_size
        self.best_candidate = None
        self.best_score = -1.0
    
    def _add_to_memory(self, candidate, score):
        """Add candidate to memory with score-based priority."""
        self.memory.append((score, candidate))
        
        # Keep memory sorted by score (descending) and limit size
        self.memory.sort(key=lambda x: x[0], reverse=True)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[:self.memory_size]
        
        # Update best candidate
        if score > self.best_score:
            self.best_score = score
            self.best_candidate = candidate

    def _print_memory(self):
        """Print the memory."""
        for score, candidate in self.memory:
            print_color(f"Score: {score}", "green")
            print_color(f"Parameters: {get_parameter_text(candidate)}", "blue")

    def train(self,
              guide,
              test_dataset: Dict[str, List[Any]] = None,
              num_steps: int = 10,
              num_candidates_per_step: int = 5,
              num_eval_samples: int = 10,
              test_frequency: int = None,
              log_frequency: int = 1,
              save_frequency: int = None,
              save_path: str = None,
              verbose: bool = False,
              **kwargs):
        """Main training loop."""
        
        print(f"Starting regressor-based search for {num_steps} steps")
        print(f"Generating {num_candidates_per_step} candidates per step")
        
        # Initialize with base module
        base_candidate = ModuleCandidate(base_module=self.agent)
        base_score = self.regressor.predict_score(base_candidate)
        print("base_score: ", base_score)
        # breakpoint()
        self._add_to_memory(base_candidate, base_score)
        
        print(f"Base candidate score: {base_score:.4f}")
        self._print_memory()
        # evaluate base candidate
        base_score = self._evaluate_candidate(base_candidate, guide, test_dataset, num_eval_samples)
        print(f"Base candidate test score: {base_score:.4f}")
        self.logger.log('test_score', base_score, 0)
        
        for step in range(num_steps):

            step_start_time = time.time()
            print(f"\n=== Step {step + 1}/{num_steps} ===")
            
            # Generate new candidates using LLM
            new_candidates = self.generator.generate_candidates(
                base_module=self.agent,
                optimizer=self.optimizer,
                memory=self.memory,
                num_candidates=num_candidates_per_step
            )
            
            print(f"Generated {len(new_candidates)} candidates")
            
            if not new_candidates:
                print("No candidates generated, skipping step")
                continue
            
            # Evaluate candidates using regressor
            candidate_scores = []
            for candidate in new_candidates:
                score = self.regressor.predict_score(candidate)
                candidate_scores.append(score)
                self._add_to_memory(candidate, score)
            self._print_memory()
            # Log results
            if candidate_scores:
                avg_score = np.mean(candidate_scores)
                max_score = np.max(candidate_scores)
                print(f"Step {step + 1} - Avg score: {avg_score:.4f}, Max score: {max_score:.4f}")
                print(f"Best overall score: {self.best_score:.4f}")
                
                if self.logger:
                    current_step = step + 1
                    self.logger.log('avg_score', avg_score, current_step)
                    self.logger.log('max_score', max_score, current_step)
                    self.logger.log('best overall score', self.best_score, current_step)
                    self.logger.log('num_candidates', len(new_candidates), current_step)
                    self.logger.log('memory_size', len(self.memory), current_step)
            
            # Test evaluation
            if test_frequency and (step + 1) % test_frequency == 0:
                print(f"\nRunning test evaluation at step {step + 1}")
                test_score = self._evaluate_candidate(self.best_candidate, guide, test_dataset, num_eval_samples)
                print(f"Test score for best candidate: {test_score:.4f}")
                
                if self.logger:
                    self.logger.log('test_score', test_score, step + 1)
            
            # Save checkpoint
            if save_frequency and save_path and (step + 1) % save_frequency == 0:
                self._save_checkpoint(save_path, step + 1)
            
            step_time = time.time() - step_start_time
            print(f"Step {step + 1} completed in {step_time:.2f} seconds")
        
        print(f"\nTraining completed. Best score: {self.best_score:.4f}")
        
        
            
        
        return self.best_candidate
    
    def _evaluate_candidate(self, candidate, guide, dataset, num_eval_samples):
        """Evaluate a candidate on the actual task."""
        if not candidate:
            return 0.0
        
        # Apply candidate to agent
        test_agent = candidate.get_module()
        
        # Run evaluation
        
        # Unpack dataset into inputs and infos
        
        
        results = evaluate(
            agent=test_agent,
            guide=guide,
            inputs=dataset['inputs'],
            infos=dataset['infos'],
            num_samples=num_eval_samples,
            num_threads=self.num_threads
        )
        
        return safe_mean(results)
                
        
    
    def _save_checkpoint(self, save_path, step):
        """Save current state to checkpoint."""
        try:
            checkpoint = {
                'step': step,
                'best_candidate': self.best_candidate,
                'best_score': self.best_score,
                'memory': self.memory[:10],  # Save top 10 candidates
            }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(f"{save_path}_step_{step}.pkl", 'wb') as f:
                pickle.dump(checkpoint, f)
            
            print(f"Checkpoint saved to {save_path}_step_{step}.pkl")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")


class TeacherGuide:
    """Guide that extracts reward and feedback from the agent's output."""
    
    def __init__(self, env: Env, config: RunConfig):
        self.env = env
        self.config = config
        
    def get_feedback(self, task, output: SolveResult, info):   
        """Get feedback from the agent's output."""
        reward, messages, info = output
        if info == "BadRequest":
            return 0, "BadRequestError. Please adjust the tool information to the correct form."
        if reward == 1:
            feedback = "Correct"
        else:
            conversation_parts = []
            for msg in messages:
                msg_str = f"{msg['role']}: {msg.get('content', '')}"
                
                if 'tool_calls' in msg and msg['tool_calls']:
                    tool_calls_str = []
                    for tool_call in msg['tool_calls']:
                        if 'function' in tool_call:
                            func_name = tool_call['function'].get('name', '')
                            func_args = tool_call['function'].get('arguments', '')
                            tool_calls_str.append(f"Tool: {func_name}({func_args})")
                    if tool_calls_str:
                        msg_str += f" [Tool Calls: {'; '.join(tool_calls_str)}]"
                
                if msg['role'] == 'tool':
                    tool_name = msg.get('name', '')
                    tool_call_id = msg.get('tool_call_id', '')
                    msg_str = f"tool ({tool_name}, ID: {tool_call_id}): {msg.get('content', '')}"
                
                conversation_parts.append(msg_str)
            
            feedback = "The agent failed to solve the task. Here is the conversation history: " + "\n".join(conversation_parts)
        return reward, feedback
        
    def metric(self, task, output: SolveResult, info):
        """Metric for the agent's performance."""
        reward, messages, info = output
        return reward


def create_dataset(env, num_tasks=10):
    """Create dataset from environment tasks."""
    inputs = []
    infos = []
    
    for task_id in range(num_tasks):
        inputs.append(task_id)
        infos.append(task_id)
    
    return {'inputs': inputs, 'infos': infos}


def main():
    """Main function for regressor-based search."""
    parser = argparse.ArgumentParser(description='Train agent using regressor-based search')
    
    # Dataset parameters
    parser.add_argument('--num_test_samples', type=int, default=10,
                       help='Number of test samples')
    
    # Training parameters
    parser.add_argument('--num_steps', type=int, default=10,
                       help='Number of search steps')
    parser.add_argument('--num_candidates_per_step', type=int, default=5,
                       help='Number of candidates to generate per step')
    parser.add_argument('--num_threads', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--test_frequency', type=int, default=None,
                       help='How often to run test evaluation')
    parser.add_argument('--save_frequency', type=int, default=None,
                       help='How often to save checkpoints')
    parser.add_argument('--save_path', type=str, default='checkpoints/regressor_search_agent.pkl',
                       help='Path to save checkpoints')
    parser.add_argument('--num_eval_samples', type=int, default=10,
                       help='Number of times to evaluate each input')
    parser.add_argument('--memory_size', type=int, default=100,
                       help='Size of memory to store candidates')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    parser.add_argument('--generator_model', type=str, default='gemini/gemini-2.0-flash',
                       help='Model to use for candidate generation')
    parser.add_argument('--generator_temperature', type=float, default=0,
                       help='Temperature for candidate generation')
    
    # Regressor parameters
    # use Oct 3 pretrained linear regressor
    # The other one is trained with LinUCB.
    parser.add_argument('--weights_path', type=str, default='regressor_models/linear_reg_dim768_reg0.0001_Oct3_weights.npy',
                       help='Path to regressor weights')
    parser.add_argument('--bias_path', type=str, default='regressor_models/linear_reg_dim768_reg0.0001_Oct3_bias.npy',
                       help='Path to regressor bias')
    parser.add_argument('--embedding_model', type=str, default='gemini/text-embedding-004',
                       help='Embedding model for regressor')
    
    # Logging parameters
    parser.add_argument('--project_name', type=str, default='tau-bench-regressor-search',
                       help='Name of the project')
    parser.add_argument('--run_name', type=str, default='debug',
                       help='Name of the run')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Whether to print verbose output')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = RunConfig(
            model_provider=provider,
            user_model_provider=provider,
            model=args.model,
            user_model=args.user_model,
            num_trials=1,
            env="retail",
            agent_strategy="tool-calling",
            temperature=0.0,
            task_split="test",
            task_ids=list(range(args.num_test_samples)),
            log_dir="results",
            max_concurrency=1,
            seed=10,
            shuffle=0,
            user_strategy="llm",
            few_shot_displays_path=None
        )
        
        # Initialize environment
        print(f"Initializing retail environment with user strategy: {config.user_strategy}")
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
            task_index=0
        )
        
        # Create datasets
        print("Creating datasets...")
        test_dataset = create_dataset(env, num_tasks=args.num_test_samples)
        
        print(f"Test samples: {len(test_dataset['inputs'])}")
        
        # Initialize agent
        print(f"Initializing agent with model: {config.model}")
        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature
        )
        agent.set_env(env)
        
        # Initialize optimizer
        from opto.optimizers import OptoPrimeV2
        optimizer = OptoPrimeV2(agent.parameters(), max_tokens=8000)
        
        # Initialize regressor
        print(f"Loading regressor from {args.weights_path} and {args.bias_path}")
        regressor = PretrainedLinearRegressor(
            weights_path=args.weights_path,
            bias_path=args.bias_path,
            embedding_model=args.embedding_model
        )
        
        # Initialize candidate generator
        generator = LLMCandidateGenerator(
            model_name=args.generator_model,
            temperature=args.generator_temperature
        )
        
        # Initialize guide and logger
        guide = TeacherGuide(env, config)
        
        config_dict = {
            'num_test_samples': args.num_test_samples,
            'num_steps': args.num_steps,
            'num_candidates_per_step': args.num_candidates_per_step,
            'num_threads': args.num_threads,
            'test_frequency': args.test_frequency,
            'save_frequency': args.save_frequency,
            'num_eval_samples': args.num_eval_samples,
            'memory_size': args.memory_size,
            'model': args.model,
            'user_model': args.user_model,
            'generator_model': args.generator_model,
            'generator_temperature': args.generator_temperature,
            'weights_path': args.weights_path,
            'bias_path': args.bias_path,
            'embedding_model': args.embedding_model,
        }
        
        logger = WandbLogger(project=args.project_name, verbose=True, name=args.run_name, config=config_dict)
        
        # Create search algorithm
        print("Creating regressor-based search algorithm...")
        algorithm = RegressorBasedSearch(
            agent=agent,
            optimizer=optimizer,
            regressor=regressor,
            generator=generator,
            logger=logger,
            num_threads=args.num_threads,
            memory_size=args.memory_size
        )
        
        # Training parameters
        train_params = {
            "guide": guide,
            "test_dataset": test_dataset,
            "num_steps": args.num_steps,
            "num_candidates_per_step": args.num_candidates_per_step,
            "num_eval_samples": args.num_eval_samples,
            "test_frequency": args.test_frequency,
            "save_frequency": args.save_frequency,
            "save_path": args.save_path,
            "verbose": args.verbose,
        }
        
        # Start training
        print("Starting regressor-based search...")
        print(f"Number of steps: {args.num_steps}")
        print(f"Candidates per step: {args.num_candidates_per_step}")
        print(f"Test frequency: {args.test_frequency}")
        print(f"Memory size: {args.memory_size}")
        
        start_time = time.time()
        best_candidate = algorithm.train(**train_params)
        duration = time.time() - start_time
        
        print(f"Search completed in {duration:.2f} seconds")
        print(f"Best candidate score: {algorithm.best_score:.4f}")
        
        # Save final result
        # if args.save_path:
        #     final_save_path = args.save_path.replace('.pkl', '_final.pkl')
        #     try:
        #         with open(final_save_path, 'wb') as f:
        #             pickle.dump({
        #                 'best_candidate': best_candidate,
        #                 'best_score': algorithm.best_score,
        #                 'memory': algorithm.memory[:10],
        #                 'config': config_dict
        #             }, f)
        #         print(f"Final result saved to {final_save_path}")
        #     except Exception as e:
        #         print(f"Error saving final result: {e}")
           
    except Exception as e:
        print(f"Error during search: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
