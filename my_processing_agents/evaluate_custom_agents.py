#!/usr/bin/env python3
"""
Script to load and evaluate custom agents (myagent_0 to myagent_9) on the test set.
This script loads the agents and evaluates their performance without training.
"""

from agents.tool_calling_agent import ToolCallingAgent_v2 as ToolCallingAgent
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
import litellm 
from tau_bench.envs.user import UserStrategy
from tau_bench.retry_utils import auto_retry_with_exponential_backoff

import opto 
from opto import trace
from opto.optimizers import OptoPrime 
from opto.trace.nodes import GRAPH
from opto.trace.modules import Module 

import json
from litellm import completion
from typing import List, Optional, Dict, Any
import argparse

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

from tau_bench.model_utils.model.utils import trim_conversation_messages
from opto.trainer.loggers import WandbLogger, DefaultLogger
from opto.trainer.guide import AutoGuide
from tau_agent_opt import create_retail_dataset, TeacherGuide
from opto.trainer.algorithms.baselines import MinibatchAlgorithm
from opto.optimizers.utils import print_color
from opto.trainer.evaluators import evaluate
import numpy as np

provider = "gemini"

def evaluate_agent(agent, guide, dataset, min_score=0, num_threads=20, num_eval_times=5):
    """Evaluate an agent - copied from run-bai-algos.py"""
    eval_scores = evaluate(agent, guide, dataset['inputs'], dataset['infos'],
                          min_score=min_score,
                          num_threads=num_threads,
                          num_samples=num_eval_times,
                          description=f"Evaluating agent")
    # Create table with explicit column names
    if eval_scores.ndim > 1:
        columns = [f'Eval_{i+1}' for i in range(eval_scores.shape[1])]
        all_valid_scores = [score for row in eval_scores for score in row if score is not None]
    else:
        all_valid_scores = [score for score in eval_scores if score is not None]
    test_score = np.mean(all_valid_scores) if all_valid_scores else 0
    return test_score

def main():
    """Load and evaluate custom agents with buffer-based tracking and WandB logging."""
    parser = argparse.ArgumentParser(description='Evaluate Custom Agents with Buffer Tracking')
    
    # Evaluation parameters
    parser.add_argument('--num_test_samples', type=int, default=10,
                       help='Number of test samples per evaluation')
    parser.add_argument('--num_threads', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--num_eval_times', type=int, default=5,
                       help='Number of evaluation runs per step')
    parser.add_argument('--num_agents', type=int, default=10,
                       help='Number of custom agents to evaluate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of evaluation epochs')
    parser.add_argument('--eval_frequency', type=int, default=1,
                       help='How often to log results')
    parser.add_argument('--run_name', type=str, default='custom_agents_eval',
                       help='Name for WandB run')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    
    args = parser.parse_args()
    
    try:
        # Create configuration - same as run-bai-algos.py
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
        
        # Initialize environment - same as run-bai-algos.py
        print(f"Initializing retail environment with user strategy: {config.user_strategy}")
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
            task_index=0
        )
        
        # Create test dataset
        print("Creating test dataset...")
        test_dataset = create_retail_dataset(env, num_tasks=args.num_test_samples)
        print(f"Test samples: {len(test_dataset['inputs'])}")
        
        # Initialize guide for evaluation
        guide = TeacherGuide(env, config)
        
        # Initialize WandB logger
        logger = WandbLogger(project="tau-bench-custom-agents-evaluation", verbose=True, name=args.run_name)
        
        # Initialize buffer for tracking agent performance
        print(f"\nInitializing buffer for {args.num_agents} custom agents...")
        buffer = []
        
        # Load all custom agents and initialize buffer entries
        for i in range(args.num_agents):
            try:
                # Create agent - same pattern as run-bai-algos.py
                agent = ToolCallingAgent(
                    tools_info=env.tools_info,
                    wiki=env.wiki,
                    model=config.model,
                    provider=config.model_provider,
                    temperature=config.temperature
                )
                
                # Load custom agent
                agent_file = f"checkpoints/myagent_{i}.pkl"
                agent.load(agent_file)
                agent.set_env(env)
                
                # Create buffer entry for this agent
                buffer_entry = {
                    "agent_id": i,
                    "agent": agent,
                    "score_sum": 0.0,
                    "eval_count": 0,
                    "mean_score": 0.0,
                    "agent_file": agent_file
                }
                buffer.append(buffer_entry)
                
                print_color(f"✓ Loaded Agent {i} into buffer", 'green')
                
                # Print agent instructions preview
                if hasattr(agent, 'additional_instructions') and agent.additional_instructions:
                    instructions = agent.additional_instructions.data if hasattr(agent.additional_instructions, 'data') else str(agent.additional_instructions)
                    preview = instructions[:80].replace('\n', ' ').strip()
                    print(f"  Instructions: {preview}{'...' if len(str(instructions)) > 80 else ''}")
                
            except FileNotFoundError:
                print_color(f"✗ Agent {i}: File myagent_{i}.pkl not found", 'red')
            except Exception as e:
                print_color(f"✗ Agent {i}: Error loading - {str(e)}", 'red')
        
        print(f"\nSuccessfully loaded {len(buffer)} agents into buffer")
        
        # Periodic evaluation loop - similar to BAI algorithms
        print("\n" + "="*80)
        print("STARTING PERIODIC EVALUATION")
        print("="*80)
        
        for epoch in range(args.num_epochs):
            print_color(f"\nEpoch {epoch+1}/{args.num_epochs}", "blue")
            
            # Evaluate each agent in the buffer
            for buffer_entry in buffer:
                agent_id = buffer_entry["agent_id"]
                agent = buffer_entry["agent"]
                
                print(f"  Evaluating Agent {agent_id}...")
                
                # Evaluate agent
                score = evaluate_agent(
                    agent=agent,
                    guide=guide,
                    dataset=test_dataset,
                    min_score=0,
                    num_threads=args.num_threads,
                    num_eval_times=args.num_eval_times
                )
                
                # Update buffer statistics
                buffer_entry["score_sum"] += score * len(test_dataset['inputs']) * args.num_eval_times
                buffer_entry["eval_count"] += len(test_dataset['inputs']) * args.num_eval_times
                buffer_entry["mean_score"] = buffer_entry["score_sum"] / buffer_entry["eval_count"]
                
                print_color(f"    Agent {agent_id} Score: {score:.4f} (Mean: {buffer_entry['mean_score']:.4f}, Eval Count: {buffer_entry['eval_count']})", 'cyan')
            
            # Log results to WandB at specified frequency
            if epoch % args.eval_frequency == 0:
                print_color(f"\nLogging results for Epoch {epoch+1}:", "green")
                
                # Log individual agent scores
                for buffer_entry in buffer:
                    agent_id = buffer_entry["agent_id"]
                    mean_score = buffer_entry["mean_score"]
                    eval_count = buffer_entry["eval_count"]
                    
                    # Log to WandB
                    logger.log(f"Agent_{agent_id}_mean_score", mean_score, epoch+1, color='blue')
                    logger.log(f"Agent_{agent_id}_eval_count", eval_count, epoch+1, color='blue')
                    
                    print(f"  Agent {agent_id}: Mean={mean_score:.4f}, Eval Count={eval_count}")
                
                # Log aggregate statistics
                all_scores = [entry["mean_score"] for entry in buffer]
                best_score = max(all_scores)
                worst_score = min(all_scores)
                avg_score = np.mean(all_scores)
                std_score = np.std(all_scores)
                
                # logger.log("Best_Agent_Score", best_score, epoch+1, color='green')
                # logger.log("Worst_Agent_Score", worst_score, epoch+1, color='red')
                # logger.log("Average_Agent_Score", avg_score, epoch+1, color='yellow')
                # logger.log("Score_Std_Deviation", std_score, epoch+1, color='yellow')
                
                # Find and log best agent
                best_agent_entry = max(buffer, key=lambda x: x['mean_score'])
                best_agent_id = best_agent_entry["agent_id"]
                logger.log("Best_Agent_ID", best_agent_id, epoch+1, color='green')
                
                print_color(f"  Best Agent: {best_agent_id} (Score: {best_score:.4f})", 'green')
                print_color(f"  Average Score: {avg_score:.4f} ± {std_score:.4f}", 'yellow')
        
        # Final summary results
        print("\n" + "="*80)
        print("FINAL EVALUATION SUMMARY")
        print("="*80)
        
        if buffer:
            # Sort agents by final mean score
            sorted_buffer = sorted(buffer, key=lambda x: x['mean_score'], reverse=True)
            
            print(f"\nSuccessfully evaluated {len(buffer)} agents")
            print(f"Total epochs: {args.num_epochs}")
            print(f"Test dataset size per evaluation: {len(test_dataset['inputs'])} samples")
            print(f"Evaluation runs per step: {args.num_eval_times}")
            print(f"Number of threads: {args.num_threads}")
            
            print("\n📈 FINAL AGENT RANKINGS (Best to Worst):")
            print("-" * 60)
            for rank, entry in enumerate(sorted_buffer, 1):
                agent_id = entry["agent_id"]
                mean_score = entry["mean_score"]
                eval_count = entry["eval_count"]
                print(f"{rank:2d}. Agent {agent_id}: {mean_score:.4f} (evaluated {eval_count} times)")
            
            # Final statistics
            final_scores = [entry["mean_score"] for entry in buffer]
            print(f"\n📊 FINAL SCORE STATISTICS:")
            print(f"  Best Score:    {max(final_scores):.4f} (Agent {sorted_buffer[0]['agent_id']})")
            print(f"  Worst Score:   {min(final_scores):.4f} (Agent {sorted_buffer[-1]['agent_id']})")
            print(f"  Average Score: {np.mean(final_scores):.4f}")
            print(f"  Std Deviation: {np.std(final_scores):.4f}")
            
            # Log final summary to WandB
            logger.log("Final_Best_Score", max(final_scores), args.num_epochs, color='green')
            logger.log("Final_Average_Score", np.mean(final_scores), args.num_epochs, color='yellow')
            logger.log("Final_Score_Range", max(final_scores) - min(final_scores), args.num_epochs, color='cyan')
            
            # Print buffer statistics like BAI algorithms
            print(f"\n📋 BUFFER STATISTICS:")
            print("-" * 60)
            for entry in sorted_buffer:
                agent_id = entry["agent_id"]
                mean_score = entry["mean_score"]
                eval_count = entry["eval_count"]
                total_evaluations = entry["score_sum"] / mean_score if mean_score > 0 else 0
                print_color(f"Agent {agent_id}: Mean score {mean_score:.4f}, eval_count {eval_count}", "blue")
            
        else:
            print("❌ No agents were successfully loaded and evaluated!")
        
        print("\nPeriodic evaluation completed!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 