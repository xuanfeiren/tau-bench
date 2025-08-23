# best-candidate-identification.py
# Runner for Best Candidate Identification (BAI) algorithms
#
# Available algorithms:
# - even: EvenlySplitAlgorithm - Evenly split evaluation budget across all candidates
# - ucb: UCBAlgorithm - Upper Confidence Bound exploration-exploitation
# - llm_model: LLMModel - LLM-based candidate selection (tabular model)
# - llm_regression: LLMRegressionModel - LLM regression analysis with pattern learning
# - llm_generator: LLMGenerator - LLM generation + regression (2 LLM calls)
# - llm_onecall: LLMRegressThenGenerate_onecall - Combined generation + regression (1 LLM call)
# - llm_simple: LLMSimpleGenerator - Regression + optional single generation (1 LLM call)
# - llm_thompson: LLMThompsonSampling - Thompson sampling with multiple score predictions
# - llm_ts_onecall: LLMTS_onecall - Thompson sampling with generation + prediction (1 LLM call)
#
# Usage examples:
# python run-bai-algos.py --bai_algo llm_onecall --enable_estimate_scores --num_epochs 15
# python run-bai-algos.py --bai_algo ucb --num_agents 5 --num_epochs 10
# python run-bai-algos.py --bai_algo llm_simple --enable_estimate_scores --enable_using_regressor
# python run-bai-algos.py --bai_algo llm_simple  # Simple mode without regression analysis
# python run-bai-algos.py --bai_algo llm_thompson --enable_estimate_scores  # Thompson sampling approach
# python run-bai-algos.py --bai_algo llm_ts_onecall --enable_estimate_scores --temperature 0.5  # Thompson sampling one call
from agents.tool_calling_agent import ToolCallingAgent_v2 as ToolCallingAgent
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
import litellm 
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from tau_bench.envs.user import UserStrategy
from tau_bench.retry_utils import auto_retry_with_exponential_backoff

import opto 
from opto import trace
from opto.optimizers import OptoPrime 
from opto.trace.nodes import GRAPH
from opto.trace.modules import Module 

# Copyright Sierra

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
from tau_agent_opt import create_retail_dataset,TeacherGuide
from opto.trainer.algorithms.baselines import MinibatchAlgorithm
from opto.optimizers.utils import print_color
from opto.trainer.evaluators import evaluate
import numpy as np
from opto.trainer.algorithms.BAI_algorithms import (
    EvenlySplitAlgorithm, 
    UCBAlgorithm, 
    LLMModel, 
    LLMRegressionModel, 
    LLMGenerator,
    LLMRegressThenGenerate_onecall,
    LLMSimpleGenerator,
    LLMThompsonSampling,
    LLMTS_onecall
)
provider = "gemini"


def evaluate_agent(agent, guide, dataset,min_score=0,num_threads=20,num_eval_times=5):
    """Evaluate an agent."""
    eval_scores = evaluate(agent,guide, dataset['inputs'],dataset['infos'],
                                        min_score=min_score,
                                        num_threads=num_threads,
                                        num_samples=num_eval_times,
                                        description=f"Evaluating agent")
    # Create table with explicit column names
    if eval_scores.ndim >1:
        columns = [f'Eval_{i+1}' for i in range(eval_scores.shape[1])]
        all_valid_scores = [score for row in eval_scores for score in row if score is not None]
    else:
        all_valid_scores = [score for score in eval_scores if score is not None]
    test_score = np.mean(all_valid_scores) if all_valid_scores else 0
    return test_score


def main():
    """Main function for LearnFromSuccessAlgorithm training."""
    parser = argparse.ArgumentParser(description='Best Candidate Identification')
    
    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=10,
                       help='Number of training samples')
    parser.add_argument('--num_validate_samples', type=int, default=10,
                       help='Number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=10,
                       help='Number of test samples')
    
    # Training parameters
    parser.add_argument('--num_threads', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--eval_frequency', type=int, default=1,
                       help='How often to run evaluation')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='How often to log results')
    parser.add_argument('--save_frequency', type=int, default=None,
                       help='How often to save the agent')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--num_agents', type=int, default=3,
                       help='Number of agents to load from checkpoints')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    parser.add_argument('--run_name', type=str, default='debug',
                       help='Name of the run')

    # Algorithm choice
    parser.add_argument('--bai_algo', type=str, default='llm_generator', 
                        choices=['even', 'ucb', 'llm_model', 'llm_regression', 'llm_generator', 'llm_onecall', 'llm_simple', 'llm_thompson', 'llm_ts_onecall'],
                        help='Which BAI algorithm to run: '
                             'even (EvenlySplitAlgorithm), '
                             'ucb (UCBAlgorithm), '
                             'llm_model (LLMModel), '
                             'llm_regression (LLMRegressionModel), '
                             'llm_generator (LLMGenerator), '
                             'llm_onecall (LLMRegressThenGenerate_onecall), '
                             'llm_simple (LLMSimpleGenerator), '
                             'llm_thompson (LLMThompsonSampling), '
                             'llm_ts_onecall (LLMTS_onecall)')
    
    # LLM-based algorithm specific parameters
    parser.add_argument('--enable_estimate_scores', action='store_true', default=False,
                        help='Enable score estimation in LLM-based algorithms (applies to llm_regression, llm_generator, llm_onecall, llm_simple)')
    parser.add_argument('--enable_using_regressor', action='store_true', default=False,
                        help='Enable regression analysis in LLMSimpleGenerator (only applies to llm_simple)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for LLM sampling in LLMThompsonSampling')
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
            task_ids=list(range(max(args.num_train_samples, args.num_validate_samples, args.num_test_samples))),
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
        train_dataset = create_retail_dataset(env, num_tasks=args.num_train_samples)
        validate_dataset = create_retail_dataset(env, num_tasks=args.num_validate_samples)
        test_dataset = create_retail_dataset(env, num_tasks=args.num_test_samples)
        
        print(f"Training samples: {len(train_dataset['inputs'])}")
        print(f"Validation samples: {len(validate_dataset['inputs'])}")
        print(f"Test samples: {len(test_dataset['inputs'])}")
        
        # Initialize agent
        print(f"Initializing agent with model: {config.model}")
        base_agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider
        )
        base_agent.set_env(env)
        guide = TeacherGuide(env, config)
        logger = WandbLogger(project="tau-bench-best-candidate-identification", verbose=True, name=args.run_name)
        
        # Load 10 agents from checkpoints folder
        agents = []
        for i in range(args.num_agents):
            agent = ToolCallingAgent(
                tools_info=env.tools_info,
                wiki=env.wiki,
                model=config.model,
                provider=config.model_provider
            )
            agent.load(f"checkpoints/agent_{i}.pkl")
            agent.set_env(env)
            agents.append(agent)
        # Construct a list of update dicts using parameter values from loaded agents
        update_dicts = []
        for agent in agents:
            update_dict = {}
            # update_dict[base_agent.tools_info] = agent.tools_info.data
            update_dict[base_agent.additional_instructions] = agent.additional_instructions.data
            update_dicts.append(update_dict)
        print_color(f"Constructed {len(update_dicts)} update dicts", 'green')
        
        # Initialize algorithm
        print_color(f"Initializing BAI algorithm: {args.bai_algo}", 'blue')
        if args.bai_algo == 'even':
            print_color("  Strategy: Evenly split evaluation budget across all candidates", 'cyan')
            algo = EvenlySplitAlgorithm(base_agent, num_threads=args.num_threads, logger=logger, update_dicts=update_dicts)
        elif args.bai_algo == 'ucb':
            print_color("  Strategy: Upper Confidence Bound exploration-exploitation", 'cyan')
            algo = UCBAlgorithm(base_agent, num_threads=args.num_threads, logger=logger, update_dicts=update_dicts)
        elif args.bai_algo == 'llm_model':
            print_color("  Strategy: LLM-based candidate selection (tabular model)", 'cyan')
            algo = LLMModel(base_agent, num_threads=args.num_threads, logger=logger, update_dicts=update_dicts)
        elif args.bai_algo == 'llm_regression':
            print_color(f"  Strategy: LLM regression analysis (score_estimation={args.enable_estimate_scores})", 'cyan')
            algo = LLMRegressionModel(base_agent, num_threads=args.num_threads, logger=logger, update_dicts=update_dicts, enable_estimate_scores=args.enable_estimate_scores)
        elif args.bai_algo == 'llm_generator':
            print_color(f"  Strategy: LLM generation + regression (2 calls, score_estimation={args.enable_estimate_scores})", 'cyan')
            algo = LLMGenerator(base_agent, num_threads=args.num_threads, logger=logger, update_dicts=update_dicts, enable_estimate_scores=args.enable_estimate_scores)
        elif args.bai_algo == 'llm_onecall':
            print_color(f"  Strategy: Combined generation + regression (1 call, score_estimation={args.enable_estimate_scores})", 'cyan')
            algo = LLMRegressThenGenerate_onecall(base_agent, num_threads=args.num_threads, logger=logger, update_dicts=update_dicts, enable_estimate_scores=args.enable_estimate_scores)
        elif args.bai_algo == 'llm_simple':
            print_color(f"  Strategy: Regression + optional single generation (1 call, score_estimation={args.enable_estimate_scores}, regressor={args.enable_using_regressor})", 'cyan')
            algo = LLMSimpleGenerator(base_agent, num_threads=args.num_threads, logger=logger, update_dicts=update_dicts, enable_estimate_scores=args.enable_estimate_scores, enable_using_regressor=args.enable_using_regressor)
        elif args.bai_algo == 'llm_thompson':
            print_color(f"  Strategy: LLM Thompson Sampling (multiple score predictions, score_estimation={args.enable_estimate_scores})", 'cyan')
            algo = LLMThompsonSampling(base_agent, num_threads=args.num_threads, logger=logger, update_dicts=update_dicts, enable_estimate_scores=args.enable_estimate_scores, temperature=args.temperature)
        elif args.bai_algo == 'llm_ts_onecall':
            print_color(f"  Strategy: LLM Thompson Sampling with generation + prediction (1 call, score_estimation={args.enable_estimate_scores})", 'cyan')
            algo = LLMTS_onecall(base_agent, num_threads=args.num_threads, logger=logger, update_dicts=update_dicts, enable_estimate_scores=args.enable_estimate_scores, temperature=args.temperature)
        else:
            raise ValueError(f"Unknown BAI algorithm: {args.bai_algo}")

        # Run a simple loop that steps through epochs and logs best candidate periodically
        algo.train(guide, validate_dataset, test_dataset, num_threads=args.num_threads, num_epochs=args.num_epochs, eval_frequency=args.eval_frequency)

        print("Completed Best Candidate Identification run.")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 


