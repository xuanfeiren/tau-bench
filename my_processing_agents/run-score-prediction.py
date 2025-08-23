# score-prediction-runner.py
# Runner for Score Prediction algorithms
#
# Available algorithms:
# - score_prediction: ScorePrediction - LLM-based score prediction for all candidates
# - score_prediction_half: ScorePrediction_half_buffer - Score prediction with half buffer evaluation
# - embedding_regression: Embedding_Regression - SGD-based linear regression on embeddings
# - embedding_regression_true: Embedding_Regression_with_true_scores - Linear regression using ground truth scores
#
# Usage examples:
# python run-score-prediction.py --bai_algo score_prediction --num_epochs 10
# python run-score-prediction.py --bai_algo score_prediction_half --num_agents 5 --num_epochs 15
# python run-score-prediction.py --bai_algo embedding_regression --num_epochs 20 --learning_rate 0.01
# python run-score-prediction.py --bai_algo embedding_regression_true --num_epochs 5
# python run-score-prediction.py --bai_algo score_prediction --temperature 0.5

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
from tau_agent_opt import create_retail_dataset, TeacherGuide
from opto.trainer.algorithms.baselines import MinibatchAlgorithm
from opto.optimizers.utils import print_color
from opto.trainer.evaluators import evaluate
import numpy as np
from Trace.opto.trainer.algorithms.score_prediction_algorithm import (
    ScorePrediction,
    ScorePrediction_half_buffer,
    Embedding_Regression,
    Embedding_Regression_with_true_scores
)
provider = "gemini"


def evaluate_agent(agent, guide, dataset, min_score=0, num_threads=20, num_eval_times=5):
    """Evaluate an agent."""
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
    """Main function for Score Prediction algorithm training."""
    parser = argparse.ArgumentParser(description='Score Prediction Algorithms')
    
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
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--num_agents', type=int, default=10,
                       help='Number of agents to load from myagent checkpoints (0-9)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    parser.add_argument('--run_name', type=str, default='debug',
                       help='Name of the run')

    # Algorithm choice
    parser.add_argument('--bai_algo', type=str, default='embedding_regression_true', 
                        choices=['score_prediction', 'score_prediction_half', 'embedding_regression', 'embedding_regression_true'],
                        help='Which Score Prediction algorithm to run: '
                             'score_prediction (ScorePrediction), '
                             'score_prediction_half (ScorePrediction_half_buffer), '
                             'embedding_regression (Embedding_Regression), '
                             'embedding_regression_true (Embedding_Regression_with_true_scores)')
    
    # Score Prediction specific parameters
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for LLM sampling in score prediction')
    
    # Embedding Regression specific parameters
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for SGD in embedding regression')
    parser.add_argument('--embedding_model', type=str, default='gemini/embedding-001',
                        help='Embedding model to use for Embedding_Regression',choices=['gemini/text-embedding-004','gemini/embedding-001'])
    
    # Ground truth scores parameter
    parser.add_argument('--ground_truth_scores', type=str, default="0.3140,0.5320,0.3186,0.2644,0.3788,0.1064,0.2780,0.4700,0.1960,0.5540",
                        help='Comma-separated ground truth scores (e.g., "0.1,0.2,0.3"). Default uses observed scores from agents 0-9.')
    
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
        
        # Initialize base agent
        print(f"Initializing base agent with model: {config.model}")
        base_agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider
        )
        base_agent.set_env(env)
        guide = TeacherGuide(env, config)
        logger = WandbLogger(project="tau-bench-score-prediction", verbose=True, name=args.run_name)
        
        # Load agents from myagent checkpoints folder
        agents = []
        for i in range(args.num_agents):
            agent = ToolCallingAgent(
                tools_info=env.tools_info,
                wiki=env.wiki,
                model=config.model,
                provider=config.model_provider
            )
            agent.load(f"checkpoints/myagent_{i}.pkl")
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
        
        # Prepare ground truth scores
        if args.ground_truth_scores:
            try:
                ground_truth_scores = [float(x.strip()) for x in args.ground_truth_scores.split(',')]
                if len(ground_truth_scores) != len(update_dicts):
                    print_color(f"Warning: Provided {len(ground_truth_scores)} ground truth scores but have {len(update_dicts)} agents. Using zeros for missing values.", 'yellow')
                    # Pad with zeros if needed
                    while len(ground_truth_scores) < len(update_dicts):
                        ground_truth_scores.append(0.0)
                    # Trim if too many provided
                    ground_truth_scores = ground_truth_scores[:len(update_dicts)]
            except ValueError as e:
                print_color(f"Error parsing ground truth scores: {e}. Using all zeros.", 'red')
                ground_truth_scores = [0.0] * len(update_dicts)
        else:
            ground_truth_scores = [0.0] * len(update_dicts)
            print_color(f"Using default ground truth scores (all zeros) for {len(update_dicts)} agents", 'yellow')
        
        print_color(f"Ground truth scores: {ground_truth_scores}", 'cyan')
        
        # Initialize algorithm
        print_color(f"Initializing Score Prediction algorithm: {args.bai_algo}", 'blue')
        if args.bai_algo == 'score_prediction':
            print_color("  Strategy: LLM-based score prediction for all candidates", 'cyan')
            algo = ScorePrediction(
                base_agent, 
                num_threads=args.num_threads, 
                logger=logger, 
                update_dicts=update_dicts,
                ground_truth_scores=ground_truth_scores
            )
        elif args.bai_algo == 'score_prediction_half':
            print_color("  Strategy: Score prediction with half buffer evaluation", 'cyan')
            algo = ScorePrediction_half_buffer(
                base_agent, 
                num_threads=args.num_threads, 
                logger=logger, 
                update_dicts=update_dicts,
                ground_truth_scores=ground_truth_scores
            )
        elif args.bai_algo == 'embedding_regression':
            print_color("  Strategy: SGD-based linear regression on embeddings", 'cyan')
            print_color(f"  Embedding model: {args.embedding_model}", 'cyan')
            print_color(f"  Learning rate: {args.learning_rate}", 'cyan')
            algo = Embedding_Regression(
                base_agent, 
                num_threads=args.num_threads, 
                logger=logger, 
                update_dicts=update_dicts,
                ground_truth_scores=ground_truth_scores,
                embedding_model=args.embedding_model,
                learning_rate=args.learning_rate
            )
        elif args.bai_algo == 'embedding_regression_true':
            print_color("  Strategy: Linear regression using ground truth scores (no SGD)", 'cyan')
            print_color(f"  Embedding model: {args.embedding_model}", 'cyan')
            print_color("  Note: Uses least squares with true scores for misspecification analysis", 'cyan')
            algo = Embedding_Regression_with_true_scores(
                base_agent, 
                num_threads=args.num_threads, 
                logger=logger, 
                update_dicts=update_dicts,
                ground_truth_scores=ground_truth_scores,
                embedding_model=args.embedding_model,
                learning_rate=args.learning_rate  # Inherited but not used in this version
            )
        else:
            raise ValueError(f"Unknown Score Prediction algorithm: {args.bai_algo}")

        # Run the score prediction training
        algo.train(
            guide, 
            validate_dataset, 
            test_dataset, 
            num_threads=args.num_threads, 
            num_epochs=args.num_epochs, 
            eval_frequency=args.eval_frequency,
            temperature=args.temperature
        )

        print("Completed Score Prediction run.")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 