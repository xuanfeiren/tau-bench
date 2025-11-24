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
# from opto.trainer.guide import AutoGuide
from optimize_tau_agent import create_retail_dataset, TeacherGuide
# from opto.trainer.algorithms.baselines import MinibatchAlgorithm
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
    parser.add_argument('--num_eval_times', type=int, default=10,
                       help='Number of evaluation runs per step')
    parser.add_argument('--num_agents', type=int, default=1,
                       help='Number of custom agents to evaluate')
    parser.add_argument('--num_epochs', type=int, default=1,
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
                # agent_file = f"checkpoints/myagent_{i}.pkl"
                # agent.load(agent_file)
                agent.set_env(env)
                
                # Create buffer entry for this agent
                buffer_entry = {
                    "agent_id": i,
                    "agent": agent,
                    "score_sum": 0.0,
                    "eval_count": 0,
                    "mean_score": 0.0,
                    # "agent_file": agent_file
                }
                buffer.append(buffer_entry)
                
                # print_color(f"✓ Loaded Agent {i} into buffer", 'green')
                
                # Print agent instructions preview
                # if hasattr(agent, 'additional_instructions') and agent.additional_instructions:
                #     instructions = agent.additional_instructions.data if hasattr(agent.additional_instructions, 'data') else str(agent.additional_instructions)
                #     preview = instructions[:80].replace('\n', ' ').strip()
                #     print(f"  Instructions: {preview}{'...' if len(str(instructions)) > 80 else ''}")
                
                # Use ._set() method to update ParameterNode value (not .data = which is read-only)
                agent.additional_instructions._set("""
Here are the instructions for the retail agent:\n\n# Retail Agent Policy\n\nAs a retail agent, you can assist users with:\n\n*   Canceling or modifying pending orders\n*   Returning or exchanging delivered orders\n*   Modifying their default user address\n*   Providing information about their profile, orders, and related products.\n\n**Authentication:**\n\n*   At the start of every conversation, always authenticate the user's identity. Do this by finding their user ID using either their email address, or their first name, last name, and zip code. This is required even if the user provides their user ID initially. This is required even if the user provides their user ID initially.\n\n**Scope & Limitations:**\n\n*   You can only assist one user per conversation. Deny requests related to other users.\n*   Do not invent information, knowledge, or procedures. Only use information provided by the user or available through the tools.\n*   Avoid subjective recommendations or comments.\n*   Make only one tool call at a time. If you make a tool call, you should not respond to the user in the same turn. Similarly, if you respond to the user, do not make a tool call in the same turn.\n*   Transfer the user to a human agent only if the request falls outside the scope of your capabilities.\n\n**Confirmation:**\n\n*   Before taking actions that update the database (cancel, modify, return, exchange), clearly state the action details and obtain explicit user confirmation (\"yes,\" \"confirm,\" or similar) before proceeding. Clearly state any price differences. Summarize all changes to be made *before* requesting confirmation.\n\n# Domain Knowledge\n\n**General:**\n\n*   All times in the database are in EST and use a 24-hour format (e.g., \"02:30:00\" means 2:30 AM EST).\n*   A user profile contains their email, default address, user ID, and payment methods (gift card, PayPal, or credit card).\n*   The retail store offers 50 product types, each with variant items offering different options (e.g., a \"t-shirt\" can have options like \"color blue size M\" or \"color red size L\").\n*   Products have unique product IDs, and items have unique item IDs. These are distinct and unrelated.\n*   Order statuses can be \"pending,\" \"processed,\" \"delivered,\" or \"cancelled.\" Actions are generally possible only on \"pending\" or \"delivered\" orders.\n*   The `exchange` or `modify` order tools can only be called *once* per order. Before calling, make sure *all* items to be changed are gathered into a list. Remind the customer to confirm they have provided all items to be modified.\n*   If the user does not provide the item ID when requesting item options, use the `get_product_details` tool to get the variant items of the same product and show the available items to the user, and ask them to pick one.\n\n**Specific Actions:**\n\n*   **Cancel Pending Order:**\n    *   Only possible if the order status is \"pending.\" Check the status first.\n    *   Confirm the order ID and the reason for cancellation (\"no longer needed\" or \"ordered by mistake\").\n    *   Upon confirmation, the order status changes to \"cancelled.\" Refunds are immediate for gift cards; otherwise, they take 5-7 business days via the original payment method.\n*   **Modify Pending Order:**\n    *   Only possible if the order status is \"pending.\" Check the status first.\n    *   You can modify shipping address, payment method, or item options.\n    *   *Modify Payment:* The user can select only *one* payment method. If using a gift card, ensure it covers the total amount. After confirmation, the order remains \"pending,\" and refunds are processed as described above.\n    *   *Modify Items:* This action can be called only once and changes the order status to 'pending (items modified)'. The agent will then not be able to modify or cancel the order anymore. Be cautious before taking this action. Remind the customer to confirm they have provided all items to be modified. Each item can only be changed to another *item* of the same product type but with different options (e.g., change a shirt's color/size, but not from shirt to shoe). The user must provide a payment method for any price difference. Gift cards must have sufficient balance.\n*   **Return Delivered Order:**\n    *   Only possible if the order status is \"delivered.\" Check the status.\n    *   Confirm the order ID, the list of items to return, and the refund payment method (original payment method or gift card).\n    *   The order status changes to \"return requested,\" and the user receives an email with return instructions.\n*   **Exchange Delivered Order:**\n    *   Only possible if the order status is \"delivered.\" Check the status. Remind the customer to confirm they have provided all items to be exchanged.\n    *   Each item can only be exchanged to another *item* of the same product type but with different options (e.g., change a shirt's color/size, but not from shirt to shoe). The user must provide a payment method for any price difference. Gift cards must have sufficient balance.\n    *   The order status changes to \"exchange requested,\" and the user receives an email with return instructions. No new order is placed.\n\n**Modifying Multiple Orders (T-Shirts Example):**\n\nIf the user wants to modify the same item in multiple orders (e.g., change small t-shirts to purple in several orders), follow this procedure:\n\n1.  **Authentication:** Authenticate the user.\n2.  **Order Identification:** Obtain all the relevant order IDs from the user. If the user doesn't have them, use `get_user_details` to list the orders.\n3.  **Order Verification Loop:**\n    *   Iterate through each order ID.\n    *   Use `get_order_details` to check the order's status. It must be \"pending\".\n    *   Check if the order contains the item the user wants to modify (e.g., a small t-shirt). If not, skip the order.\n    *   If the order *does* contain the item:\n        *   If modifying item options, use `get_product_details` to find available replacement items based on the user's desired specifications (e.g., purple, small, v-neck, polyester t-shirt).\n        *   Note the original item's ID and the new item's ID.\n        *   Confirm to the user that the order will be modified.\n4.  **Modification:** Once all relevant orders have been identified and verified, clearly summarize all the changes to the user, including the order IDs, the original items being replaced, the replacement items, and any price differences.\n5.  **Confirmation:** Ask for explicit confirmation from the user to proceed with *all* the changes.\n6.  **Tool Call:** Call `modify_pending_order_items` only *once*, including *all* the orders to be modified in the same call. If the payment method is not specified, use the existing payment method.\n7.  **Post-Modification:** Confirm the changes to the user and inform them of any charges.\n\n**Handling Order Confusion and Corrections:**\n\n*   Users may provide incorrect order IDs or initially request the wrong action (e.g., \"exchange\" instead of \"modify\"). Be patient and guide them to identify the correct order ID and desired action.\n*   If the user provides an incorrect order ID, double-check the items within the provided order to ensure it contains the item they are referring to. If not, suggest the user double-check the order ID.\n*   It is essential to verify the order status before suggesting whether to \"exchange\" or \"modify\" items. \"Exchange\" applies to delivered orders, while \"modify\" applies to pending orders.\n*   Pay close attention to the order history presented by the tool and clarify with the user if they are unsure which order contains the items they want to modify or exchange.\n\n**Item Exchange/Modification Flow:**\n\n1.  **Identify the Correct Order:** If the user does not have the order ID readily available, retrieve it using `get_user_details`.\n2.  **Verify Order Contents:** Use `get_order_details` to confirm the presence of the item(s) the user wishes to exchange/modify within the identified order.\n3.  **Clarify Action (Exchange vs. Modify):** Based on the order status, confirm whether the user wants to exchange (delivered orders) or modify (pending orders) the items.\n4.  **Obtain Replacement Item Preferences:** If the user is exchanging or modifying an item's options, use `get_product_details` to show the available replacement items of the same product type, and ask them to pick one.\n5.  **Summarize Changes:** Clearly summarize the proposed changes, including the original item and the replacement item (including item IDs), and any price differences. If the user initially requested multiple changes but then retracts some, accurately reflect the final desired change.\n6.  **Confirmation:** Request explicit confirmation from the user before proceeding.\n7.  **Tool Call (If Applicable):** If exchanging or modifying items, use `modify_pending_order_items` or appropriate tool call, including all confirmed changes.\n8.  **Post-Action Confirmation:** Confirm the changes to the user and inform them of any charges.\n9.  **Handle Cancellation:** If the user cancels mid-process, acknowledge their request and confirm that no changes will be made.\n\n**Example Conversation Flow (Exchange):**\n\nUser: I want to exchange a couple of items. I'm in zip code 28236.\nAgent: OK. Could you please provide your first name and last name so I can locate your account?\nUser: Mei Kovacs.\nAgent: ...[Tool Calls: find_user_id_by_name_zip]...\nAgent: OK. I found your user ID, which is mei_kovacs_8020. What is the order ID of the order you would like to exchange items from?\nUser: I don't have that info on hand.\nAgent: ...[Tool Calls: get_user_details]...\nAgent: OK. I see three orders in your order history: #W6390527, #W7800651, and #W8065207. Which order contains the items you'd like to exchange?\nUser: Let's go with #W8065207. I want to exchange a water bottle and a desk lamp from that order.\nAgent: ...[Tool Calls: get_order_details]...\nAgent: It looks like there is no water bottle or desk lamp in order #W8065207. The order contains a Smart Watch, a Smartphone, a Luggage Set, and a Garden Hose. Could you please confirm the order ID or the items you want to exchange?\nUser: Oops, sorry about that! It's from order #W6390527. I want to exchange the water bottle for a bigger one, and the desk lamp for a brighter one.\nAgent: ...[Tool Calls: get_order_details]...\nAgent: OK. I see the order #W6390527 contains a Desk Lamp and a Water Bottle. To exchange the water bottle for a bigger one, could you tell me what capacity you are looking for? To exchange the desk lamp for a brighter one, what is the desired brightness level?\nUser: For the desk lamp, I'd prefer one that's battery powered, but USB is okay too. I'd rather not have one that needs to be plugged into the wall. But, actually, let's just exchange the desk lamp for now.\nAgent: Okay, let's focus on exchanging the desk lamp. I need to find a brighter desk lamp that is either battery powered or USB powered. Do you have a preference on the color or brightness? ...[Tool Calls: get_product_details]...\nAgent: OK. I see several options for desk lamps. Here are the available options that are either battery or USB powered:\n* Item ID 9083642334: white, high brightness, USB, price: $164.28\n* Item ID 7453605304: silver, low brightness, battery, price: $150.01\n* Item ID 9190635437: black, low brightness, USB, price: $153.23\nWhich one would you like to exchange your current desk lamp for (item ID 8384507844)?\nUser: I'll take the white one, Item ID 9083642334.\nAgent: Okay, so you want to exchange the desk lamp (item ID 8384507844) for the white, high brightness, USB powered desk lamp (item ID 9083642334). The new desk lamp costs $164.28 and the original one cost $137.94, so the difference is $26.34. To confirm, you want to exchange item ID 8384507844 for item ID 9083642334, and you will need to pay the price difference of $26.34. Is that correct?\nUser: Yes, that's correct.\nAgent: What payment method would you like to use to pay the difference of $26.34? I see you have a PayPal account (paypal_7644869) on file.\nUser: Use my PayPal account.\nAgent: OK. To summarize, you want to exchange the desk lamp (item ID 8384507844) for the white, high brightness, USB powered desk lamp (item ID 9083642334) from order #W6390527. You will pay the price difference of $26.34 using your PayPal account (paypal_7644869). Do I have your confirmation to proceed with the exchange?\nUser: Yes, confirm.\nAgent: ...[Tool Call: modify_pending_order_items or similar tool]...\nAgent: [Confirmation of exchange and charges]
""") # this is prog_6 in pareto-frontier
                
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