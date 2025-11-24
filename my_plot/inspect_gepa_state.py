#!/usr/bin/env python3
"""
Inspect GEPA state file to find total metric calls
"""

import pickle
import json

state_file = "results/gepa/gepa_state.bin"

print(f"Loading GEPA state from: {state_file}\n")

try:
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
    
    print("✅ Successfully loaded GEPA state\n")
    
    # Explore the state structure
    print("📦 State Contents:")
    if isinstance(state, dict):
        print(f"  Type: Dictionary with {len(state)} keys")
        print(f"  Keys: {list(state.keys())}\n")
        
        # Look for metric calls information
        for key in state.keys():
            value = state[key]
            print(f"  {key}:")
            print(f"    Type: {type(value).__name__}")
            
            if key in ['total_metric_calls', 'metric_calls', 'num_evaluations', 'budget_used']:
                print(f"    ⭐ VALUE: {value}")
            elif isinstance(value, (int, float, str, bool)):
                print(f"    Value: {value}")
            elif isinstance(value, (list, tuple)):
                print(f"    Length: {len(value)}")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0]).__name__}")
            elif isinstance(value, dict):
                print(f"    Dict keys: {list(value.keys())[:10]}")
            print()
    else:
        print(f"  Type: {type(state).__name__}")
        print(f"  Attributes: {dir(state)}\n")
        
        # Try to access common attributes
        for attr in ['total_metric_calls', 'num_evaluations', 'iteration', 
                     'frontier', 'candidates', 'history']:
            if hasattr(state, attr):
                value = getattr(state, attr)
                print(f"  {attr}: {value if not isinstance(value, (list, dict)) else f'{type(value).__name__} (len={len(value)})'}")
    
    # Try to estimate from frontier size
    if isinstance(state, dict) and 'frontier' in state:
        frontier = state['frontier']
        print(f"\n📊 Pareto Frontier Analysis:")
        print(f"  Frontier size: {len(frontier) if hasattr(frontier, '__len__') else 'N/A'}")
    
except FileNotFoundError:
    print(f"❌ Error: File not found: {state_file}")
except Exception as e:
    print(f"❌ Error loading state: {e}")
    print(f"   Error type: {type(e).__name__}")
    
print("\n" + "="*60)
print("Alternative: Check saved JSON outputs")
print("="*60)

# Count evaluations from saved outputs
import os
import glob

output_dir = "results/gepa/generated_best_outputs_valset"
json_files = glob.glob(f"{output_dir}/task_*/*.json")

print(f"\nSaved evaluation outputs: {len(json_files)} files")

# Parse iteration numbers
iterations = set()
programs = set()

for filepath in json_files:
    filename = os.path.basename(filepath)
    # Parse: iter_1798_prog_2.json
    if filename.startswith("iter_"):
        parts = filename.replace(".json", "").split("_")
        if len(parts) >= 4:
            iter_num = int(parts[1])
            prog_id = int(parts[3])
            iterations.add(iter_num)
            programs.add(prog_id)

print(f"Unique iterations seen: {len(iterations)}")
print(f"  Range: {min(iterations)} to {max(iterations)}")
print(f"Unique program IDs: {len(programs)}")
print(f"  IDs: {sorted(programs)}")

print(f"\n💡 Estimation:")
print(f"  - If each program was evaluated on all 10 tasks:")
print(f"    {len(programs)} programs × 10 tasks = {len(programs) * 10} full evaluations")
print(f"  - Plus minibatch evaluations (hard to estimate from files)")
print(f"  - Plus Pareto frontier re-evaluations")
print(f"\n⚠️  Total metric calls likely reached the budget limit (~2000)")

