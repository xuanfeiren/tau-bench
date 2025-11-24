import wandb
api = wandb.Api()

# First, list all runs in the project to see what's available
project_path = "xuanfeiren-university-of-wisconsin-madison/debug-DSPy"
print(f"Fetching runs from project: {project_path}\n")

try:
    runs = api.runs(project_path)
    print(f"Found {len(runs)} runs in the project:\n")
    
    for i, run in enumerate(runs):
        print(f"{i+1}. Run Name: {run.name}")
        print(f"   Run ID: {run.id}")
        print(f"   State: {run.state}")
        print(f"   Created: {run.created_at}")
        print()
    
    # If runs exist, access the first one
    if len(runs) > 0:
        first_run = runs[0]
        print(f"\nAccessing first run: {first_run.name}")
        history = first_run.history()
        print(f"\nAvailable columns in run history:")
        print(list(history.columns))
        
        # Show first few rows
        print(f"\nFirst 5 rows of data:")
        print(history.head())
    else:
        print("No runs found. You may need to run the GEPA optimization first.")
        
except Exception as e:
    print(f"Error: {e}")
    print("\nPossible reasons:")
    print("1. The project 'debug-DSPy' doesn't exist yet")
    print("2. No runs have been logged to this project")
    print("3. You need to run: python my_processing_agents/dspy_opt.py") 