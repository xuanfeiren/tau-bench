# PowerShell Script to run tau_trainer.py with all different algorithm types
# All other parameters will use their default values

Write-Host "Starting training with all algorithm types..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Array of all algorithm types
$algorithms = @("minibatch", "basicsearch", "beamsearch", "beamsearchhistory", "UCBsearch")

# Loop through each algorithm type
foreach ($algo in $algorithms) {
    Write-Host ""
    Write-Host "Running algorithm: $algo" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    
    # Run the training script with the current algorithm
    python tau_trainer.py --algorithm_type $algo 
    
    # Check if the command was successful
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Algorithm $algo completed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Algorithm $algo failed" -ForegroundColor Red
    }
    
    Write-Host "----------------------------------------" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "All algorithms have been executed!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green 