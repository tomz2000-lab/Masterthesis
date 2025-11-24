#!/bin/bash
#SBATCH --job-name=company_car
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=company_car_%j.out
#SBATCH --error=company_car_%j.err

echo "=== Company Car Game - 10 Iterations ==="
echo "Job ID: $SLURM_JOB_ID, Node: $SLURMD_NODENAME"

# Robust environment detection
if [ -d "/home/s391129/venv" ]; then
    source /home/s391129/venv/bin/activate
elif [ -d "/home/s391129/negotiation_env" ]; then
    source /home/s391129/negotiation_env/bin/activate
else
    echo "❌ No virtual environment found"; exit 1
fi

# Environment and GPU check
echo "=== Environment Check ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Set up configuration
#cp /home/s391129/negotiation_platform/configs/model_configs_gpu.yaml /home/s391129/negotiation_platform/configs/model_configs.yaml 2>/dev/null || true

# Clear cache
find /home/s391129/negotiation_platform -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Verify torch scope fix
echo "=== Verifying Torch Scope Fix ==="
if grep -q "torch._dynamo as torch_dynamo" /home/s391129/negotiation_platform/models/hf_model_wrapper.py; then
    echo "✅ Torch scope fix is present"
else
    echo "❌ WARNING: Torch scope fix not found"
fi

cd /home/s391129

echo "=== Running Company Car Game - 10 Iterations ==="
echo "Game: Employee vs Manager company car negotiation"

ITERATIONS=300
SUCCESS_COUNT=0

for i in $(seq 1 $ITERATIONS); do
    echo ""
    echo "=== Iteration $i/$ITERATIONS ==="
    echo "$(date): Starting company car negotiation iteration $i on node $SLURMD_NODENAME"
    
    python -m negotiation_platform.main --quick --game company_car
    
    if [ $? -eq 0 ]; then
        echo "✅ Iteration $i completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Iteration $i failed"
    fi
    
    echo "$(date): Completed iteration $i"
done

echo ""
echo "=== Company Car Game Batch Complete ==="
echo "Successful iterations: $SUCCESS_COUNT/$ITERATIONS"

if [ $SUCCESS_COUNT -eq $ITERATIONS ]; then
    echo "✅ ALL ITERATIONS SUCCESSFUL: Platform working perfectly on node $SLURMD_NODENAME"
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo "⚠️ PARTIAL SUCCESS: $SUCCESS_COUNT out of $ITERATIONS iterations completed on node $SLURMD_NODENAME"
else
    echo "❌ ALL ITERATIONS FAILED on node $SLURMD_NODENAME"
fi

echo "=== Company Car Run Complete ==="