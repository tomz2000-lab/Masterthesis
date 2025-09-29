#!/bin/bash
#SBATCH --job-name=resource_allocation_game
#SBATCH --partition=standard
#SBATCH --nodelist=jn117,jn118,jn120  # Target known working nodes
#SBATCH --gres=gpu:L40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=resource_allocation_%j.out
#SBATCH --error=resource_allocation_%j.err

echo "=== Resource Allocation Game - Targeted Node Run ==="
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
export TORCHDYNAMO_DISABLE=1  # Critical: Prevent hanging during model loading
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Set up configuration
cp /home/s391129/negotiation_platform/configs/model_configs_gpu.yaml /home/s391129/negotiation_platform/configs/model_configs.yaml 2>/dev/null || true

# Clear cache
find /home/s391129/negotiation_platform -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Verify torch scope fix
echo "=== Verifying Torch Scope Fix ==="
if grep -q "torch._dynamo as torch_dynamo" /home/s391129/negotiation_platform/models/hf_model_wrapper.py; then
    echo "✅ Torch scope fix is present"
else
    echo "❌ WARNING: Torch scope fix not found"
fi

# Verify resource allocation game prompt update
echo "=== Verifying Resource Allocation Game Updates ==="
if grep -q "Generate role-specific system prompt" /home/s391129/negotiation_platform/games/resource_exchange_car_structure.py; then
    echo "✅ Role-based prompts are implemented"
else
    echo "❌ WARNING: Role-based prompts not found"
fi

cd /home/s391129

# Run resource allocation game
echo "=== Running Resource Allocation Game - 10 Iterations ==="
echo "Game: Development Team vs Marketing Team resource negotiation"
echo "Utility Functions: Development=12x+3y+ε, Marketing=3x+12y+i"

ITERATIONS=10
SUCCESS_COUNT=0

for i in $(seq 1 $ITERATIONS); do
    echo ""
    echo "=== Iteration $i/$ITERATIONS ==="
    echo "$(date): Starting resource allocation negotiation iteration $i"
    
    echo "🔧 About to run Python command..."
    python -m negotiation_platform.main --quick --game resource_allocation --log-level INFO 2>&1
    EXIT_CODE=$?
    echo "🔧 Python command exit code: $EXIT_CODE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Iteration $i completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Iteration $i failed"
    fi
    
    echo "$(date): Completed iteration $i"
done

echo ""
echo "=== Resource Allocation Game Batch Complete ==="
echo "Successful iterations: $SUCCESS_COUNT/$ITERATIONS"
echo "Check output logs for negotiation results and final utilities"