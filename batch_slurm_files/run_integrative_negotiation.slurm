#!/bin/bash
#SBATCH --job-name=integrative_negotiation
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=integrative_negotiation_%j.out
#SBATCH --error=integrative_negotiation_%j.err

echo "=== Integrative Negotiations Game - Targeted Node Run ==="
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

# Verify integrative negotiations game updates
echo "=== Verifying Integrative Negotiations Game Updates ==="
if grep -q "IntegrativeNegotiationsGame" /home/s391129/negotiation_platform/games/integrative_negotiation.py 2>/dev/null; then
    echo "✅ Integrative negotiations game found"
else
    echo "❌ WARNING: Integrative negotiations game not found"
fi

cd /home/s391129

echo "=== Running Integrative Negotiations Game - 10 Iterations ==="
echo "Game: IT Team vs Marketing Team office space negotiation"
echo "Issues: Server Room Size, Meeting Access, Cleaning Responsibility, Branding Visibility"

ITERATIONS=300
SUCCESS_COUNT=0

for i in $(seq 1 $ITERATIONS); do
    echo ""
    echo "=== Iteration $i/$ITERATIONS ==="
    echo "$(date): Starting integrative negotiations iteration $i"
    
    echo "🔧 About to run Python command..."
    python -m negotiation_platform.main --quick --game integrative_negotiations --models model_a model_b --log-level INFO
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
echo "=== Integrative Negotiations Game Batch Complete ==="
echo "Successful iterations: $SUCCESS_COUNT/$ITERATIONS"
echo "Check output logs for negotiation results and final utilities"