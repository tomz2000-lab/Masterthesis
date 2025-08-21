#!/bin/bash
#SBATCH --job-name=negotiation_gpu
#SBATCH --partition=standard      # Good L40 GPU nodes available
#SBATCH --gres=gpu:L40:1         # Request 1 L40 GPU (48GB VRAM)
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # Tasks per node
#SBATCH --cpus-per-task=8        # CPU cores per task
#SBATCH --mem=64G                # Memory request (plenty available)
#SBATCH --time=02:00:00          # Time limit (2 hours)
#SBATCH --output=negotiation_%j.out
#SBATCH --error=negotiation_%j.err

echo "=== Starting GPU Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "========================"

# Activate your virtual environment
source /home/s391129/venv/bin/activate

# Check GPU availability
echo "=== GPU Information ==="
nvidia-smi
echo "========================"

# Check PyTorch GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Install GPU-enabled bitsandbytes if needed
echo "=== Installing GPU bitsandbytes ==="
pip install bitsandbytes --upgrade --no-cache-dir

# Copy GPU config to use
cp /home/s391129/negotiation_platform/configs/model_configs_gpu.yaml /home/s391129/negotiation_platform/configs/model_configs.yaml

# Clear Python cache to ensure updated files are used
echo "=== Clearing Python Cache ==="
find /home/s391129/negotiation_platform -name "*.pyc" -delete
find /home/s391129/negotiation_platform -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "Python cache cleared!"

# Verify our enhanced prompts are in the session manager
echo "=== Verifying Enhanced Prompts ==="
grep -c "🚗\|ACCEPT IMMEDIATELY\|expert negotiator" /home/s391129/negotiation_platform/core/session_manager.py || echo "Enhanced prompts not found!"

# Run your negotiation platform
cd /home/s391129
echo "=== Running Negotiation Platform ==="

# Test all three negotiation games
echo "=== Testing All Three Negotiation Games ==="

# Test 1: Resource Allocation
echo ""
echo "🔧 Testing RESOURCE ALLOCATION game..."
echo "======================================"
python -m negotiation_platform.main --quick --game resource_allocation

if [ $? -eq 0 ]; then
    echo "✅ Resource allocation test PASSED"
else
    echo "❌ Resource allocation test FAILED"
fi

# Test 2: Company Car
echo ""
echo "🚗 Testing COMPANY CAR game..."
echo "=============================="
python -m negotiation_platform.main --quick --game company_car

if [ $? -eq 0 ]; then
    echo "✅ Company car test PASSED"
else
    echo "❌ Company car test FAILED"
fi

# Test 3: Integrative Negotiations (if available)
echo ""
echo "🤝 Testing INTEGRATIVE NEGOTIATIONS game..."
echo "=========================================="
python -m negotiation_platform.main --quick --game integrative_negotiations

if [ $? -eq 0 ]; then
    echo "✅ Integrative negotiations test PASSED"
else
    echo "❌ Integrative negotiations test FAILED - may not be implemented yet"
fi

echo ""
echo "🎉 ALL GAME TESTS COMPLETED!"
echo "============================"
echo "Check the output above to see which games are working."
