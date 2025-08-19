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

# Test all three games to verify corrected utility surplus calculations
echo "=== Testing Company Car (baseline) ==="
python -m negotiation_platform.main --quick --game company_car

echo "=== Testing Resource Allocation (corrected utility surplus) ==="
python -m negotiation_platform.main --quick --game resource_allocation

echo "=== Testing Integrative Negotiations (corrected utility surplus) ==="
python -m negotiation_platform.main --quick --game integrative_negotiations

echo "=== All Utility Surplus Tests Complete ==="
