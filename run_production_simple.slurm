#!/bin/bash
#SBATCH --job-name=negotiation_production
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=negotiation_%j.out
#SBATCH --error=negotiation_%j.err

echo "=== Production Negotiation Run ==="
echo "Job ID: $SLURM_JOB_ID, Node: $SLURMD_NODENAME"

# Activate environment (robust detection)
if [ -d "/home/s391129/venv" ]; then
    source /home/s391129/venv/bin/activate
elif [ -d "/home/s391129/negotiation_env" ]; then
    source /home/s391129/negotiation_env/bin/activate
else
    echo "❌ No virtual environment found"; exit 1
fi

# Quick environment check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Set up configuration
cp /home/s391129/negotiation_platform/configs/model_configs_gpu.yaml /home/s391129/negotiation_platform/configs/model_configs.yaml 2>/dev/null || true

# Clear cache and run
find /home/s391129/negotiation_platform -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

cd /home/s391129

echo "=== Running Company Car Negotiation ==="
python -m negotiation_platform.main --quick --game company_car

echo "=== Production Run Complete ==="