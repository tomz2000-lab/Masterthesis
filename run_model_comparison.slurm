#!/bin/bash
#SBATCH --job-name=model_comparison
#SBATCH --output=model_comparison_%j.out
#SBATCH --error=model_comparison_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

echo "=== Model Comparison Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "GPU info:"
nvidia-smi

# Load necessary modules
module load python/3.9
module load cuda/11.8

# Activate virtual environment if needed
# source negotiation_env/bin/activate

# Change to the correct directory
cd negotiation_platform

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== Starting Model Comparison ==="
echo "Testing Llama 3B vs 8B models across all three games"

# Run the model comparison
python run_model_comparison.py

echo "=== Model Comparison Complete ==="
echo "End time: $(date)"

# Show final results summary
echo "=== Results Summary ==="
ls -la model_comparison_*
echo "Check the summary file for detailed results."