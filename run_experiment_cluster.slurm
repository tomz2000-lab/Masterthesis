#!/bin/bash
#SBATCH --job-name=negotiation_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@uni-wuerzburg.de

# HPC Cluster Game Testing Script
# ==============================
# This script tests all three negotiation games with minimal sessions
# to verify the platform works correctly before running full experiments.

echo "🎮 Testing All Negotiation Games on HPC Cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "=============================================="

# Create necessary directories
mkdir -p logs

# Load required modules
module load python/3.9
module load cuda/11.8

# Activate virtual environment
source negotiation_env/bin/activate

# Verify GPU availability
echo "🔍 Checking GPU availability:"
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Set environment variables for optimal performance
export CUDA_LAUNCH_BLOCKING=1
export HF_HOME="/tmp/huggingface_cache_$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="/tmp/transformers_cache_$SLURM_JOB_ID"

# Install any missing packages
pip install -r requirements.txt

echo ""
echo "🎯 Testing All Three Games (1 session each)..."
echo "=============================================="

# Test Game 1: Resource Allocation
echo ""
echo "🔧 Testing RESOURCE ALLOCATION game..."
python -c "
import sys
sys.path.insert(0, '.')
from negotiation_platform.core.config_manager import ConfigManager
from negotiation_platform.core.llm_manager import LLMManager
from negotiation_platform.core.game_engine import GameEngine
from negotiation_platform.core.metrics_calculator import MetricsCalculator
from negotiation_platform.core.session_manager import SessionManager

print('Initializing components...')
config_manager = ConfigManager()
llm_manager = LLMManager(config_manager.get_config('model_configs'))
game_engine = GameEngine()
metrics_calculator = MetricsCalculator()
session_manager = SessionManager(llm_manager, game_engine, metrics_calculator)

print('Running resource_allocation negotiation...')
result = session_manager.run_negotiation(
    players=['model_a', 'model_b'],
    game_type='resource_allocation',
    game_config=config_manager.get_game_config('resource_allocation'),
    num_rounds=5
)

if result:
    print('✅ Resource allocation test PASSED')
    print(f'Agreement reached: {result.get(\"agreement_reached\", False)}')
    if 'metrics' in result:
        print(f'Metrics calculated: {list(result[\"metrics\"].keys())}')
else:
    print('❌ Resource allocation test FAILED')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Resource allocation test failed - stopping"
    exit 1
fi

# Test Game 2: Company Car
echo ""
echo "🚗 Testing COMPANY CAR game..."
python -c "
import sys
sys.path.insert(0, '.')
from negotiation_platform.core.config_manager import ConfigManager
from negotiation_platform.core.llm_manager import LLMManager
from negotiation_platform.core.game_engine import GameEngine
from negotiation_platform.core.metrics_calculator import MetricsCalculator
from negotiation_platform.core.session_manager import SessionManager

print('Initializing components...')
config_manager = ConfigManager()
llm_manager = LLMManager(config_manager.get_config('model_configs'))
game_engine = GameEngine()
metrics_calculator = MetricsCalculator()
session_manager = SessionManager(llm_manager, game_engine, metrics_calculator)

print('Running company_car negotiation...')
result = session_manager.run_negotiation(
    players=['model_a', 'model_b'],
    game_type='company_car',
    game_config=config_manager.get_game_config('company_car'),
    num_rounds=5
)

if result:
    print('✅ Company car test PASSED')
    print(f'Agreement reached: {result.get(\"agreement_reached\", False)}')
    if 'metrics' in result:
        print(f'Metrics calculated: {list(result[\"metrics\"].keys())}')
else:
    print('❌ Company car test FAILED')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Company car test failed - stopping"
    exit 1
fi

# Test Game 3: Integrative Negotiations
echo ""
echo "🤝 Testing INTEGRATIVE NEGOTIATIONS game..."
python -c "
import sys
sys.path.insert(0, '.')
from negotiation_platform.core.config_manager import ConfigManager
from negotiation_platform.core.llm_manager import LLMManager
from negotiation_platform.core.game_engine import GameEngine
from negotiation_platform.core.metrics_calculator import MetricsCalculator
from negotiation_platform.core.session_manager import SessionManager

print('Initializing components...')
config_manager = ConfigManager()
llm_manager = LLMManager(config_manager.get_config('model_configs'))
game_engine = GameEngine()
metrics_calculator = MetricsCalculator()
session_manager = SessionManager(llm_manager, game_engine, metrics_calculator)

print('Running integrative_negotiations negotiation...')
result = session_manager.run_negotiation(
    players=['model_a', 'model_b'],
    game_type='integrative_negotiations',
    game_config=config_manager.get_game_config('integrative_negotiations'),
    num_rounds=5
)

if result:
    print('✅ Integrative negotiations test PASSED')
    print(f'Agreement reached: {result.get(\"agreement_reached\", False)}')
    if 'metrics' in result:
        print(f'Metrics calculated: {list(result[\"metrics\"].keys())}')
else:
    print('❌ Integrative negotiations test FAILED')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Integrative negotiations test failed - stopping"
    exit 1
fi

echo ""
echo "🎉 ALL GAME TESTS COMPLETED SUCCESSFULLY!"
echo "========================================"
echo "✅ resource_allocation - Working"
echo "✅ company_car - Working"  
echo "✅ integrative_negotiations - Working"
echo ""
echo "🚀 Ready for full experiment batch!"
echo "Time completed: $(date)"

# Cleanup temporary cache
rm -rf /tmp/huggingface_cache_$SLURM_JOB_ID
rm -rf /tmp/transformers_cache_$SLURM_JOB_ID

echo "✅ Test job completed successfully!"
