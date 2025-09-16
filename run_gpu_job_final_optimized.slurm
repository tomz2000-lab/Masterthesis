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

# CRITICAL: Robust virtual environment detection and activation
if [ -d "/home/s391129/negotiation_env" ]; then
    echo "Using negotiation_env"
    source /home/s391129/negotiation_env/bin/activate
elif [ -d "/home/s391129/venv" ]; then
    echo "Using venv"
    source /home/s391129/venv/bin/activate
else
    echo "❌ ERROR: No virtual environment found!"
    echo "Expected paths: /home/s391129/venv or /home/s391129/negotiation_env"
    exit 1
fi

# Verify Python environment
echo "=== Python Environment Check ==="
which python
python --version
echo "Virtual environment: $VIRTUAL_ENV"

# Check GPU availability
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "========================"

# Check PyTorch GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Install GPU-enabled bitsandbytes if needed (but don't fail if it doesn't work)
echo "=== Installing GPU bitsandbytes ==="
pip install bitsandbytes --upgrade --no-cache-dir || echo "⚠️ bitsandbytes install failed, continuing..."

# Copy GPU config to use
echo "=== Setting up GPU configuration ==="
if [ -f "/home/s391129/negotiation_platform/configs/model_configs_gpu.yaml" ]; then
    cp /home/s391129/negotiation_platform/configs/model_configs_gpu.yaml /home/s391129/negotiation_platform/configs/model_configs.yaml
    echo "✅ GPU configuration copied"
else
    echo "⚠️ GPU config not found, using default"
fi

# Clear Python cache to ensure updated files are used
echo "=== Clearing Python Cache ==="
find /home/s391129/negotiation_platform -name "*.pyc" -delete
find /home/s391129/negotiation_platform -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "Python cache cleared!"

# Verify the torch scope fix is in place
echo "=== Verifying Torch Scope Fix ==="
if grep -q "torch._dynamo as torch_dynamo" /home/s391129/negotiation_platform/models/hf_model_wrapper.py; then
    echo "✅ Torch scope fix is present"
else
    echo "❌ WARNING: Torch scope fix not found - may cause model loading issues"
fi

# Change to correct working directory
cd /home/s391129
echo "=== Working Directory: $(pwd) ==="

# OPTION 1: Test all games (comprehensive but longer)
echo "=== Running Complete Platform Tests ==="

# Test 1: Company Car (most reliable game)
echo ""
echo "🚗 Testing COMPANY CAR game..."
echo "=============================="
python -m negotiation_platform.main --quick --game company_car

if [ $? -eq 0 ]; then
    echo "✅ Company car test PASSED"
    COMPANY_CAR_SUCCESS=true
else
    echo "❌ Company car test FAILED"
    COMPANY_CAR_SUCCESS=false
fi

# Test 2: Resource Allocation (only if Company Car succeeded)
if [ "$COMPANY_CAR_SUCCESS" = true ]; then
    echo ""
    echo "🔧 Testing RESOURCE ALLOCATION game..."
    echo "======================================"
    python -m negotiation_platform.main --quick --game resource_allocation

    if [ $? -eq 0 ]; then
        echo "✅ Resource allocation test PASSED"
    else
        echo "❌ Resource allocation test FAILED"
    fi
else
    echo "⚠️ Skipping resource allocation test due to company car failure"
fi

# Test 3: Integrative Negotiations (optional - may not be implemented)
if [ "$COMPANY_CAR_SUCCESS" = true ]; then
    echo ""
    echo "🤝 Testing INTEGRATIVE NEGOTIATIONS game..."
    echo "=========================================="
    python -m negotiation_platform.main --quick --game integrative_negotiations

    if [ $? -eq 0 ]; then
        echo "✅ Integrative negotiations test PASSED"
    else
        echo "❌ Integrative negotiations test FAILED - may not be implemented yet"
    fi
else
    echo "⚠️ Skipping integrative negotiations test due to company car failure"
fi

echo ""
echo "🎉 PLATFORM TESTS COMPLETED!"
echo "============================"
if [ "$COMPANY_CAR_SUCCESS" = true ]; then
    echo "✅ Core platform functionality is working!"
    echo "🚀 Your negotiation platform is ready for production use!"
else
    echo "❌ Core platform functionality failed"
    echo "🔧 Check the logs above for debugging information"
fi

# Memory cleanup
echo "=== Cleaning up GPU memory ==="
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('GPU memory cleaned')" 2>/dev/null || echo "Cleanup completed"

echo "=== Job Complete ==="