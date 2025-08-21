#!/bin/bash
# Simple file transfer script to upload test files to HPC cluster

echo "📤 Transferring test files to HPC cluster..."

# Upload the modified SLURM test script
echo "Uploading SLURM test script..."
scp run_experiment_cluster.slurm s391129@julia2.hpc.uni-wuerzburg.de:~/Masterthesis/

# Check if upload was successful
if [ $? -eq 0 ]; then
    echo "✅ SLURM test script uploaded successfully"
else
    echo "❌ Failed to upload SLURM test script"
    exit 1
fi

echo ""
echo "🚀 Files transferred! Now you can:"
echo "1. SSH to the cluster: ssh s391129@julia2.hpc.uni-wuerzburg.de"
echo "2. Navigate to project: cd Masterthesis"
echo "3. Submit test job: sbatch run_experiment_cluster.slurm"
echo "4. Monitor job: squeue -u \$USER"
echo "5. Check results: cat logs/test_*.out"
