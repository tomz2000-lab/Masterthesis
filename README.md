# Negotiation Platform 

A comprehensive framework for conducting LLM negotiations to compare the models in theri performance.

## Features

- **Multi-Game Support**: Company car bargaining, resource allocation, integrative negotiations
- **LLM Integration**: Support for Hugging Face models (Llama, Mistral, Qwen families)
- **Bias Detection**: Advanced statistical analysis for negotiation fairness
- **Performance Metrics**: Utility surplus, risk minimization, feasibility, deadline sensitivity
- **Memory Optimization**: Efficient GPU usage with quantization support

## Quick Start

1. Install the package:
```bash
   git clone https://github.com/tomz2000-lab/Masterthesis.git
   cd Masterthesis
   pip install -r requirements.txt
```

2. Transfer project to a remote cluster (preserve directory structure)
```bash
   # Using scp (PowerShell / Linux): replace <user> and <host> and <remote_path>
   scp -r . <user>@<host>:/home/<user>/<remote_path>/Masterthesis

   # Using rsync (recommended for large repos / resumable transfers):
   rsync -avz --progress --exclude "negotiation_env/" --exclude ".git/" ./ <user>@<host>:/home/<user>/<remote_path>/Masterthesis

   # Example (your earlier cluster):
   scp -r . s123456@julia2.hpc.uni-wuerzburg.de:/home/s123456/Masterthesis
```

3. Submit a batch run on the cluster using the provided Slurm scripts
```bash
   # SSH into the cluster
   ssh <user>@<host>

   # Change to project directory
   cd /home/<user>/<remote_path>/Masterthesis

   # Inspect available slurm scripts (examples are in `batch_slurm_files/` and top-level)
   ls -l batch_slurm_files/ run_*.slurm

   # Submit a pre-provided batch job (adjust the script to set correct paths or env vars)
   sbatch batch_slurm_files/run_company_car.slurm

   # Or submit a generic single-game runner
   sbatch run_single_game.slurm

   # Use `squeue -u <user>` to check job status, and `sacct -j <jobid>` to inspect job output
```

4. Transfer result files from cluster to local machine
```bash
   # Wait for jobs to complete, then transfer result files back to local machine
   # Download the generated .out files from the cluster
   scp <user>@<host>:/home/<user>/<remote_path>/Masterthesis/batch_comparison/ ./batch_comparison/ -r

   # Or download specific result files
   scp <user>@<host>:/home/<user>/<remote_path>/Masterthesis/batch_comparison/resource_game/resource_allocation_*.out ./batch_comparison/resource_game/

   # Example with specific job output:
   scp s123456@julia2.hpc.uni-wuerzburg.de:/home/s123456/Masterthesis/batch_comparison/resource_game/resource_allocation_2037717.out ./batch_comparison/resource_game/
```

5. Run win and metric comparisons locally
```bash
   # Run win statistics analysis on specific result file
   python results/win_statistics.py batch_comparison/resource_game/resource_allocation_2037717.out

   # Run metrics statistics analysis on the same file
   python results/metrics_statistics.py batch_comparison/resource_game/resource_allocation_2037717.out

   # For company car results:
   python results/win_statistics.py batch_comparison/company_car/company_car_1862304.out
   python results/metrics_statistics.py batch_comparison/company_car/company_car_1862304.out

   # For integrative negotiation results:
   python results/win_statistics.py batch_comparison/integrative_game/integrative_negotiation_1858216.out
   python results/metrics_statistics.py batch_comparison/integrative_game/integrative_negotiation_1858216.out
```

## Documentation

📖 **Full Documentation**: [masterthesis-tom-ziegler.readthedocs.io](https://masterthesis-tom-ziegler.readthedocs.io/en/latest/)

## Citation

```bibtex
@mastersthesis{ziegler2025negotiation,
  title={Large Language Models in Business Negotiations: Capabilities and Performance Under Uncertainty},
  author={Ziegler, Tom},
  year={2025},
  school={University of Würzburg}
}
```

## Acknowledgement
This work adapted on the ideas and fundamentals of:
```bibtex
@article{bianchi2024llms,
      title={How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis}, 
      author={Federico Bianchi and Patrick John Chia and Mert Yuksekgonul and Jacopo Tagliabue and Dan Jurafsky and James Zou},
      year={2024},
      eprint={2402.05863},
      journal={arXiv},
}
```
