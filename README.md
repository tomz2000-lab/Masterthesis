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
   # Using scp: replace <user> and <host> and <remote_path>
   scp -r . <user>@<host>:/home/<user>/<remote_path>/Masterthesis

   # Example:
   scp -r . s123456@julia2.hpc.uni-wuerzburg.de:/home/s123456/Masterthesis
```

3. Submit a batch run on the cluster using the provided Slurm scripts
```bash
   # SSH into the cluster
   ssh <user>@<host>

   # Change to project directory
   cd /home/<user>/<remote_path>/Masterthesis

   # Inspect available slurm scripts (examples are in `batch_slurm_files/`)
   ls -l batch_slurm_files/ run_*.slurm

   # Submit a pre-provided batch job (adjust the script to set correct paths or env vars)
   sbatch batch_slurm_files/run_company_car.slurm

   # Use `squeue -u <user>` to check job status
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
   # Run resource allocation results:
   python results/win_statistics.py batch_comparison/resource_game/resource_allocation_2021495.out
   python results/metrics_statistics.py batch_comparison/resource_game/resource_allocation_2021495.out

   # For company car results:
   python results/win_statistics.py batch_comparison/company_car/company_car_2021072.out
   python results/metrics_statistics.py batch_comparison/company_car/company_car_2021072.out

   # For integrative negotiation results:
   python results/win_statistics.py batch_comparison/integrative_game/integrative_negotiation_2021496.out
   python results/metrics_statistics.py batch_comparison/integrative_game/integrative_negotiation_2021496.out
```


## Runs according to constellation

Within this table you can find all the nubers of the runs with thier corresponding comparison constellation:

| Pairs                       | Car Game | Integrative Game | Resource Game |
|-----------------------------|----------|------------------|--------------|
| Mistral-8B vs Mistral-7B    | 2021072  | 2021496          | 2021495      |
| Mistral-8B vs Llama-3B      | 2030276  | 2030278          | 2030277      |
| Mistral-8B vs Llama-8B      | 2023089  | 2023062          | 2023060      |
| Mistral-8B vs Qwen-3B       | 2032207  | 2032209          | 2032208      |
| Mistral-8B vs Qwen-7B       | 2033475  | 2033478          | 2033476      |
| Mistral-7B vs Llama-3B      | 2034154  | 2034156          | 2024155      |
| Mistral-7B vs Llama-8B      | 2034768  | 2034776          | 2034775      |
| Mistral-7B vs Qwen-3B       | 2035478  | 2035480          | 2035479      |
| Mistral-7B vs Qwen-7B       | 2023648  | 2023654          | 2023653      |
| Llama-3B vs Llama-8B        | 2019174  | 2019176          | 2019173      |
| Llama-3B vs Qwen-3B         | 2028782  | 2028788          | 2028783      |
| Llama-3B vs Qwen-7B         | 2036337  | 2036339          | 2036338      |
| Llama-8B vs Qwen-3B         | 2037716  | 2037718          | 2937717      |
| Llama-8B vs Qwen-7B         | 2016569  | 2016572          | 2016568      |
| Qwen-3B vs Qwen-7B          | 2013438  | 2013439          | 2013440      |


## Documentation

📖 **Full Documentation**: [masterthesis-tom-ziegler.readthedocs.io](https://masterthesis-tom-ziegler.readthedocs.io/en/latest/)

## Citation

If you use the negotiation platform described in this work, please cite it as follows:

```bibtex
@mastersthesis{ziegler2025negotiation,
  title={Large Language Models in Business Negotiations: Capabilities and Performance Under Uncertainty},
  author={Ziegler, Tom},
  year={2025},
  school={University of Würzburg}
}
```

Feel free to include this citation when referencing the negotiation framework or related experiments.


## Acknowledgement
This work builds on the concepts and methods established by Bianchi et al. (2024), who introduced the NegotiationArena platform—an evaluation framework for assessing how effectively Large Language Models negotiate across different types of two-agent negotiation games. Their study provides foundational insights for analyzing agent interactions, negotiation dynamics, and benchmarking LLM negotiation capabilities in varied scenarios.

```bibtex
@article{bianchi2024llms,
      title={How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis}, 
      author={Federico Bianchi and Patrick John Chia and Mert Yuksekgonul and Jacopo Tagliabue and Dan Jurafsky and James Zou},
      year={2024},
      eprint={2402.05863},
      journal={arXiv},
}
```
