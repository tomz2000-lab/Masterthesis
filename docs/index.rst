Welcome to Negotiation Platform's documentation!
==============================================

The Negotiation Platform is a comprehensive framework for conducting automated negotiations between Large Language Models (LLMs). 
This platform enables researchers to study negotiation behaviors, biases, and strategies in controlled environments.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   configuration

.. toctree::
   :maxdepth: 2
   :caption: Negotiation Platform:

   api/negotiation_platform

.. toctree::
   :maxdepth: 1
   :caption: Development:

   docstring_guidelines

Features
--------

* **Multi-Game Support**: Company car negotiations, resource allocation, integrative negotiations
* **LLM Integration**: Support for various language models via Hugging Face
* **Bias Detection**: Advanced statistical analysis tools for detecting negotiation biases
* **Configurable**: Flexible YAML-based configuration system
* **Metrics**: Comprehensive performance and fairness metrics
* **Analysis Tools**: Built-in statistical analysis and visualization capabilities

Quick Start
-----------

1. Install the package:

.. code-block:: bash

   git clone https://github.com/tomz2000-lab/Masterthesis.git
   cd Masterthesis
   pip install -r requirements.txt


2. Transfer project to a remote cluster (preserve directory structure)

.. code-block:: bash

   # Using scp (PowerShell / Linux): replace <user> and <host> and <remote_path>
   scp -r . <user>@<host>:/home/<user>/<remote_path>/Masterthesis

   # Using rsync (recommended for large repos / resumable transfers):
   rsync -avz --progress --exclude "negotiation_env/" --exclude ".git/" ./ <user>@<host>:/home/<user>/<remote_path>/Masterthesis

   # Example (your earlier cluster):
   scp -r . s123456@julia2.hpc.uni-wuerzburg.de:/home/s123456/Masterthesis


3. Submit a batch run on the cluster using the provided Slurm scripts

.. code-block:: bash

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


4. Transfer result files from cluster to local machine

.. code-block:: bash

   # Wait for jobs to complete, then transfer result files back to local machine
   # Download the generated .out files from the cluster
   scp <user>@<host>:/home/<user>/<remote_path>/Masterthesis/batch_comparison/ ./batch_comparison/ -r

   # Or download specific result files
   scp <user>@<host>:/home/<user>/<remote_path>/Masterthesis/batch_comparison/resource_game/resource_allocation_*.out ./batch_comparison/resource_game/

   # Example with specific job output:
   scp s123456@julia2.hpc.uni-wuerzburg.de:/home/s123456/Masterthesis/batch_comparison/resource_game/resource_allocation_2037717.out ./batch_comparison/resource_game/


5. Run win and metric comparisons locally

.. code-block:: bash

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

Notes
-----

* Replace placeholder usernames, hostnames and paths with your cluster credentials and target directories.
* Ensure the cluster has required Python packages installed or create a virtual environment and install `requirements.txt` there.
* If using GPUs, load the appropriate modules (CUDA, etc.) before running `sbatch` or inside your Slurm scripts.
* Slurm scripts in `batch_slurm_files/` may need path adjustments for the Python executable or the virtual environment activation commands.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`