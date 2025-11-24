Installation
============

Requirements
------------

* Python 3.8 or higher
* CUDA-compatible GPU (recommended for LLM inference)
* At least 16GB RAM (32GB recommended for larger models)

Dependencies
------------

The platform requires several Python packages:

**Core Dependencies**:
* torch >= 1.9.0
* transformers >= 4.20.0
* huggingface-hub >= 0.10.0
* tokenizers >= 0.12.0

**Configuration and Data Handling**:
* pyyaml >= 6.0
* pandas >= 1.3.0
* numpy >= 1.21.0
* pydantic >= 1.8.0

**Utilities**:
* tqdm >= 4.62.0
* requests >= 2.28.0

**Statistical Analysis**:
* scipy >= 1.7.0
* statsmodels >= 0.12.0

**Optional Dependencies** (for advanced features):
* accelerate >= 0.18.0 (faster model loading)
* bitsandbytes >= 0.39.0 (model quantization)

Installation Steps
------------------

1. **Clone the repository**:

.. code-block:: bash

   git clone https://github.com/tomz2000-lab/Masterthesis.git
   cd Masterthesis

2. **Create a virtual environment** (recommended):

.. code-block:: bash

   python -m venv negotiation_env
   source negotiation_env/bin/activate  # On Windows: negotiation_env\Scripts\activate

3. **Install dependencies**:

.. code-block:: bash

   pip install -r requirements.txt

4. **Verify installation**:

.. code-block:: bash

   python -m negotiation_platform.main --help
   
   # Or run a simple test
   python -c "from negotiation_platform.core.config_manager import ConfigManager; print('Installation successful!')"

GPU Setup (Optional but Recommended)
-------------------------------------

For better performance with large language models:

1. **Install CUDA** (if not already installed):
   
   Visit `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ and follow installation instructions.

2. **Install PyTorch with CUDA support**:

.. code-block:: bash

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Development Installation
------------------------

For contributing to the project:

.. code-block:: bash

   git clone https://github.com/tomz2000-lab/Masterthesis.git
   cd Masterthesis
   pip install -e .
   
   # Optional: Install development dependencies
   pip install pytest>=7.0.0 black>=22.0.0 flake8>=4.0.0

Configuration
-------------

The platform comes with pre-configured YAML files ready to use:

.. code-block:: bash

   # Configuration files are located in:
   negotiation_platform/configs/model_configs.yaml    # Model settings
   negotiation_platform/configs/game_configs.yaml     # Game parameters
   negotiation_platform/configs/platform_config.yaml  # Platform settings

**Important**: You may need to set your Hugging Face token for model access:

.. code-block:: bash

   export HF_TOKEN="your_huggingface_token_here"
   
   # On Windows:
   set HF_TOKEN=your_huggingface_token_here

Edit the configuration files according to your needs and available hardware.