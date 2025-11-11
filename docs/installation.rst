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

* torch >= 1.9.0
* transformers >= 4.20.0
* pandas >= 1.3.0
* numpy >= 1.21.0
* pydantic >= 1.8.0
* PyYAML >= 5.4.0
* statsmodels >= 0.12.0
* scipy >= 1.7.0

Installation Steps
------------------

1. **Clone the repository**:

.. code-block:: bash

   git clone https://github.com/yourusername/negotiation-platform.git
   cd negotiation-platform

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

   git clone https://github.com/yourusername/negotiation-platform.git
   cd negotiation-platform
   pip install -e .
   pip install -r requirements-dev.txt

Configuration
-------------

After installation, copy and modify the configuration files:

.. code-block:: bash

   cp negotiation_platform/configs/model_configs.yaml.example negotiation_platform/configs/model_configs.yaml
   cp negotiation_platform/configs/game_configs.yaml.example negotiation_platform/configs/game_configs.yaml

Edit these files according to your needs and available models.