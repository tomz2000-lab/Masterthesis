Welcome to Negotiation Platform's documentation!
==============================================

The Negotiation Platform is a comprehensive framework for conducting automated negotiations between Large Language Models (LLMs). 
This platform enables researchers to study negotiation behaviors, biases, and strategies in controlled environments.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/modules
   examples
   configuration
   games
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

   git clone https://github.com/yourusername/negotiation-platform
   cd negotiation-platform
   pip install -r requirements.txt

2. Run a simple negotiation:

.. code-block:: python

   from negotiation_platform.main import run_single_negotiation
   from negotiation_platform.core.config_manager import ConfigManager
   
   config = ConfigManager()
   models = ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"]
   result = run_single_negotiation(config, models, "company_car")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`