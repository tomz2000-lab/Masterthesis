Main Module
===========

The main entry point for the Negotiation Platform, providing command-line interface and demonstration capabilities.

negotiation_platform.main module
--------------------------------

.. automodule:: negotiation_platform.main
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Command-line usage::

    # Run a single negotiation
    python main.py --quick --models model_a model_b --game company_car
    
    # Run model comparison
    python main.py --comparison --models model_a model_b model_c
    
    # Interactive mode
    python main.py

Programmatic usage::

    from negotiation_platform.main import run_single_negotiation
    from negotiation_platform.core.config_manager import ConfigManager
    
    config = ConfigManager()
    models = ["model_a", "model_b"]
    result = run_single_negotiation(config, models, "company_car")