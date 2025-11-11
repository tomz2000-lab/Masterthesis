Quick Start Guide
================

This guide will help you get started with the Negotiation Platform quickly.

Basic Usage
-----------

1. **Import the necessary modules**:

.. code-block:: python

   from negotiation_platform.main import run_single_negotiation
   from negotiation_platform.core.config_manager import ConfigManager

2. **Set up configuration**:

.. code-block:: python

   config_manager = ConfigManager()
   
   # Define models to use
   models = ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"]

3. **Run a negotiation**:

.. code-block:: python

   result = run_single_negotiation(
       config_manager, 
       models, 
       game_type="company_car"
   )
   
   print(f"Agreement reached: {result.get('agreement_reached', False)}")
   print(f"Final utilities: {result.get('final_utilities', {})}")

Running Different Game Types
----------------------------

**Company Car Negotiation**:

.. code-block:: python

   result = run_single_negotiation(config_manager, models, "company_car")

**Resource Allocation**:

.. code-block:: python

   result = run_single_negotiation(config_manager, models, "resource_allocation")

**Integrative Negotiation**:

.. code-block:: python

   result = run_single_negotiation(config_manager, models, "integrative_negotiations")

Analyzing Results
-----------------

The negotiation results contain comprehensive information:

.. code-block:: python

   # Check if agreement was reached
   if result['agreement_reached']:
       print(f"Agreement reached in round {result['agreement_round']}")
       print(f"Final price: {result.get('agreed_price', 'N/A')}")
   
   # View player utilities
   for player, utility in result['final_utilities'].items():
       print(f"{player}: {utility}")
   
   # Access detailed metrics
   metrics = result.get('metrics', {})
   for metric_name, player_scores in metrics.items():
       print(f"{metric_name}: {player_scores}")

Advanced Configuration
----------------------

**Custom Game Configuration**:

.. code-block:: python

   custom_config = {
       "starting_price": 50000,
       "buyer_budget": 45000,
       "seller_cost": 40000,
       "rounds": 6
   }
   
   result = run_single_negotiation(
       config_manager, 
       models, 
       "company_car",
       game_config=custom_config
   )

**Using Custom Models**:

.. code-block:: python

   # First register your model in model_configs.yaml, then:
   models = ["custom_model_1", "custom_model_2"]
   result = run_single_negotiation(config_manager, models, "company_car")

Batch Processing
----------------

For running multiple negotiations:

.. code-block:: python

   from negotiation_platform.main import run_model_comparison
   
   results = run_model_comparison(
       config_manager,
       models=["model_a", "model_b", "model_c"],
       games=["company_car", "resource_allocation"]
   )

Command Line Usage
------------------

You can also run negotiations from the command line:

.. code-block:: bash

   # Run single negotiation
   python -m negotiation_platform.main --game company_car --models model1 model2
   
   # Run model comparison
   python -m negotiation_platform.main --comparison --models model1 model2 model3

Next Steps
----------

* Read the :doc:`api/modules` for detailed API documentation
* Check out :doc:`examples` for more complex usage patterns
* Learn about :doc:`configuration` options
* Explore available :doc:`games` and their parameters