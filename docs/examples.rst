Examples
========

This section provides comprehensive examples of using the Negotiation Platform.

Basic Company Car Negotiation
------------------------------

.. code-block:: python

   from negotiation_platform.main import run_single_negotiation
   from negotiation_platform.core.config_manager import ConfigManager

   # Initialize configuration
   config = ConfigManager()
   
   # Define negotiating models
   models = ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"]
   
   # Run negotiation
   result = run_single_negotiation(config, models, "company_car")
   
   # Display results
   print(f"Agreement: {result['agreement_reached']}")
   if result['agreement_reached']:
       print(f"Final price: €{result['agreed_price']:,}")
       print(f"Agreement round: {result['agreement_round']}")

Custom Configuration Example
----------------------------

.. code-block:: python

   # Custom game parameters
   custom_config = {
       "starting_price": 50000,  # Higher starting price
       "buyer_budget": 48000,    # Higher buyer budget
       "seller_cost": 42000,     # Higher seller cost
       "buyer_batna": 46000,     # Alternative option cost
       "seller_batna": 43000,    # Minimum acceptable price
       "rounds": 8,              # More negotiation rounds
       "batna_decay": {
           "buyer": 0.02,        # BATNA decay per round
           "seller": 0.015
       }
   }
   
   result = run_single_negotiation(
       config, 
       models, 
       "company_car",
       game_config=custom_config
   )

Resource Allocation Example
---------------------------

.. code-block:: python

   # Resource allocation between development teams
   resource_config = {
       "total_gpu_hours": 100,
       "total_cpu_hours": 200,
       "dev_team_priorities": {
           "gpu_weight": 0.7,
           "cpu_weight": 0.3
       },
       "marketing_team_priorities": {
           "gpu_weight": 0.4,
           "cpu_weight": 0.6
       }
   }
   
   result = run_single_negotiation(
       config, 
       models, 
       "resource_allocation",
       game_config=resource_config
   )
   
   print("Resource allocation result:")
   for team, resources in result.get('final_allocation', {}).items():
       print(f"{team}: {resources}")

Batch Model Comparison
----------------------

.. code-block:: python

   from negotiation_platform.main import run_model_comparison
   
   # Compare multiple models across different games
   models = [
       "microsoft/DialoGPT-medium",
       "microsoft/DialoGPT-large", 
       "facebook/blenderbot-400M-distill"
   ]
   
   games = ["company_car", "resource_allocation", "integrative_negotiations"]
   
   results = run_model_comparison(config, models, games)
   
   # Analyze results
   for game_type, game_results in results.items():
       print(f"\n{game_type.upper()} RESULTS:")
       for model_pair, negotiation_result in game_results.items():
           agreement = negotiation_result.get('agreement_reached', False)
           print(f"  {model_pair}: {'Agreement' if agreement else 'No Deal'}")

Statistical Analysis Example
----------------------------

.. code-block:: python

   from results.compare_games_statistics_FIXED import (
       parse_negotiation_log_corrected,
       analyze_role_bias,
       analyze_first_move_advantage
   )
   
   # Parse negotiation logs
   df, game_type = parse_negotiation_log_corrected("negotiation_logs.out")
   
   # Analyze for role bias
   role_bias_results = analyze_role_bias(df, game_type)
   print("Role bias analysis:")
   for test_name, result in role_bias_results.items():
       print(f"  {test_name}: p-value = {result.get('p_value', 'N/A')}")
   
   # Analyze first-move advantage
   fma_results = analyze_first_move_advantage(df)
   print(f"First-move advantage p-value: {fma_results.get('p_value', 'N/A')}")

Advanced Session Management
---------------------------

.. code-block:: python

   from negotiation_platform.core.session_manager import SessionManager
   from negotiation_platform.core.llm_manager import LLMManager
   from negotiation_platform.core.game_engine import GameEngine
   from negotiation_platform.core.metrics_calculator import MetricsCalculator
   
   # Manual session setup for advanced control
   llm_manager = LLMManager(config.get_config("model_configs"))
   game_engine = GameEngine()
   metrics_calculator = MetricsCalculator()
   
   session_manager = SessionManager(
       llm_manager=llm_manager,
       game_engine=game_engine,
       metrics_calculator=metrics_calculator,
       max_turn_retries=5  # Custom retry limit
   )
   
   # Custom seed messages for different behavior
   seed_messages = {
       "model_a": "You are an aggressive negotiator focused on winning.",
       "model_b": "You prioritize finding mutually beneficial solutions."
   }
   
   result = session_manager.run_negotiation(
       game_type="company_car",
       players=["model_a", "model_b"],
       game_config=custom_config,
       seed_messages=seed_messages
   )

Metrics Analysis Example
------------------------

.. code-block:: python

   # Extract and analyze metrics from results
   metrics = result.get('metrics', {})
   
   # Feasibility analysis
   feasibility_scores = metrics.get('feasibility', {})
   print("Feasibility scores:")
   for player, score in feasibility_scores.items():
       print(f"  {player}: {score:.3f}")
   
   # Utility surplus analysis  
   surplus_scores = metrics.get('utility_surplus', {})
   print("Utility surplus:")
   for player, surplus in surplus_scores.items():
       print(f"  {player}: {surplus:.2f}")
   
   # Risk minimization scores
   risk_scores = metrics.get('risk_minimization', {})
   print("Risk minimization:")
   for player, risk in risk_scores.items():
       print(f"  {player}: {risk:.3f}")

Error Handling Example
----------------------

.. code-block:: python

   import logging
   from negotiation_platform.main import setup_logging
   
   # Set up detailed logging
   setup_logging("DEBUG")
   logger = logging.getLogger(__name__)
   
   try:
       result = run_single_negotiation(config, models, "company_car")
       
       if not result.get('agreement_reached', False):
           logger.warning("Negotiation failed to reach agreement")
           
           # Analyze why negotiation failed
           actions_history = result.get('actions_history', [])
           final_round = len(actions_history)
           logger.info(f"Negotiation ended after {final_round} rounds")
           
   except Exception as e:
       logger.error(f"Negotiation failed with error: {e}")
       # Handle gracefully or re-raise