API Documentation
=================

.. toctree::
   :maxdepth: 4

   negotiation_platform
   results

Core Modules
------------

.. autosummary::
   :toctree: generated/
   
   negotiation_platform.core.session_manager
   negotiation_platform.core.game_engine
   negotiation_platform.core.llm_manager
   negotiation_platform.core.metrics_calculator
   negotiation_platform.core.config_manager

Game Modules
------------

.. autosummary::
   :toctree: generated/
   
   negotiation_platform.games.base_game
   negotiation_platform.games.price_bargaining_guidance_free
   negotiation_platform.games.resource_exchange_car_structure_guidance_free
   negotiation_platform.games.integrative_negotiation_guidance_free

Model Modules
-------------

.. autosummary::
   :toctree: generated/
   
   negotiation_platform.models.base_model
   negotiation_platform.models.hf_model_wrapper
   negotiation_platform.models.action_schemas

Metrics Modules
---------------

.. autosummary::
   :toctree: generated/
   
   negotiation_platform.metrics.feasibility
   negotiation_platform.metrics.deadline_sensitivity
   negotiation_platform.metrics.utility_surplus
   negotiation_platform.metrics.risk_minimization

Analysis Tools
--------------

.. autosummary::
   :toctree: generated/
   
   results.compare_games_statistics_FIXED