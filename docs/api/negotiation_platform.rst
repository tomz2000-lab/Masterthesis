Negotiation Platform
====================

A comprehensive framework for conducting automated negotiations between Large Language Models (LLMs).
This platform enables researchers to study negotiation behaviors, biases, and strategies in controlled environments.

Main Entry Point
---------------

.. automodule:: negotiation_platform.main
   :members:
   :undoc-members:
   :show-inheritance:

Platform Architecture
-------------------

The negotiation platform consists of several key components:

* **Core Modules**: Session management, game engine, LLM management, configuration
* **Games**: Different negotiation scenarios (company car, resource allocation, integrative)
* **Models**: AI model wrappers and interfaces
* **Metrics**: Performance measurement and analysis tools
* **Results**: Statistical analysis and reporting utilities

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   negotiation_platform.core
   negotiation_platform.games
   negotiation_platform.models
   negotiation_platform.metrics

Package Contents
---------------

.. automodule:: negotiation_platform
   :members:
   :undoc-members:
   :show-inheritance: