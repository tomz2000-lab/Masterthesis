"""
Negotiation Platform
====================

A comprehensive framework for conducting automated negotiations between Large Language Models (LLMs).
This platform enables researchers to study negotiation behaviors, biases, and strategies in controlled environments.

Key Components:
    - Core: Session management, game engine, LLM management, configuration
    - Games: Different negotiation scenarios and implementations
    - Models: AI model wrappers and interfaces  
    - Metrics: Performance measurement and analysis tools

Example:
    >>> from negotiation_platform.main import run_single_negotiation
    >>> from negotiation_platform.core.config_manager import ConfigManager
    >>> config = ConfigManager()
    >>> models = ["model_a", "model_b"]
    >>> result = run_single_negotiation(config, models, "company_car")
"""

from negotiation_platform.core import (
    LLMManager,
    GameEngine, 
    MetricsCalculator,
    SessionManager,
    ConfigManager
)

__version__ = "1.0.0"
__author__ = "Tom Ziegler"

__all__ = [
    "LLMManager",
    "GameEngine", 
    "MetricsCalculator",
    "SessionManager",
    "ConfigManager"
]