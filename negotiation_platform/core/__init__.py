"""
Core Platform Components
========================

Central coordination layer for the negotiation platform architecture.

This module provides the core components that orchestrate negotiation sessions,
manage AI models, compute performance metrics, and handle configuration. These
components work together to provide a comprehensive framework for running and
analyzing multi-agent negotiations.

Core Components:
    - LLMManager: AI model loading, management, and interaction coordination
    - GameEngine: Game instance creation, registration, and lifecycle management  
    - MetricsCalculator: Performance metric computation and analysis
    - SessionManager: End-to-end negotiation session orchestration
    - ConfigManager: Configuration loading, validation, and management

Architecture:
    The core components follow a modular design where each component has clear
    responsibilities and well-defined interfaces. They can be used independently
    or composed together for comprehensive negotiation platform functionality.

Example:
    >>> from negotiation_platform.core import (
    ...     LLMManager, GameEngine, MetricsCalculator, 
    ...     SessionManager, ConfigManager
    ... )
    >>> 
    >>> # Initialize core components
    >>> config_manager = ConfigManager("config.yaml")
    >>> llm_manager = LLMManager(config_manager.get_model_configs())
    >>> game_engine = GameEngine()
    >>> metrics_calculator = MetricsCalculator()
    >>> 
    >>> # Create session manager with all components
    >>> session_manager = SessionManager(
    ...     llm_manager=llm_manager,
    ...     game_engine=game_engine,
    ...     metrics_calculator=metrics_calculator
    ... )
    >>> 
    >>> # Run complete negotiation session
    >>> result = session_manager.run_negotiation(
    ...     game_type="price_bargaining",
    ...     players=["model_a", "model_b"]
    ... )
"""

from .llm_manager import LLMManager
from .game_engine import GameEngine
from .metrics_calculator import MetricsCalculator
from .session_manager import SessionManager
from .config_manager import ConfigManager


__all__ = [
    'LLMManager', 'GameEngine', 'MetricsCalculator',
    'SessionManager', 'ConfigManager'
]
