"""
Negotiation Games
=================

Complete collection of negotiation game implementations for the platform.

This module provides access to all available negotiation game types, from simple
bilateral price negotiations to complex multi-issue resource allocation scenarios.
Each game implements the BaseGame interface while providing unique mechanics,
scoring systems, and strategic challenges.

Available Games:
    - BaseGame: Abstract foundation for all negotiation implementations
    - CompanyCarGame: Bilateral price negotiation with market dynamics
    - ResourceAllocationGame: Multi-resource distribution between teams
    - IntegrativeNegotiationsGame: Multi-issue office space and responsibility allocation

Game Categories:
    1. Bilateral Negotiations: Two-party scenarios with clear opposing interests
       - CompanyCarGame: Buyer vs. Seller price negotiations
       
    2. Multi-Issue Negotiations: Complex scenarios with multiple decision points
       - IntegrativeNegotiationsGame: IT vs. Marketing office resource allocation
       
    3. Resource Distribution: Allocation of limited resources between parties
       - ResourceAllocationGame: Development vs. Marketing team resource sharing

Design Philosophy:
    All games follow consistent design principles:
    - Clear role definitions with distinct interests and priorities
    - Realistic constraints and market dynamics
    - BATNA (Best Alternative to Negotiated Agreement) mechanisms
    - Win-win outcome possibilities requiring creative problem-solving
    - Structured proposal systems for clear communication
    - Comprehensive utility tracking and performance analysis

Usage Patterns:
    Games can be used individually for specific research questions or combined
    in comparative studies to analyze negotiation strategies across different
    scenarios and complexity levels.

Example:
    >>> from negotiation_platform.games import (
    ...     CompanyCarGame, ResourceAllocationGame, IntegrativeNegotiationsGame
    ... )
    >>> 
    >>> # Simple bilateral price negotiation
    >>> car_config = {"starting_price": 42000, "rounds": 5}
    >>> car_game = CompanyCarGame(car_config)
    >>> 
    >>> # Complex multi-resource allocation
    >>> resource_config = {"total_gpus": 10, "total_budget": 100000}
    >>> resource_game = ResourceAllocationGame(resource_config)
    >>> 
    >>> # Multi-issue integrative negotiation
    >>> office_config = {"rounds": 8, "batna_decay": 0.02}
    >>> office_game = IntegrativeNegotiationsGame(office_config)

Research Applications:
    - AI negotiation strategy development and testing
    - Comparative analysis of negotiation algorithms
    - Human-AI negotiation interaction studies
    - Game theory and mechanism design research
    - Organizational behavior and team dynamics analysis
"""

from .base_game import BaseGame
from .price_bargaining import CompanyCarGame
from .resource_allocation import ResourceAllocationGame
from .integrative_negotiation import IntegrativeNegotiationsGame

__all__ = [
    'BaseGame', 'CompanyCarGame', 'ResourceAllocationGame',
    'IntegrativeNegotiationsGame'
]
