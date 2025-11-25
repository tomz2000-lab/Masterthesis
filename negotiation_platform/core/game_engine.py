"""
Game Engine
===========

The GameEngine serves as a factory and registry for all negotiation game types.
It manages game registration, instantiation, and provides discovery capabilities
for available game types within the platform.

Key Features:
    - Dynamic game type registration and discovery
    - Type-safe game instantiation with configuration validation
    - Built-in registry of default negotiation games
    - Extensible architecture for custom game implementations
    - Comprehensive game metadata and information retrieval

Supported Game Types:
    - company_car: Bilateral price negotiation for vehicle purchases
    - resource_allocation: Multi-resource distribution between teams
    - integrative_negotiations: Multi-issue collaborative negotiations

Architecture:
    The GameEngine follows the Factory pattern, maintaining a registry of
    game classes mapped to string identifiers. This allows for clean separation
    between game logic and session management while enabling runtime game
    type selection and dynamic extension.
"""

from typing import Dict, Type, Any, Optional
import logging
from negotiation_platform.games.base_game import BaseGame
from negotiation_platform.games.price_bargaining import CompanyCarGame
from negotiation_platform.games.resource_allocation import ResourceAllocationGame
from negotiation_platform.games.integrative_negotiation import IntegrativeNegotiationsGame


class GameEngine:
    """
    Factory and registry for negotiation game types with dynamic instantiation capabilities.
    
    The GameEngine manages the complete lifecycle of game type registration and instance
    creation. It maintains a registry of available game types and provides type-safe
    instantiation with configuration validation.
    
    Design Pattern:
        Implements the Factory pattern combined with a Registry pattern to enable
        clean separation between game logic and session orchestration. Game types
        are registered once and can be instantiated multiple times with different
        configurations.
    
    Key Features:
        - Type-safe game registration with BaseGame inheritance validation
        - Configuration-driven game instantiation
        - Built-in registration of default game types
        - Runtime game type discovery and metadata retrieval
        - Extensible architecture for custom game implementations
        - Comprehensive error handling for invalid game types
    
    Attributes:
        registered_games (Dict[str, Type[BaseGame]]): Registry mapping game type names
            to their corresponding game class implementations.
        logger (logging.Logger): Logger for game engine events and debugging.
    
    Example:
        >>> engine = GameEngine()
        >>> print(engine.get_available_games())
        ['company_car', 'resource_allocation', 'integrative_negotiations']
        >>> game = engine.create_game('company_car', {'max_rounds': 5})
        >>> isinstance(game, BaseGame)
        True
    
    Raises:
        ValueError: If attempting to register invalid game class or create unknown game type.
    """

    def __init__(self):
        """
        Initialize a new GameEngine instance with default game types.
        
        Creates an empty game registry and automatically registers all built-in
        negotiation game types. The engine is immediately ready for use after
        initialization.
        
        Default Game Types Registered:
            - company_car: Bilateral vehicle price negotiation
            - resource_allocation: Multi-resource team distribution
            - integrative_negotiations: Multi-issue collaborative negotiation
        
        Example:
            >>> engine = GameEngine()
            >>> 'company_car' in engine.get_available_games()
            True
        """
        self.registered_games: Dict[str, Type[BaseGame]] = {}
        self.logger = logging.getLogger(__name__) #Must become before next line in order to work
        self._register_default_games()

    def _register_default_games(self):
        """
        Register the built-in negotiation game types with the engine.
        
        Automatically registers all default game implementations that ship
        with the platform. These games provide comprehensive negotiation
        scenarios covering different strategic contexts and complexity levels.
        
        Registered Games:
            - company_car: Bilateral vehicle price negotiation with time pressure
            - resource_allocation: Multi-resource distribution between teams
            - integrative_negotiations: Multi-issue collaborative negotiations
        
        Game Selection:
            Each game type represents a different research context and strategic
            challenge, enabling comparative studies across negotiation scenarios.
        
        Example:
            >>> engine = GameEngine()
            >>> available = engine.get_available_games()
            >>> print(available)
            ['company_car', 'resource_allocation', 'integrative_negotiations']
        
        Note:
            This method is called automatically during initialization and should
            not typically be called directly. Use register_game_type() to add
            custom game implementations.
        """
        self.register_game_type("company_car", CompanyCarGame)
        self.register_game_type("resource_allocation", ResourceAllocationGame)
        self.register_game_type("integrative_negotiations", IntegrativeNegotiationsGame)

    def register_game_type(self, game_name: str, game_class: Type[BaseGame]):
        """
        Register a new game type with the engine for future instantiation.
        
        Adds a new game class to the registry, making it available for creation
        via create_game(). Validates that the provided class properly inherits
        from BaseGame to ensure interface compliance.
        
        Args:
            game_name (str): The unique identifier for this game type. Used in
                create_game() calls and must be unique within the registry.
            game_class (Type[BaseGame]): The game class to register. Must inherit
                from BaseGame and implement all required abstract methods.
        
        Raises:
            ValueError: If game_class does not inherit from BaseGame.
        
        Example:
            >>> from my_games import CustomNegotiationGame
            >>> engine = GameEngine()
            >>> engine.register_game_type("custom_game", CustomNegotiationGame)
            >>> "custom_game" in engine.get_available_games()
            True
        """
        if not issubclass(game_class, BaseGame):
            raise ValueError(f"Game class must inherit from BaseGame")

        self.registered_games[game_name] = game_class
        self.logger.info(f"Registered game type: {game_name}")

    def create_game(self, game_type: str, config: Dict[str, Any]) -> BaseGame:
        """
        Create a new game instance of the specified type with given configuration.
        
        Instantiates a registered game class with the provided configuration dictionary.
        The configuration is passed directly to the game's constructor and should contain
        all necessary parameters for that specific game type.
        
        Args:
            game_type (str): The registered name of the game type to create. Must be
                a key that exists in the registered_games registry.
            config (Dict[str, Any]): Configuration dictionary containing game-specific
                parameters. The exact structure depends on the target game type:
                - company_car: max_rounds, batna_decay_rate, etc.
                - resource_allocation: resource_limits, team_preferences, etc.
                - integrative_negotiations: issues, priorities, etc.
        
        Returns:
            BaseGame: A fully initialized game instance ready for play.
        
        Raises:
            ValueError: If game_type is not registered in the engine.
            TypeError: If config is missing required parameters for the game type.
        
        Example:
            >>> engine = GameEngine()
            >>> config = {"max_rounds": 5, "batna_decay_rate": 0.1}
            >>> game = engine.create_game("company_car", config)
            >>> game.max_rounds
            5
        """
        if game_type not in self.registered_games:
            raise ValueError(f"Unknown game type: {game_type}")

        game_class = self.registered_games[game_type]
        return game_class(config)

    def get_available_games(self) -> list:
        """
        Retrieve list of all registered game type identifiers.
        
        Returns the identifiers of all game types currently registered with
        the engine, including both built-in games and any custom games that
        have been added via register_game_type().
        
        Returns:
            list: List of string identifiers for all registered game types.
                These identifiers can be used with create_game() to instantiate
                specific game instances.
        
        Example:
            >>> engine = GameEngine()
            >>> games = engine.get_available_games()
            >>> print(games)
            ['company_car', 'resource_allocation', 'integrative_negotiations']
        
        Note:
            The returned list reflects the current state of the registry and
            will include any custom games registered after initialization.
        """
        return list(self.registered_games.keys())

    def get_game_info(self, game_type: str) -> Dict[str, Any]:
        """
        Retrieve detailed metadata information about a registered game type.
        
        Provides comprehensive introspection capabilities for registered games,
        returning class information, documentation, and metadata that can be
        used for debugging, documentation generation, or dynamic game analysis.
        
        Args:
            game_type (str): Identifier of the registered game type to inspect.
        
        Returns:
            Dict[str, Any]: Dictionary containing comprehensive game metadata:
                - name (str): The registered identifier for the game type
                - class (str): Name of the game class implementation
                - description (str): Class docstring or fallback description
        
        Example:
            >>> engine = GameEngine()
            >>> info = engine.get_game_info("company_car")
            >>> print(info["name"])
            'company_car'
            >>> print(info["class"])
            'CompanyCarGame'
        
        Raises:
            ValueError: If the specified game_type is not registered.
        
        Note:
            This method is primarily useful for debugging, testing, and
            documentation generation rather than normal game operation.
        """
        if game_type not in self.registered_games:
            raise ValueError(f"Unknown game type: {game_type}")

        game_class = self.registered_games[game_type]
        return {
            "name": game_type,
            "class": game_class.__name__,
            "description": game_class.__doc__ or "No description available"
        }
