from typing import Dict, Type, Any, Optional
import logging
from negotiation_platform.games.base_game import BaseGame
from negotiation_platform.games.price_bargaining import CompanyCarGame
from negotiation_platform.games.resource_exchange import ResourceAllocationGame
from negotiation_platform.games.integrative_negotiation import IntegrativeNegotiationsGame


class GameEngine:
    """Manages game registration and instantiation."""

    def __init__(self):
        self.registered_games: Dict[str, Type[BaseGame]] = {}
        self.logger = logging.getLogger(__name__) #Must become before next line in order to work
        self._register_default_games()

    def _register_default_games(self):
        """Register built-in games."""
        self.register_game_type("company_car", CompanyCarGame)
        self.register_game_type("resource_allocation", ResourceAllocationGame)
        self.register_game_type("integrative_negotiations", IntegrativeNegotiationsGame)

    def register_game_type(self, game_name: str, game_class: Type[BaseGame]):
        """Register a new game type."""
        if not issubclass(game_class, BaseGame):
            raise ValueError(f"Game class must inherit from BaseGame")

        self.registered_games[game_name] = game_class
        self.logger.info(f"Registered game type: {game_name}")

    def create_game(self, game_type: str, config: Dict[str, Any]) -> BaseGame:
        """Create game instance from configuration."""
        if game_type not in self.registered_games:
            raise ValueError(f"Unknown game type: {game_type}")

        game_class = self.registered_games[game_type]
        return game_class(config)

    def get_available_games(self) -> list:
        """Get list of registered game types."""
        return list(self.registered_games.keys())

    def get_game_info(self, game_type: str) -> Dict[str, Any]:
        """Get information about a specific game type."""
        if game_type not in self.registered_games:
            raise ValueError(f"Unknown game type: {game_type}")

        game_class = self.registered_games[game_type]
        return {
            "name": game_type,
            "class": game_class.__name__,
            "description": game_class.__doc__ or "No description available"
        }
