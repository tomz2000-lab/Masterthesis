"""
Base game interface for plug-and-play game integration
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class GameState(Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PlayerAction:
    """Represents a player action in the game"""
    player_id: str
    action_type: str
    action_data: Dict[str, Any]
    timestamp: float
    round_number: int

@dataclass
class GameResult:
    """Represents the final result of a game"""
    game_id: str
    players: List[str]
    winner: Optional[str]
    final_scores: Dict[str, float]
    total_rounds: int
    game_data: Dict[str, Any]
    success: bool

class BaseGame(ABC):
    """Abstract base class for all negotiation games"""

    def __init__(self, game_id: str, config: Dict[str, Any]):
        self.game_id = game_id
        self.config = config
        self.state = GameState.WAITING
        self.players: List[str] = []
        self.current_round = 0
        self.max_rounds = config.get('max_rounds', 10)
        self.actions_history: List[PlayerAction] = []
        self.game_data: Dict[str, Any] = {}

    @abstractmethod
    def initialize_game(self, players: List[str]) -> bool:
        """Initialize the game with given players"""
        pass

    @abstractmethod
    def is_valid_action(self, player: str, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Check if an action is valid in current game state"""
        pass

    @abstractmethod
    def process_action(self, action: PlayerAction) -> Dict[str, Any]:
        """Process a player action and update game state"""
        pass

    @abstractmethod
    def check_end_conditions(self) -> bool:
        """Check if the game should end"""
        pass

    @abstractmethod
    def calculate_scores(self) -> Dict[str, float]:
        """Calculate final scores for all players"""
        pass

    @abstractmethod
    def get_game_prompt(self, player_id: str) -> str:
        """Get the current game prompt for a specific player"""
        pass

    def add_action(self, action: PlayerAction):
        """Add an action to the game history"""
        self.actions_history.append(action)

    def get_game_info(self) -> Dict[str, Any]:
        """Get current game information"""
        return {
            "game_id": self.game_id,
            "state": self.state.value,
            "players": self.players,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "config": self.config
        }