"""
Base metrics interface for plug-and-play metrics integration
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from negotiation_platform.games.base_game import GameResult, PlayerAction

class BaseMetric(ABC):
    """Abstract base class for all negotiation metrics"""

    def __init__(self, metric_name: str, config: Dict[str, Any] = None):
        self.metric_name = metric_name
        self.config = config or {}

    @abstractmethod
    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate the metric for each player"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get description of what this metric measures"""
        pass

    def get_name(self) -> str:
        """Get metric name"""
        return self.metric_name