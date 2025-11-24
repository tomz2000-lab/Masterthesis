"""
Base Metric Interface
====================

Defines the abstract interface for all negotiation performance metrics within
the platform. This interface enables plug-and-play metric integration with
consistent API and type safety across all metric implementations.

Key Features:
    - Abstract base class ensuring consistent metric API
    - Type-safe metric calculation with standardized inputs
    - Configurable metrics with parameter support
    - Self-documenting metrics with description methods
    - Plug-and-play architecture for custom metric development

Architecture:
    All metrics inherit from BaseMetric and implement the abstract methods.
    This ensures consistent behavior and enables dynamic metric registration
    and calculation within the MetricsCalculator system.

Metric Types:
    The platform supports various metric categories:
    - Utility Metrics: Measure player value extraction and satisfaction
    - Efficiency Metrics: Analyze negotiation process effectiveness
    - Strategic Metrics: Evaluate tactical decision-making quality
    - Behavioral Metrics: Assess communication and interaction patterns
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from negotiation_platform.games.base_game import GameResult, PlayerAction

class BaseMetric(ABC):
    """
    Abstract base class defining the interface for all negotiation performance metrics.
    
    The BaseMetric class provides the foundation for all metric implementations within
    the negotiation platform. It defines a consistent API that enables plug-and-play
    metric integration and ensures type safety across the metrics system.
    
    Design Pattern:
        Implements the Template Method pattern where concrete metrics provide specific
        calculation logic while inheriting common infrastructure and interface methods.
        This enables consistent metric behavior and easy integration with the
        MetricsCalculator system.
    
    Key Responsibilities:
        - Define abstract interface for metric calculation
        - Provide common metric infrastructure (naming, configuration)
        - Ensure type safety with standardized input/output types
        - Enable self-documentation through description methods
        - Support configurable metrics with parameter dictionaries
    
    Attributes:
        metric_name (str): Human-readable name for this metric.
        config (Dict[str, Any]): Configuration parameters for metric behavior.
    
    Implementation Requirements:
        Subclasses must implement:
        - calculate(): Core metric computation logic
        - get_description(): Human-readable metric explanation
    
    Example:
        >>> class CustomMetric(BaseMetric):
        ...     def calculate(self, game_result, actions_history):
        ...         return {player: 1.0 for player in game_result.players}
        ...     def get_description(self):
        ...         return "Example custom metric"
        >>> metric = CustomMetric("custom", {"param": "value"})
        >>> result = metric.calculate(game_result, actions_history)
    
    Note:
        This is an abstract class and cannot be instantiated directly. Use concrete
        implementations like UtilitySurplusMetric, FeasibilityMetric, etc.
    """

    def __init__(self, metric_name: str, config: Dict[str, Any] = None):
        """
        Initialize a new metric instance with name and configuration.
        
        Sets up the basic infrastructure for a metric implementation including
        the human-readable name and any configuration parameters needed for
        metric calculation.
        
        Args:
            metric_name (str): Human-readable name for this metric. Used in reports,
                logging, and metric identification. Should be descriptive and unique.
            config (Dict[str, Any], optional): Configuration dictionary containing
                metric-specific parameters. Common parameters include:
                    - weights: Importance weights for different components
                    - thresholds: Cutoff values for categorical metrics
                    - normalization: Scaling parameters for metric values
                    - enabled_features: Feature flags for metric behavior
                Defaults to empty dictionary if not provided.
        
        Example:
            >>> config = {"weight": 0.7, "threshold": 0.5}
            >>> metric = CustomMetric("utility_surplus", config)
            >>> print(metric.metric_name)
            'utility_surplus'
            >>> print(metric.config['weight'])
            0.7
        
        Note:
            Configuration parameters are metric-specific and should be documented
            in each concrete metric implementation.
        """
        self.metric_name = metric_name
        self.config = config or {}

    @abstractmethod
    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """
        Calculate the metric value for each player based on game outcome and actions.
        
        This is the core method that concrete metrics must implement. It analyzes
        the final game state and complete action history to compute metric values
        for all players who participated in the negotiation.
        
        Args:
            game_result (GameResult): Complete game outcome containing:
                - game_id: Unique identifier for the game session
                - players: List of all player identifiers
                - winner: Winning player(s) if applicable
                - final_scores: Player scores from game-specific scoring
                - total_rounds: Number of rounds played
                - game_data: Complete final game state
                - success: Whether the game concluded successfully
            
            actions_history (List[PlayerAction]): Chronological list of all actions
                taken during the negotiation. Each PlayerAction contains:
                - player_id: Which player took the action
                - action_type: Type of action (proposal, acceptance, etc.)
                - action_data: Action-specific data and parameters
                - timestamp: When the action was taken
                - round_number: Which round the action occurred in
        
        Returns:
            Dict[str, float]: Dictionary mapping each player ID to their metric value.
                All players from game_result.players must be included in the result.
                Values should be normalized to a consistent scale when possible.
        
        Implementation Guidelines:
            - Handle edge cases gracefully (no actions, early termination, etc.)
            - Return consistent value ranges for comparison across games
            - Use configuration parameters to customize calculation behavior
            - Provide meaningful values even for unsuccessful negotiations
            - Consider both final outcomes and process quality in calculations
        
        Example:
            >>> def calculate(self, game_result, actions_history):
            ...     values = {}
            ...     for player in game_result.players:
            ...         # Calculate player-specific metric value
            ...         values[player] = self._compute_player_value(player, game_result)
            ...     return values
        
        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Provide a human-readable description of what this metric measures.
        
        Returns a clear, concise explanation of the metric's purpose, calculation
        method, and interpretation. This description is used in reports, documentation,
        and user interfaces to help users understand metric meanings.
        
        Returns:
            str: Human-readable description explaining:
                - What aspect of negotiation performance the metric measures
                - How the metric is calculated (high-level approach)
                - How to interpret the metric values (higher/lower is better)
                - Any important limitations or assumptions
        
        Description Guidelines:
            - Use clear, non-technical language when possible
            - Explain the business/strategic meaning, not just the calculation
            - Mention the scale and interpretation of values
            - Include any important caveats or limitations
            - Keep descriptions concise but informative (1-3 sentences)
        
        Example:
            >>> def get_description(self):
            ...     return ("Measures the utility surplus each player achieved "
            ...             "compared to their BATNA (Best Alternative to Negotiated Agreement). "
            ...             "Higher values indicate better negotiation outcomes for the player.")
        
        Common Metric Categories:
            - Utility Metrics: "Measures player value extraction and satisfaction"
            - Efficiency Metrics: "Evaluates negotiation process effectiveness"
            - Strategic Metrics: "Assesses tactical decision-making quality"
            - Behavioral Metrics: "Analyzes communication and interaction patterns"
        
        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        pass

    def get_name(self) -> str:
        """
        Return the human-readable name of this metric.
        
        Provides access to the metric's display name as specified during initialization.
        This name is used for identification in reports, logging, metric registration,
        and user interfaces.
        
        Returns:
            str: The human-readable name of this metric as provided during initialization.
        
        Example:
            >>> metric = CustomMetric("utility_surplus", {})
            >>> print(metric.get_name())
            'utility_surplus'
        
        Note:
            This is a simple accessor method that returns the metric_name attribute.
            The name is set during initialization and remains constant throughout
            the metric's lifecycle.
        """
        return self.metric_name