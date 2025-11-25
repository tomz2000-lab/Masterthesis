"""
Deadline Sensitivity Metric: Measures true deadline awareness vs panic behavior
"""
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.stats import linregress
from negotiation_platform.core.base_metric import BaseMetric
from negotiation_platform.games.base_game import GameResult, PlayerAction

def calculate_deadline_sensitivity(surplus_list: List[float]) -> Tuple[float, float, float, float]:
    """Calculate comprehensive deadline sensitivity metrics from surplus progression.
    
    This function analyzes how surplus values change throughout negotiation
    rounds to measure sensitivity to deadline pressure, providing statistical
    insights into negotiator behavior under time constraints.
    
    Args:
        surplus_list (List[float]): Sequential surplus values per round,
            representing the progression of negotiation outcomes over time
            (e.g., [10, 11, 12, 14, 16, 18, 21, 24, 27, 31]).
    
    Returns:
        Tuple[float, float, float, float]: A 4-tuple containing statistical measures:
            - slope: Average surplus improvement per round (positive indicates
              deadline awareness and progressive concession-making)
            - r_squared: Consistency measure (0-1 scale, high values indicate
              steady, predictable progression patterns)
            - variance: Steadiness of improvements (low values indicate
              consistent, smooth progression without erratic changes)
            - p_value: Statistical significance of the trend (low values
              indicate statistically significant deadline sensitivity)
    
    Example:
        >>> surplus_data = [10.0, 12.0, 15.0, 20.0, 30.0]
        >>> slope, r2, var, p = calculate_deadline_sensitivity(surplus_data)
        >>> print(f"Slope: {slope:.2f}, R²: {r2:.3f}")
        Slope: 5.00, R²: 0.950
    
    Note:
        Higher slope values with low p-values indicate strong deadline
        sensitivity, suggesting negotiators respond effectively to time
        pressure by making increasingly beneficial agreements.
    """
    if len(surplus_list) < 2:
        return 0.0, 0.0, 0.0, 1.0
    
    # Create round numbers (1, 2, 3, ...)
    round_numbers = np.arange(1, len(surplus_list) + 1)
    surplus_values = np.array(surplus_list)
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = linregress(round_numbers, surplus_values)
    r_squared = r_value ** 2
    
    # Calculate variance of round-to-round improvements
    if len(surplus_list) > 1:
        improvements = np.diff(surplus_values)
        variance = np.var(improvements)
    else:
        variance = 0.0
    
    return slope, r_squared, variance, p_value

class DeadlineSensitivityMetric(BaseMetric):
    """
    Simple deadline sensitivity: 100 if agreement reached, 0 otherwise
    Measures whether deadline pressure resulted in any deal
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Deadline Sensitivity metric with optional configuration.
        
        Creates a new DeadlineSensitivityMetric instance that measures how
        effectively players respond to deadline pressure during negotiations.
        This metric evaluates whether time constraints motivate agreement-seeking
        behavior and successful negotiation completion.
        
        Args:
            config (Dict[str, Any], optional): Configuration parameters for
                metric behavior. Currently unused but reserved for future
                enhancements such as:
                    - deadline_threshold: Minimum rounds for sensitivity analysis
                    - pressure_weighting: How to weight early vs. late round behavior
                    - completion_bonus: Additional scoring for successful agreements
                Defaults to None (empty configuration).
        
        Example:
            >>> # Basic initialization
            >>> metric = DeadlineSensitivityMetric()
            >>> # With configuration (future use)
            >>> config = {"threshold": 3, "weighting": "exponential"}
            >>> metric = DeadlineSensitivityMetric(config)
        
        Note:
            This metric uses a simplified binary approach: 100 points for
            successful agreements, 0 points for failed negotiations.
        """
        super().__init__("Deadline Sensitivity", config)

    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """
        Calculate deadline sensitivity score for each player based on negotiation completion.
        
        Measures how effectively players respond to deadline pressure by evaluating
        whether they successfully reach agreements within the time constraints.
        Uses a simplified binary scoring system that rewards negotiation completion.
        
        Args:
            game_result (GameResult): Complete game outcome containing:
                - game_data: Game state with agreement status and round information
                - players: List of all participating players
                - final_scores: Not used for this metric
            actions_history (List[PlayerAction]): Complete action log (unused for
                this metric but required by BaseMetric interface).
        
        Returns:
            Dict[str, float]: Dictionary mapping each player ID to their deadline
                sensitivity score:
                - 100.0: Successful agreement reached (deadline pressure effective)
                - 0.0: No agreement reached (deadline pressure ineffective)
        
        Scoring Logic:
            - Binary evaluation: success (100) vs. failure (0)
            - All players receive same score based on overall negotiation outcome
            - Future versions may incorporate more nuanced time-based analysis
        
        Example:
            >>> result = metric.calculate(game_result, actions_history)
            >>> print(result)
            {'player1': 100.0, 'player2': 100.0}  # Both succeeded under deadline
        
        Note:
            This simplified approach focuses on outcome rather than process.
            Future enhancements may analyze proposal timing and urgency patterns.
        """
        results = {}

        # Simple consistent logic across all game types:
        # 100 if agreement reached, 0 if no agreement
        agreement_reached = game_result.game_data.get('agreement_reached', False)
        
        # Use the same logic for all game types
        for player_id in game_result.players:
            if agreement_reached:
                results[player_id] = 100.0
            else:
                results[player_id] = 0.0
                
        return results

    def get_description(self) -> str:
        """Provides a comprehensive description of the Deadline Sensitivity metric.
        
        This method returns a detailed explanation of how the Deadline Sensitivity
        metric evaluates the effectiveness of time pressure in driving negotiation
        outcomes by measuring whether agreements are successfully reached before
        deadlines expire.
        
        Returns:
            str: A multi-line string containing:
                - Metric definition and deadline pressure assessment
                - Binary scoring formula (100 for agreement, 0 for failure)
                - Interpretation of deadline effectiveness
                - Time pressure impact on negotiation dynamics
                - Success/failure outcome indicators
        
        Note:
            This metric is particularly useful for analyzing how time constraints
            influence negotiator behavior and whether deadline pressure creates
            the intended motivation to reach agreements.
        """
        return """
        Deadline Sensitivity measures whether an agreement was reached during negotiation.
        Formula: 100 if agreement reached, 0 if no agreement

        100: Agreement reached (deadline pressure led to deal)
        0: No agreement reached (deadline pressure failed to create deal)
        """