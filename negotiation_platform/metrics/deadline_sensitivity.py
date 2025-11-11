"""
Deadline Sensitivity Metric: Measures true deadline awareness vs panic behavior
"""
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.stats import linregress
from negotiation_platform.core.base_metric import BaseMetric
from negotiation_platform.games.base_game import GameResult, PlayerAction

def calculate_deadline_sensitivity(surplus_list: List[float]) -> Tuple[float, float, float, float]:
    """
    Calculate deadline sensitivity metrics from surplus progression.
    
    Args:
        surplus_list: List of surplus values, one per round (e.g., [10, 11, 12, 14, 16, 18, 21, 24, 27, 31])
    
    Returns:
        Tuple of (slope, r_squared, variance, p_value):
        - slope: Average surplus improvement per round (positive = deadline awareness)
        - r_squared: Consistency measure (0-1, high = steady pattern)
        - variance: Steadiness of improvements (low = consistent)
        - p_value: Statistical significance of the trend
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
        super().__init__("Deadline Sensitivity", config)

    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate deadline sensitivity for each player"""
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
        return """
        Deadline Sensitivity measures whether an agreement was reached during negotiation.
        Formula: 100 if agreement reached, 0 if no agreement

        100: Agreement reached (deadline pressure led to deal)
        0: No agreement reached (deadline pressure failed to create deal)
        """