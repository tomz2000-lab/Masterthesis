"""
Utility Surplus Metric: Utility from Agreement - Utility of BATNA
"""
from typing import Dict, List, Any
from .base_metric import BaseMetric
from .base_game import GameResult, PlayerAction

class UtilitySurplusMetric(BaseMetric):
    """
    Calculates utility surplus: Utility from Agreement - Utility of BATNA
    Higher values indicate better negotiation outcomes compared to no-deal scenario
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Utility Surplus", config)

    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate utility surplus for each player"""
        results = {}

        # Get BATNA values from game data
        batna_values = game_result.game_data.get('batna_values', {})
        final_scores = game_result.final_scores

        for player_id in game_result.players:
            if player_id in batna_values and player_id in final_scores:
                utility_from_agreement = final_scores[player_id]
                utility_of_batna = batna_values[player_id]

                # Calculate surplus
                surplus = utility_from_agreement - utility_of_batna
                results[player_id] = surplus
            else:
                # If no agreement was reached, surplus is 0 (stayed at BATNA)
                results[player_id] = 0.0

        return results

    def get_description(self) -> str:
        return """
        Utility Surplus measures how much better a player did compared to their BATNA.
        Formula: Utility from Agreement - Utility of BATNA

        Positive values: Player did better than their best alternative
        Zero: Player achieved exactly their BATNA outcome  
        Negative values: Player did worse than their BATNA (bad negotiation)
        """