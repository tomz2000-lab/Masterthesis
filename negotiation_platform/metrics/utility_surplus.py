"""
Utility Surplus Metric: Utility from Agreement - Utility of BATNA
"""
from typing import Dict, List, Any
from negotiation_platform.core.base_metric import BaseMetric
from negotiation_platform.games.base_game import GameResult, PlayerAction

class UtilitySurplusMetric(BaseMetric):
    """
    Calculates utility surplus: Utility from Agreement - Utility of BATNA
    Higher values indicate better negotiation outcomes compared to no-deal scenario
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Utility Surplus", config)

    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate utility surplus for each player with game-specific handling (Option 1)"""
        results = {}
        
        # Get game type to determine how to calculate surplus
        game_type = game_result.game_data.get("game_type", "unknown")
        final_scores = game_result.final_scores
        
        # Check if agreement was reached
        agreement_reached = game_result.game_data.get("agreement_reached", False)
        if not agreement_reached:
            # No agreement = no surplus (stayed at BATNA)
            for player_id in game_result.players:
                results[player_id] = 0.0
            return results

        for player_id in game_result.players:
            if player_id in final_scores:
                if game_type == "company_car":
                    # Company car game: final_scores are absolute utilities, need to subtract BATNA
                    raw_utility = final_scores[player_id]
                    batnas_at_agreement = game_result.game_data.get("batnas_at_agreement", {})
                    
                    if player_id in batnas_at_agreement:
                        batna = batnas_at_agreement[player_id]
                        surplus = raw_utility - batna
                    else:
                        # Fallback: treat raw utility as surplus if BATNA not available
                        surplus = raw_utility
                else:
                    # Unknown game type - try to detect BATNA and calculate surplus
                    raw_utility = final_scores[player_id]
                    batnas_at_agreement = game_result.game_data.get("batnas_at_agreement", {})
                    
                    if player_id in batnas_at_agreement:
                        batna = batnas_at_agreement[player_id]
                        surplus = raw_utility - batna
                        print(f"💰 [UTILITY SURPLUS] {player_id}: utility={raw_utility:.2f} - BATNA={batna:.2f} = surplus={surplus:.2f}")

                    else:
                        # Fallback: treat raw utility as surplus if BATNA not available
                        surplus = raw_utility
                        print(f"💰 [UTILITY SURPLUS] {player_id}: No BATNA found, using raw utility={surplus:.2f}")

                        
                results[player_id] = surplus
            else:
                # If player not in final scores, surplus is 0
                results[player_id] = 0.0

        return results

    def get_description(self) -> str:
        return """
        Utility Surplus measures how much better a player did compared to their BATNA.
        Formula: Utility from Agreement - Utility of BATNA

        For all game types: Calculates final_utility - BATNA to determine surplus
        - company_car: Subtracts BATNA from absolute utility values
        - integrative_negotiations: Subtracts BATNA from point-based utilities  
        - resource_allocation: Subtracts BATNA from resource-based utilities

        Positive values: Player did better than their best alternative
        Zero: Player achieved exactly their BATNA outcome (or no agreement)
        Negative values: Player did worse than their BATNA (bad negotiation)
        """