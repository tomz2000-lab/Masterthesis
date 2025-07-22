"""
Feasibility Metric: Is the agreement even possible
"""
from typing import Dict, List, Any
from .base_metric import BaseMetric
from .base_game import GameResult, PlayerAction

class FeasibilityMetric(BaseMetric):
    """
    Calculates feasibility: whether the agreement is actually possible given constraints
    Binary metric: 1.0 if feasible, 0.0 if not feasible
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Feasibility", config)

    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate feasibility for each player's perspective"""
        results = {}

        # Check if an agreement was actually reached
        agreement_reached = 'agreement' in game_result.game_data

        if not agreement_reached:
            # No agreement reached - feasibility is 0 for all players
            for player_id in game_result.players:
                results[player_id] = 0.0
            return results

        # Get the final agreement details
        agreement = game_result.game_data['agreement']
        initial_inventories = game_result.game_data.get('initial_inventories', {})

        for player_id in game_result.players:
            feasible = self._check_agreement_feasibility(
                player_id, agreement, initial_inventories
            )
            results[player_id] = 1.0 if feasible else 0.0

        return results

    def _check_agreement_feasibility(self, player_id: str, agreement: Dict[str, Any],
                                   initial_inventories: Dict[str, Dict[str, int]]) -> bool:
        """Check if the agreement is feasible for a specific player"""
        if player_id not in initial_inventories:
            return False

        player_inventory = initial_inventories[player_id].copy()
        trade = agreement['trade']
        proposer = agreement['proposer']

        # Determine what this player needs to give
        if player_id == proposer:
            # This player proposed the trade - they give 'offer'
            required_resources = trade['offer']
        else:
            # This player accepted the trade - they give 'request'
            required_resources = trade['request']

        # Check if player has enough resources
        for resource, amount in required_resources.items():
            available = player_inventory.get(resource, 0)
            if available < amount:
                return False  # Not enough resources

        # Additional feasibility checks can be added here:
        # - Time constraints
        # - External dependencies
        # - Regulatory constraints
        # - Physical limitations

        return True

    def get_description(self) -> str:
        return """
        Feasibility measures whether the reached agreement is actually implementable.
        Returns 1.0 if feasible, 0.0 if not feasible.

        Checks include:
        - Do players have sufficient resources to fulfill their commitments?
        - Are there any constraint violations?  
        - Is the agreement logically consistent?

        1.0: Agreement is fully feasible and can be implemented
        0.0: Agreement is not feasible or no agreement was reached
        """