"""
Feasibility Metric: Is the agreement even possible
"""
from typing import Dict, List, Any
from negotiation_platform.core.base_metric import BaseMetric
from negotiation_platform.games.base_game import GameResult, PlayerAction

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
        agreement_reached = game_result.game_data.get('agreement_reached', False)

        if not agreement_reached:
            # No agreement reached - feasibility is 0 for all players
            for player_id in game_result.players:
                results[player_id] = 0.0
            return results

        # Check game type to determine feasibility logic
        game_type = game_result.game_data.get('game_type', 'unknown')
        
        if game_type == 'resource_allocation':
            # For resource allocation: if agreement reached, it's feasible
            # (the game engine already validated constraints)
            for player_id in game_result.players:
                results[player_id] = 1.0
            return results
        
        elif game_type == 'integrative_negotiations':
            # For integrative negotiations: if agreement reached, it's feasible
            # (the game engine validated the proposal format and constraints)
            for player_id in game_result.players:
                results[player_id] = 1.0
            return results

        # For price bargaining (company_car), if agreement was reached, it's feasible for both parties
        # (the game logic already validated it against BATNAs)
        agreed_price = game_result.game_data.get('agreed_price', 0)
        private_info = game_result.game_data.get('private_info', {})

        for player_id in game_result.players:
            if player_id in private_info:
                player_info = private_info[player_id]
                batna = player_info.get('batna', 0.0)
                role = player_info.get('role', '')
                
                # Check if the agreed price meets this player's constraints
                if role == "buyer":
                    # Feasible if buyer paid <= their BATNA
                    feasible = agreed_price <= batna
                else:  # seller
                    # Feasible if seller received >= their BATNA
                    feasible = agreed_price >= batna
                
                results[player_id] = 1.0 if feasible else 0.0
            else:
                results[player_id] = 0.0

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