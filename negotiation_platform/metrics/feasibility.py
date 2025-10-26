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

        # Extract common data that all game types need
        private_info = game_result.game_data.get('private_info', {})
        decay_rate = game_result.game_data.get('batna_decay_rate', 0.015)
        current_round = game_result.game_data.get('current_round', 1)
        game_type = game_result.game_data.get('game_type', 'unknown')

        # Game-specific feasibility logic
        if game_type == 'company_car':
            # For price bargaining (company car): check if agreed price meets decayed BATNAs
            agreed_price = game_result.game_data.get('agreed_price', 0)

            for player_id in game_result.players:
                if player_id in private_info:
                    player_info = private_info[player_id]
                    raw_batna = player_info.get('batna', 0.0)
                    role = player_info.get('role', '')

                    # Apply decay to the BATNA
                    decayed_batna = raw_batna * (1 - decay_rate) ** (current_round - 1)

                    # Check if the agreed price meets this player's constraints
                    if role == "buyer":
                        # Feasible if buyer paid <= their decayed BATNA
                        feasible = agreed_price <= decayed_batna
                    elif role == "seller":
                        # Feasible if seller received >= their decayed BATNA
                        feasible = agreed_price >= decayed_batna
                    else:
                        feasible = False  # Unknown role

                    results[player_id] = 1.0 if feasible else 0.0
                else:
                    results[player_id] = 0.0

        elif game_type == 'resource_allocation':
            # For resource allocation: check if final utilities meet decayed BATNAs
            final_utilities = game_result.game_data.get('final_utilities', {})

            for player_id in game_result.players:
                if player_id in private_info and player_id in final_utilities:
                    player_info = private_info[player_id]
                    raw_batna = player_info.get('batna', 0.0)
                    final_utility = final_utilities[player_id]

                    # Apply decay to the BATNA
                    decayed_batna = raw_batna * (1 - decay_rate) ** (current_round - 1)

                    # Feasible if final utility >= decayed BATNA
                    feasible = final_utility >= decayed_batna
                    results[player_id] = 1.0 if feasible else 0.0
                else:
                    results[player_id] = 0.0

        elif game_type == 'integrative_negotiations':
            # For integrative negotiations: check decayed BATNAs and game constraints
            final_utilities = game_result.game_data.get('final_utilities', {})
            constraints_met = game_result.game_data.get('constraints_met', True)

            for player_id in game_result.players:
                if player_id in private_info and player_id in final_utilities:
                    player_info = private_info[player_id]
                    raw_batna = player_info.get('batna', 0.0)
                    final_utility = final_utilities[player_id]

                    # Apply decay to the BATNA
                    decayed_batna = raw_batna * (1 - decay_rate) ** (current_round - 1)

                    # Check if the agreement meets the decayed BATNA
                    feasible_batna = final_utility >= decayed_batna

                    # Feasibility is true only if both BATNA and constraints are satisfied
                    feasible = feasible_batna and constraints_met
                    results[player_id] = 1.0 if feasible else 0.0
                else:
                    results[player_id] = 0.0

        else:
            # Unknown game type - assume not feasible
            for player_id in game_result.players:
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
        Feasibility measures whether the reached agreement is actually implementable
        given each player's constraints and time pressure (decaying BATNAs).
        Returns 1.0 if feasible, 0.0 if not feasible.

        Game-specific feasibility checks:
        
        Company Car (Price Bargaining):
        - Buyer: feasible if agreed_price <= decayed_batna
        - Seller: feasible if agreed_price >= decayed_batna
        
        Resource Allocation:
        - Feasible if final_utility >= decayed_batna for each player
        
        Integrative Negotiation:
        - Feasible if final_utility >= decayed_batna AND constraints_met
        
        BATNA Decay Formula: decayed_batna = raw_batna * (1 - 0.015) ^ (round - 1)
        This creates time pressure making agreements easier to reach over time.

        1.0: Agreement is fully feasible and can be implemented
        0.0: Agreement is not feasible or no agreement was reached
        """