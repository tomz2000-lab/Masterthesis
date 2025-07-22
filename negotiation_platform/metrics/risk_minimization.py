"""
Risk Minimization Metric: (worse than BATNA / all deals) * 100
"""
from typing import Dict, List, Any
from .base_metric import BaseMetric
from .base_game import GameResult, PlayerAction

class RiskMinimizationMetric(BaseMetric):
    """
    Calculates risk minimization: percentage of deals that are worse than BATNA
    Lower percentages indicate better risk management
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Risk Minimization", config)

    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate risk minimization percentage for each player"""
        results = {}

        # Extract all proposed deals from action history
        proposed_deals = []
        for action in actions_history:
            if action.action_type == "propose_trade":
                proposed_deals.append({
                    'proposer': action.player_id,
                    'offer': action.action_data.get('offer', {}),
                    'request': action.action_data.get('request', {}),
                    'round': action.round_number
                })

        batna_values = game_result.game_data.get('batna_values', {})

        for player_id in game_result.players:
            if player_id not in batna_values:
                results[player_id] = 0.0
                continue

            player_batna = batna_values[player_id]
            total_deals = 0
            worse_than_batna = 0

            # Analyze each proposed deal from this player's perspective
            for deal in proposed_deals:
                total_deals += 1

                # Simulate what this player's utility would be if this deal was accepted
                simulated_utility = self._simulate_deal_utility(
                    player_id, deal, game_result
                )

                if simulated_utility < player_batna:
                    worse_than_batna += 1

            # Calculate percentage
            if total_deals > 0:
                risk_percentage = (worse_than_batna / total_deals) * 100
            else:
                risk_percentage = 0.0

            results[player_id] = risk_percentage

        return results

    def _simulate_deal_utility(self, player_id: str, deal: Dict[str, Any],
                               game_result: GameResult) -> float:
        """Simulate what a player's utility would be if a deal was accepted"""
        # Get initial inventories
        initial_inventories = game_result.game_data.get('initial_inventories', {})

        if player_id not in initial_inventories:
            return 0.0

        # Simulate the trade
        simulated_inventory = initial_inventories[player_id].copy()

        if deal['proposer'] == player_id:
            # This player is proposing - they give 'offer' and get 'request'
            offer = deal['offer']
            request = deal['request']

            for resource, amount in offer.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) - amount

            for resource, amount in request.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) + amount
        else:
            # This player is receiving the proposal - they give 'request' and get 'offer'
            offer = deal['offer']  # What they would receive
            request = deal['request']  # What they would give

            for resource, amount in request.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) - amount

            for resource, amount in offer.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) + amount

        # Calculate utility with simulated inventory
        return self._calculate_utility(player_id, simulated_inventory, game_result.players)

    def _calculate_utility(self, player_id: str, resources: Dict[str, int],
                          all_players: List[str]) -> float:
        """Calculate utility for a player (matches game logic)"""
        # Default utility function - should match the game's utility calculation
        if player_id == all_players[0]:  # Player A prefers X
            return resources.get('X', 0) * 2.0 + resources.get('Y', 0) * 0.5
        else:  # Player B prefers Y
            return resources.get('X', 0) * 0.5 + resources.get('Y', 0) * 2.0

    def get_description(self) -> str:
        return """
        Risk Minimization measures the percentage of proposed deals that would be worse than BATNA.
        Formula: (Number of deals worse than BATNA / Total number of deals) * 100

        0%: All proposed deals were better than BATNA (excellent risk management)
        50%: Half of deals were risky (moderate risk management)  
        100%: All deals were worse than BATNA (poor risk management)
        """