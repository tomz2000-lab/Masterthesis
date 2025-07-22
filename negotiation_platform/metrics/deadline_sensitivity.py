"""
Deadline Sensitivity Metric: Average Surplus Loss per round compared to first round
"""
from typing import Dict, List, Any
from .base_metric import BaseMetric
from .base_game import GameResult, PlayerAction

class DeadlineSensitivityMetric(BaseMetric):
    """
    Calculates deadline sensitivity: how much surplus is lost per round compared to first round
    Measures how deadline pressure affects negotiation performance
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Deadline Sensitivity", config)

    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate deadline sensitivity for each player"""
        results = {}

        # Group actions by round
        actions_by_round = {}
        for action in actions_history:
            round_num = action.round_number
            if round_num not in actions_by_round:
                actions_by_round[round_num] = []
            actions_by_round[round_num].append(action)

        batna_values = game_result.game_data.get('batna_values', {})

        for player_id in game_result.players:
            if player_id not in batna_values:
                results[player_id] = 0.0
                continue

            player_batna = batna_values[player_id]
            round_surpluses = []

            # Calculate surplus for each round's best offer
            for round_num in sorted(actions_by_round.keys()):
                round_actions = actions_by_round[round_num]

                best_surplus_this_round = 0.0

                # Find the best offer for this player in this round
                for action in round_actions:
                    if action.action_type == "propose_trade":
                        simulated_utility = self._simulate_deal_utility(
                            player_id, {
                                'proposer': action.player_id,
                                'offer': action.action_data.get('offer', {}),
                                'request': action.action_data.get('request', {})
                            },
                            game_result
                        )

                        surplus = simulated_utility - player_batna
                        best_surplus_this_round = max(best_surplus_this_round, surplus)

                round_surpluses.append(best_surplus_this_round)

            # Calculate sensitivity (decline from first round)
            if len(round_surpluses) > 1:
                first_round_surplus = round_surpluses[0] if round_surpluses[0] > 0 else 1.0

                # Calculate average decline per round
                total_decline = 0.0
                valid_comparisons = 0

                for i in range(1, len(round_surpluses)):
                    decline = first_round_surplus - round_surpluses[i]
                    total_decline += decline
                    valid_comparisons += 1

                if valid_comparisons > 0:
                    avg_surplus_loss_per_round = total_decline / valid_comparisons
                    sensitivity = (avg_surplus_loss_per_round / first_round_surplus) * 100
                else:
                    sensitivity = 0.0
            else:
                sensitivity = 0.0

            results[player_id] = sensitivity

        return results

    def _simulate_deal_utility(self, player_id: str, deal: Dict[str, Any],
                               game_result: GameResult) -> float:
        """Simulate utility from a deal (reused from RiskMinimization)"""
        initial_inventories = game_result.game_data.get('initial_inventories', {})

        if player_id not in initial_inventories:
            return 0.0

        simulated_inventory = initial_inventories[player_id].copy()

        if deal['proposer'] == player_id:
            offer = deal['offer']
            request = deal['request']

            for resource, amount in offer.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) - amount

            for resource, amount in request.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) + amount
        else:
            offer = deal['offer']
            request = deal['request']

            for resource, amount in request.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) - amount

            for resource, amount in offer.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) + amount

        return self._calculate_utility(player_id, simulated_inventory, game_result.players)

    def _calculate_utility(self, player_id: str, resources: Dict[str, int],
                          all_players: List[str]) -> float:
        """Calculate utility matching game logic"""
        if player_id == all_players[0]:
            return resources.get('X', 0) * 2.0 + resources.get('Y', 0) * 0.5
        else:
            return resources.get('X', 0) * 0.5 + resources.get('Y', 0) * 2.0

    def get_description(self) -> str:
        return """
        Deadline Sensitivity measures how negotiation performance degrades as deadline approaches.
        Formula: Average surplus loss per round compared to first round (as percentage)

        0%: No degradation in offers over time (deadline insensitive)
        25%: Moderate decline in offer quality as deadline approaches  
        50%+ High sensitivity to deadline pressure (significant performance decline)
        """