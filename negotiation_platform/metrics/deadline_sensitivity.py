"""
Deadline Sensitivity Metric: Average Surplus Loss per round compared to first round
"""
from typing import Dict, List, Any
from negotiation_platform.core.base_metric import BaseMetric
from negotiation_platform.games.base_game import GameResult, PlayerAction

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

        # Check game type to determine sensitivity calculation
        game_type = game_result.game_data.get('game_type', 'unknown')
        
        if game_type == 'resource_allocation':
            # For resource allocation: analyze how proposals changed over rounds
            return self._calculate_resource_allocation_sensitivity(game_result, actions_history)

        elif game_type == 'integrative_negotiations':
            # For integrative negotiations: analyze deadline pressure effects
            return self._calculate_integrative_sensitivity(game_result, actions_history)

        # Original price bargaining logic
        # Group actions by round
        actions_by_round = {}
        for action in actions_history:
            round_num = action.round_number
            if round_num not in actions_by_round:
                actions_by_round[round_num] = []
            actions_by_round[round_num].append(action)

        # Get BATNA values from private_info
        private_info = game_result.game_data.get('private_info', {})

        for player_id in game_result.players:
            if player_id not in private_info:
                results[player_id] = 0.0
                continue

            player_info = private_info[player_id]
            player_batna = player_info.get('batna', 0.0)
            role = player_info.get('role', '')
            
            round_surpluses = []

            # Calculate potential surplus for each round based on offers seen
            for round_num in sorted(actions_by_round.keys()):
                round_actions = actions_by_round[round_num]
                
                best_surplus_this_round = 0.0

                # Find the best potential surplus for this player in this round
                for action in round_actions:
                    if action.action_type == "offer" and action.player_id != player_id:
                        # This is an offer from the other player
                        offer_price = action.action_data.get('price', 0)
                        
                        # Calculate surplus if this player accepted this offer
                        if role == "buyer":
                            # Buyer surplus: BATNA - offer price (if positive)
                            surplus = max(0, player_batna - offer_price)
                        else:  # seller
                            # Seller surplus: offer price - BATNA (if positive)
                            surplus = max(0, offer_price - player_batna)
                        
                        best_surplus_this_round = max(best_surplus_this_round, surplus)

                round_surpluses.append(best_surplus_this_round)

            # Calculate sensitivity: how much surplus degraded from first to last round
            if len(round_surpluses) >= 2:
                first_round_surplus = round_surpluses[0]
                last_round_surplus = round_surpluses[-1]
                
                if first_round_surplus > 0:
                    # Sensitivity as percentage loss per round
                    total_loss = first_round_surplus - last_round_surplus
                    num_rounds = len(round_surpluses) - 1
                    sensitivity = (total_loss / first_round_surplus) / num_rounds * 100
                else:
                    sensitivity = 0.0
            else:
                sensitivity = 0.0

            results[player_id] = max(0.0, sensitivity)

        return results

    def _calculate_integrative_sensitivity(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate deadline sensitivity for integrative negotiation games"""
        results = {}
        
        # For integrative negotiations: if agreement reached in later rounds,
        # it suggests deadline pressure was effective
        total_rounds = len(set(action.round_number for action in actions_history))
        agreement_reached = game_result.game_data.get('agreement_reached', False)
        
        for player_id in game_result.players:
            if agreement_reached and total_rounds > 1:
                # Sensitivity based on how late the agreement came
                # Consider BATNA decay which makes later agreements more costly
                final_round = max(action.round_number for action in actions_history)
                
                # Higher sensitivity for later agreements due to BATNA decay
                sensitivity = (final_round - 1) / (total_rounds - 1) * 100 if total_rounds > 1 else 0.0
                
                # Integrative negotiations have BATNA decay, so add that factor
                batna_decay_rate = 0.02  # 2% per round as specified
                decay_factor = (final_round - 1) * batna_decay_rate * 100
                sensitivity = min(100.0, sensitivity + decay_factor)
                
                results[player_id] = sensitivity
            else:
                # No agreement or single round = no deadline sensitivity detected
                results[player_id] = 0.0
                
        return results

    def _calculate_resource_allocation_sensitivity(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate deadline sensitivity for resource allocation games"""
        results = {}
        
        # For resource allocation: if agreement reached in later rounds, 
        # it suggests deadline pressure was effective
        total_rounds = len(set(action.round_number for action in actions_history))
        agreement_reached = game_result.game_data.get('agreement_reached', False)
        
        for player_id in game_result.players:
            if agreement_reached and total_rounds > 1:
                # Sensitivity based on how late the agreement came
                # Earlier agreement = lower sensitivity to deadline pressure
                final_round = max(action.round_number for action in actions_history)
                sensitivity = (final_round - 1) / (total_rounds - 1) * 100 if total_rounds > 1 else 0.0
                results[player_id] = sensitivity
            else:
                # No agreement or single round = no deadline sensitivity detected
                results[player_id] = 0.0
                
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