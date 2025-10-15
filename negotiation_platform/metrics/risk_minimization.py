"""
Risk Minimization Metric: (worse than BATNA / all deals) * 100
"""
from typing import Dict, List, Any
from negotiation_platform.core.base_metric import BaseMetric
from negotiation_platform.games.base_game import GameResult, PlayerAction

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

        # DEBUG: Print actions_history to understand structure
        print(f"🔍 [RISK DEBUG] Total actions: {len(actions_history)}")
        for i, action in enumerate(actions_history[:5]):  # Show first 5 actions
            print(f"🔍 [RISK DEBUG] Action {i}: type={getattr(action, 'action_type', 'N/A')}, player={getattr(action, 'player_id', 'N/A')}, data={getattr(action, 'action_data', 'N/A')}")

        # Check game type to determine risk calculation logic
        game_type = game_result.game_data.get('game_type', 'unknown')
        print(f"🔍 [RISK DEBUG] Game type: {game_type}")
        
        if game_type == 'resource_allocation':
            # For resource allocation: check proposals against BATNA
            return self._calculate_resource_allocation_risk(game_result, actions_history)
        
        elif game_type == 'integrative_negotiations':
            # For integrative negotiations: check proposals against BATNA
            return self._calculate_integrative_risk(game_result, actions_history)
        
        # Original price bargaining logic with time-decayed BATNAs
        # For price bargaining: analyze offers made by each player
        player_offers = {}
        offer_rounds = {}  # Track which round each offer was made
        
        for action in actions_history:
            if hasattr(action, 'action_type') and action.action_type == "offer":
                player_id = getattr(action, 'player_id', None)
                action_data = getattr(action, 'action_data', {})
                price = action_data.get('price', 0)
                
                # Use the actual round number from the action
                action_round = getattr(action, 'round_number', 1)
                
                if player_id and player_id not in player_offers:
                    player_offers[player_id] = []
                    offer_rounds[player_id] = []
                    
                if player_id:
                    player_offers[player_id].append(price)
                    offer_rounds[player_id].append(action_round)

        print(f"🔍 [RISK DEBUG] Player offers: {player_offers}")
        print(f"🔍 [RISK DEBUG] Offer rounds: {offer_rounds}")

        # Get BATNA values and decay rates for time-adjusted calculation
        private_info = game_result.game_data.get('private_info', {})
        game_config = game_result.game_data.get('game_config', {})
        
        # Try to get decay rates from game config
        batna_decay = game_config.get('batna_decay', {'buyer': 0.015, 'seller': 0.015})
        print(f"🔍 [RISK DEBUG] BATNA decay: {batna_decay}")

        for player_id in game_result.players:
            if player_id not in private_info:
                results[player_id] = 0.0
                continue

            player_info = private_info[player_id]
            base_batna = player_info.get('batna', 0.0)
            role = player_info.get('role', '')
            
            # Get offers made by this player
            offers = player_offers.get(player_id, [])
            rounds = offer_rounds.get(player_id, [])
            
            print(f"🔍 [RISK DEBUG] Player {player_id} ({role}): base_batna={base_batna}, offers={offers}")
            
            if not offers:
                results[player_id] = 0.0
                continue

            risky_offers = 0
            total_offers = len(offers)

            for i, offer_price in enumerate(offers):
                round_num = rounds[i] if i < len(rounds) else 1
                
                # Calculate time-decayed BATNA for this round
                decay_rate = batna_decay.get(role, 0.015)
                current_batna = base_batna * ((1 - decay_rate) ** (round_num - 1))
                
                print(f"🔍 [RISK DEBUG] Round {round_num}: offer={offer_price}, current_batna={current_batna:.2f}")
                
                # Check if offer is risky (worse than current BATNA for this player)
                is_risky = False
                if role == "buyer":
                    # For buyer: risky if offering more than BATNA
                    if offer_price > current_batna:
                        risky_offers += 1
                        is_risky = True
                else:  # seller
                    # For seller: risky if offering less than BATNA
                    if offer_price < current_batna:
                        risky_offers += 1
                        is_risky = True
                
                print(f"🔍 [RISK DEBUG] Offer {offer_price} is {'RISKY' if is_risky else 'SAFE'} for {role}")

            # Calculate risk percentage (lower is better)
            risk_percentage = (risky_offers / total_offers) * 100 if total_offers > 0 else 0.0
            results[player_id] = risk_percentage
            print(f"🔍 [RISK DEBUG] Player {player_id} final risk: {risk_percentage}% ({risky_offers}/{total_offers})")

        return results

        return results

    def _calculate_integrative_risk(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate risk for integrative negotiation games"""
        results = {}

        # Use time-decayed BATNA values from agreement, not static private_info values
        batnas_at_agreement = game_result.game_data.get('batnas_at_agreement', {})

        # For integrative negotiations, check if final utility is better than BATNA
        final_utilities = game_result.final_scores

        for player_id in game_result.players:
            if player_id not in batnas_at_agreement or player_id not in final_utilities:
                results[player_id] = 0.0
                continue

            current_batna = batnas_at_agreement[player_id]
            final_utility = final_utilities[player_id]

            # Risk: 100% if all deals >= BATNA, 0% if all deals < BATNA
            results[player_id] = 100.0 if final_utility >= current_batna else 0.0

        return results

    def _calculate_resource_allocation_risk(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate risk for resource allocation games"""
        results = {}

        # Use time-decayed BATNA values from agreement, not static private_info values
        batnas_at_agreement = game_result.game_data.get('batnas_at_agreement', {})

        # For resource allocation, check if final allocation is better than BATNA
        final_utilities = game_result.final_scores

        for player_id in game_result.players:
            if player_id not in batnas_at_agreement or player_id not in final_utilities:
                results[player_id] = 0.0
                continue

            current_batna = batnas_at_agreement[player_id]
            final_utility = final_utilities[player_id]

            # Risk: 100% if all deals >= BATNA, 0% if all deals < BATNA
            results[player_id] = 100.0 if final_utility >= current_batna else 0.0

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