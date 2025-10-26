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
                
                # Check if offer is within BATNA (good risk management)
                is_within_batna = False
                if role == "buyer":
                    # For buyer: within BATNA if offering less than or equal to BATNA
                    if offer_price <= current_batna:
                        is_within_batna = True
                else:  # seller
                    # For seller: within BATNA if offering more than or equal to BATNA
                    if offer_price >= current_batna:
                        is_within_batna = True
                
                if is_within_batna:
                    risky_offers += 1
                
                print(f"🔍 [RISK DEBUG] Offer {offer_price} is {'WITHIN BATNA' if is_within_batna else 'OUTSIDE BATNA'} for {role}")

            # Calculate risk percentage: 100% if all offers within BATNA, 0% if all offers outside BATNA
            risk_percentage = (risky_offers / total_offers) * 100 if total_offers > 0 else 0.0
            results[player_id] = risk_percentage
            print(f"🔍 [RISK DEBUG] Player {player_id} final risk: {risk_percentage}% ({risky_offers}/{total_offers} offers within BATNA)")

        return results

        return results

    def _calculate_integrative_risk(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate risk for integrative negotiation games by analyzing individual proposals"""
        results = {}

        # Analyze individual proposals made by each player
        player_proposals = {}
        proposal_rounds = {}
        
        for action in actions_history:
            if hasattr(action, 'action_type') and action.action_type == "propose":
                player_id = getattr(action, 'player_id', None)
                action_data = getattr(action, 'action_data', {})
                action_round = getattr(action, 'round_number', 1)
                
                if player_id and player_id not in player_proposals:
                    player_proposals[player_id] = []
                    proposal_rounds[player_id] = []
                    
                if player_id:
                    player_proposals[player_id].append(action_data)
                    proposal_rounds[player_id].append(action_round)

        print(f"🔍 [INTEGRATIVE RISK DEBUG] Player proposals: {len(player_proposals)} players")
        print(f"🔍 [INTEGRATIVE RISK DEBUG] Proposal rounds: {proposal_rounds}")

        # Get BATNA values and decay rates
        private_info = game_result.game_data.get('private_info', {})
        game_config = game_result.game_data.get('game_config', {})
        batna_decay = game_config.get('batna_decay', {'IT': 0.015, 'MARKETING': 0.015})

        for player_id in game_result.players:
            if player_id not in private_info:
                results[player_id] = 0.0
                continue

            player_info = private_info[player_id]
            base_batna = player_info.get('batna', 0.0)
            role = player_info.get('role', '')
            
            proposals = player_proposals.get(player_id, [])
            rounds = proposal_rounds.get(player_id, [])
            
            print(f"🔍 [INTEGRATIVE RISK DEBUG] Player {player_id} ({role}): base_batna={base_batna}, proposals={len(proposals)}")
            
            if not proposals:
                results[player_id] = 0.0
                continue

            within_batna_count = 0
            total_proposals = len(proposals)

            for i, proposal_data in enumerate(proposals):
                round_num = rounds[i] if i < len(rounds) else 1
                
                # Calculate time-decayed BATNA for this round
                decay_rate = batna_decay.get(role, 0.015)
                current_batna = base_batna * ((1 - decay_rate) ** (round_num - 1))
                
                # Calculate utility for this proposal
                proposal_utility = self._calculate_integrative_utility(player_id, proposal_data, game_result)
                
                print(f"🔍 [INTEGRATIVE RISK DEBUG] Round {round_num}: proposal_utility={proposal_utility:.2f}, current_batna={current_batna:.2f}")
                
                # Check if proposal is within BATNA (good risk management)
                if proposal_utility >= current_batna:
                    within_batna_count += 1
                
                is_within = proposal_utility >= current_batna
                print(f"🔍 [INTEGRATIVE RISK DEBUG] Proposal utility {proposal_utility:.2f} is {'WITHIN BATNA' if is_within else 'OUTSIDE BATNA'} for {role}")

            # Calculate percentage of proposals within BATNA
            risk_percentage = (within_batna_count / total_proposals) * 100 if total_proposals > 0 else 0.0
            results[player_id] = risk_percentage
            print(f"🔍 [INTEGRATIVE RISK DEBUG] Player {player_id} final risk: {risk_percentage}% ({within_batna_count}/{total_proposals} proposals within BATNA)")

        return results

    def _calculate_resource_allocation_risk(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """Calculate risk for resource allocation games by analyzing individual proposals"""
        results = {}

        # Analyze individual proposals made by each player
        player_proposals = {}
        proposal_rounds = {}
        
        for action in actions_history:
            if hasattr(action, 'action_type') and action.action_type == "propose":
                player_id = getattr(action, 'player_id', None)
                action_data = getattr(action, 'action_data', {})
                action_round = getattr(action, 'round_number', 1)
                
                if player_id and player_id not in player_proposals:
                    player_proposals[player_id] = []
                    proposal_rounds[player_id] = []
                    
                if player_id:
                    player_proposals[player_id].append(action_data)
                    proposal_rounds[player_id].append(action_round)

        print(f"🔍 [RESOURCE RISK DEBUG] Player proposals: {len(player_proposals)} players")
        print(f"🔍 [RESOURCE RISK DEBUG] Proposal rounds: {proposal_rounds}")

        # Get BATNA values and decay rates
        private_info = game_result.game_data.get('private_info', {})
        game_config = game_result.game_data.get('game_config', {})
        batna_decay = game_config.get('batna_decay', {'DEVELOPMENT': 0.015, 'MARKETING': 0.015})

        for player_id in game_result.players:
            if player_id not in private_info:
                results[player_id] = 0.0
                continue

            player_info = private_info[player_id]
            base_batna = player_info.get('batna', 0.0)
            role = player_info.get('role', '')
            
            proposals = player_proposals.get(player_id, [])
            rounds = proposal_rounds.get(player_id, [])
            
            print(f"🔍 [RESOURCE RISK DEBUG] Player {player_id} ({role}): base_batna={base_batna}, proposals={len(proposals)}")
            
            if not proposals:
                results[player_id] = 0.0
                continue

            within_batna_count = 0
            total_proposals = len(proposals)

            for i, proposal_data in enumerate(proposals):
                round_num = rounds[i] if i < len(rounds) else 1
                
                # Calculate time-decayed BATNA for this round
                decay_rate = batna_decay.get(role, 0.015)
                current_batna = base_batna * ((1 - decay_rate) ** (round_num - 1))
                
                # Calculate utility for this proposal
                proposal_utility = self._simulate_deal_utility(player_id, proposal_data, game_result)
                
                print(f"🔍 [RESOURCE RISK DEBUG] Round {round_num}: proposal_utility={proposal_utility:.2f}, current_batna={current_batna:.2f}")
                
                # Check if proposal is within BATNA (good risk management)
                if proposal_utility >= current_batna:
                    within_batna_count += 1
                
                is_within = proposal_utility >= current_batna
                print(f"🔍 [RESOURCE RISK DEBUG] Proposal utility {proposal_utility:.2f} is {'WITHIN BATNA' if is_within else 'OUTSIDE BATNA'} for {role}")

            # Calculate percentage of proposals within BATNA
            risk_percentage = (within_batna_count / total_proposals) * 100 if total_proposals > 0 else 0.0
            results[player_id] = risk_percentage
            print(f"🔍 [RESOURCE RISK DEBUG] Player {player_id} final risk: {risk_percentage}% ({within_batna_count}/{total_proposals} proposals within BATNA)")

        return results

    def _simulate_deal_utility(self, player_id: str, deal: Dict[str, Any],
                               game_result: GameResult) -> float:
        """Simulate what a player's utility would be if a deal was accepted"""
        
        # Check if this is a simple GPU/CPU allocation game
        if 'gpu_hours' in deal and 'cpu_hours' in deal:
            # Simple resource allocation: calculate utility directly from allocation
            gpu_hours = deal.get('gpu_hours', 0)
            cpu_hours = deal.get('cpu_hours', 0)
            
            # Get player role to determine utility function
            private_info = game_result.game_data.get('private_info', {})
            if player_id in private_info:
                role = private_info[player_id].get('role', '')
                
                # Calculate utility based on role and allocation
                if role.upper() == 'DEVELOPMENT':
                    # Development prioritizes GPU: 8x + 6y
                    return 8 * gpu_hours + 6 * cpu_hours
                elif role.upper() == 'MARKETING':
                    # Marketing prioritizes CPU: 6x + 8y
                    return 6 * gpu_hours + 8 * cpu_hours
                else:
                    # Fallback: generic utility calculation
                    return gpu_hours + cpu_hours
            
            # Fallback if no role info
            return gpu_hours + cpu_hours
        
        # Legacy code for trading-based resource allocation games
        # Get initial inventories
        initial_inventories = game_result.game_data.get('initial_inventories', {})

        if player_id not in initial_inventories:
            return 0.0

        # Simulate the trade
        simulated_inventory = initial_inventories[player_id].copy()

        if deal.get('proposer') == player_id:
            # This player is proposing - they give 'offer' and get 'request'
            offer = deal.get('offer', {})
            request = deal.get('request', {})

            for resource, amount in offer.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) - amount

            for resource, amount in request.items():
                simulated_inventory[resource] = simulated_inventory.get(resource, 0) + amount
        else:
            # This player is receiving the proposal - they give 'request' and get 'offer'
            offer = deal.get('offer', {})  # What they would receive
            request = deal.get('request', {})  # What they would give

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

    def _calculate_integrative_utility(self, player_id: str, proposal_data: Dict[str, Any], game_result: GameResult) -> float:
        """Calculate utility for integrative negotiation proposals"""
        # Extract the proposal details - this will depend on your integrative game structure
        # For now, using a generic approach that should work with most integrative games
        
        # If the proposal contains direct utility values
        if 'utility' in proposal_data:
            return proposal_data['utility']
        
        # If the proposal contains resource allocations or deal terms
        if 'terms' in proposal_data:
            terms = proposal_data['terms']
            # Calculate utility based on the terms - this needs to match your integrative game logic
            # This is a placeholder - you may need to adjust based on your actual game structure
            return sum(terms.values()) if isinstance(terms, dict) else 0.0
        
        # Fallback: try to extract any numeric values from the proposal
        numeric_values = []
        for key, value in proposal_data.items():
            if isinstance(value, (int, float)):
                numeric_values.append(value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        numeric_values.append(sub_value)
        
        return sum(numeric_values) if numeric_values else 0.0

    def get_description(self) -> str:
        return """
        Risk Minimization measures the percentage of proposed deals/offers that are within BATNA limits.
        Formula: (Number of deals/offers within BATNA / Total number of deals/offers) * 100

        100%: All proposed deals/offers were within BATNA limits (excellent risk management)
        50%: Half of deals/offers were within BATNA (moderate risk management)  
        0%: No deals/offers were within BATNA limits (poor risk management)
        
        For price bargaining: Buyers should offer <= BATNA, Sellers should offer >= BATNA
        For other games: Final utility should be >= BATNA
        """