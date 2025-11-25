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
        """
        Initialize the Feasibility metric with optional configuration.
        
        Creates a new FeasibilityMetric instance that evaluates whether
        negotiated agreements are actually achievable given game constraints
        and player limitations. This binary metric helps identify unrealistic
        or impossible negotiation outcomes.
        
        Args:
            config (Dict[str, Any], optional): Configuration parameters for
                metric behavior. Currently unused but reserved for future
                enhancements such as:
                    - constraint_tolerance: Flexibility in feasibility checking
                    - validation_mode: Strict vs. lenient constraint enforcement
                    - custom_constraints: Additional feasibility rules
                Defaults to None (empty configuration).
        
        Example:
            >>> # Basic initialization
            >>> metric = FeasibilityMetric()
            >>> # With configuration (future use)
            >>> config = {"tolerance": 0.05, "mode": "strict"}
            >>> metric = FeasibilityMetric(config)
        
        Note:
            Feasibility checking adapts automatically to different game types
            and their specific constraint systems.
        """
        super().__init__("Feasibility", config)

    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """
        Calculate feasibility score for each player's perspective on the negotiated agreement.
        
        Evaluates whether the final negotiated agreement is actually achievable
        given game constraints, player resources, and external limitations.
        Returns binary scores indicating feasibility from each player's viewpoint.
        
        Args:
            game_result (GameResult): Complete game outcome containing:
                - game_data: Game state with agreement details and constraints
                - final_scores: Final utility outcomes for validation
                - players: List of all participating players
            actions_history (List[PlayerAction]): Complete action log (unused for
                this metric but required by BaseMetric interface).
        
        Returns:
            Dict[str, float]: Dictionary mapping each player ID to their feasibility
                score as a binary value:
                - 1.0: Agreement is feasible from this player's perspective
                - 0.0: Agreement is not feasible or no agreement reached
        
        Feasibility Criteria (Game-Specific):
            - Price Bargaining: Agreed price within BATNA constraints
            - Resource Allocation: Total resources don't exceed available supply
            - Integrative: All issue selections are valid and achievable
        
        Example:
            >>> result = metric.calculate(game_result, actions_history)
            >>> print(result)
            {'buyer': 1.0, 'seller': 1.0}  # Feasible for both parties
        
        Note:
            Different players may have different feasibility scores if the
            agreement violates constraints for some but not all participants.
        """
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
        """Check if an agreement is feasible for a specific player.
        
        This method validates whether a proposed agreement can be implemented
        by a specific player given their initial resource inventory and the
        trade requirements specified in the agreement.
        
        Args:
            player_id (str): Unique identifier for the player whose
                feasibility is being assessed.
            agreement (Dict[str, Any]): Agreement details containing trade
                specifications, proposer information, and resource exchanges.
            initial_inventories (Dict[str, Dict[str, int]]): Starting resource
                inventories for all players, mapping player IDs to their
                available resource quantities.
        
        Returns:
            bool: True if the agreement is feasible for the specified player
                (sufficient resources to fulfill trade requirements), False
                if resources are insufficient or player not found.
        
        Note:
            Feasibility assessment considers resource availability after
            executing the proposed trade, ensuring the player has sufficient
            inventory to meet their trade obligations.
        """
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
        """Provides a comprehensive description of the Feasibility metric.
        
        This method returns a detailed explanation of how the Feasibility metric
        evaluates whether negotiated agreements can be realistically implemented
        given the constraints, resources, and time pressures present in the
        negotiation scenario.
        
        Returns:
            str: A multi-line string containing:
                - Metric definition and implementability assessment
                - Binary scoring system (1.0 for feasible, 0.0 for infeasible)
                - Game-specific feasibility validation criteria
                - Constraint categories and resource limitations
                - Time pressure and BATNA decay considerations
        
        Note:
            The description covers various negotiation contexts including resource
            allocation, budget constraints, compatibility requirements, and
            acceptable range validations across different game types.
        """
        return 
        """
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