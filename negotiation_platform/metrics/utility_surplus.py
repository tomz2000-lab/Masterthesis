"""
Utility Surplus Metric
======================

Implements the Utility Surplus metric, which measures how much value each player
extracted from the negotiation compared to their Best Alternative to Negotiated
Agreement (BATNA). This is a fundamental metric for evaluating negotiation effectiveness.

Formula:
    Utility Surplus = Final Utility from Agreement - BATNA Utility

Interpretation:
    - Positive values: Player achieved better outcome than their best alternative
    - Zero values: Player achieved exactly their BATNA (or no agreement reached)
    - Negative values: Player achieved worse outcome than their best alternative

Key Features:
    - Game-agnostic calculation supporting all negotiation types
    - Handles both absolute and relative utility scales
    - Graceful handling of missing BATNA data with fallback strategies
    - Special handling for no-agreement scenarios
    - Detailed logging for debugging and analysis

Applications:
    - Measuring individual player negotiation performance
    - Comparing negotiation outcomes across different sessions
    - Evaluating the effectiveness of negotiation strategies
    - Analyzing win-win vs. win-lose negotiation outcomes
"""

from typing import Dict, List, Any
from negotiation_platform.core.base_metric import BaseMetric
from negotiation_platform.games.base_game import GameResult, PlayerAction

class UtilitySurplusMetric(BaseMetric):
    """
    Metric for calculating utility surplus achieved by each player in negotiations.
    
    The UtilitySurplusMetric quantifies how much better (or worse) each player
    performed compared to their Best Alternative to Negotiated Agreement (BATNA).
    This provides a normalized measure of negotiation success that accounts for
    each player's outside options.
    
    Calculation Method:
        For each player: Final Utility - BATNA Utility = Surplus
        
        The metric handles different game types automatically:
        - company_car: Subtracts BATNA from absolute monetary utilities
        - integrative_negotiations: Subtracts BATNA from point-based utilities
        - resource_allocation: Subtracts BATNA from resource-based utilities
    
    Special Cases:
        - No Agreement: All players receive 0.0 surplus (stayed at BATNA)
        - Missing BATNA: Uses raw utility as fallback with warning
        - Missing Player: Assigns 0.0 surplus for missing players
    
    Value Interpretation:
        - > 0: Player improved upon their BATNA through negotiation
        - = 0: Player achieved exactly their BATNA value
        - < 0: Player accepted worse outcome than their BATNA
    
    Example:
        >>> metric = UtilitySurplusMetric()
        >>> game_result = GameResult(
        ...     final_scores={"player1": 45000, "player2": 38000},
        ...     game_data={
        ...         "agreement_reached": True,
        ...         "batnas_at_agreement": {"player1": 41000, "player2": 35000}
        ...     }
        ... )
        >>> surplus = metric.calculate(game_result, [])
        >>> print(surplus)
        {"player1": 4000.0, "player2": 3000.0}
    
    Attributes:
        Inherits from BaseMetric:
        - metric_name: "Utility Surplus"
        - config: Optional configuration parameters
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Utility Surplus metric with optional configuration.
        
        Creates a new UtilitySurplusMetric instance with the standard name
        "Utility Surplus" and any provided configuration parameters.
        
        Args:
            config (Dict[str, Any], optional): Configuration parameters for
                metric behavior. Currently unused but reserved for future
                enhancements such as:
                    - normalization_method: How to scale surplus values
                    - missing_batna_strategy: Handling of missing BATNA data
                    - precision: Decimal places for calculations
                Defaults to None (empty configuration).
        
        Example:
            >>> # Basic initialization
            >>> metric = UtilitySurplusMetric()
            >>> # With configuration (future use)
            >>> config = {"precision": 2, "normalize": True}
            >>> metric = UtilitySurplusMetric(config)
        """
        super().__init__("Utility Surplus", config)

    def calculate(self, game_result: GameResult, actions_history: List[PlayerAction]) -> Dict[str, float]:
        """
        Calculate utility surplus for each player based on final outcomes and BATNA values.
        
        Computes how much each player improved (or worsened) their position compared
        to their Best Alternative to Negotiated Agreement. The calculation method
        adapts automatically to different game types and utility scales.
        
        Args:
            game_result (GameResult): Complete game outcome containing:
                - final_scores: Dict mapping player IDs to final utility values
                - game_data: Game state including agreement status and BATNA values
                - players: List of all participating players
            actions_history (List[PlayerAction]): Complete action log (unused for
                this metric but required by BaseMetric interface).
        
        Returns:
            Dict[str, float]: Dictionary mapping each player ID to their utility
                surplus value. Positive values indicate improvement over BATNA,
                negative values indicate worse outcomes than BATNA.
        
        Calculation Logic:
            1. Check if agreement was reached (no agreement = 0 surplus for all)
            2. For each player, extract final utility and BATNA value
            3. Calculate surplus as: Final Utility - BATNA Utility
            4. Handle missing data with appropriate fallbacks
        
        Game Type Handling:
            - company_car: Uses absolute monetary values with time-decayed BATNAs
            - integrative_negotiations: Uses point-based utilities with fixed BATNAs
            - resource_allocation: Uses resource-based utilities with team BATNAs
            - unknown: Generic handling with BATNA detection
        
        Special Cases:
            - No Agreement: Returns 0.0 for all players (stayed at BATNA)
            - Missing BATNA: Uses raw utility with warning message
            - Missing Player: Returns 0.0 for players not in final_scores
        
        Example:
            >>> # Company car negotiation
            >>> game_result = GameResult(
            ...     players=["buyer", "seller"],
            ...     final_scores={"buyer": 43000, "seller": 43000},
            ...     game_data={
            ...         "agreement_reached": True,
            ...         "batnas_at_agreement": {"buyer": 41000, "seller": 39000}
            ...     }
            ... )
            >>> metric = UtilitySurplusMetric()
            >>> surplus = metric.calculate(game_result, [])
            >>> print(surplus)
            {'buyer': 2000.0, 'seller': 4000.0}
        
        Error Handling:
            - Missing final_scores: Returns 0.0 for affected players
            - Missing BATNA data: Falls back to raw utility with logging
            - Invalid game_data: Graceful degradation with warnings
        """
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
        """
        Provide a comprehensive description of the Utility Surplus metric.
        
        Returns:
            str: Detailed explanation of metric purpose, calculation, and interpretation.
        """
        return """
        Utility Surplus measures how much value each player extracted from the negotiation 
        compared to their Best Alternative to Negotiated Agreement (BATNA).
        
        Formula: Final Utility from Agreement - BATNA Utility = Utility Surplus
        
        The metric adapts to different game types automatically:
        • company_car: Monetary surplus from price negotiations (e.g., $2,000 above BATNA)
        • integrative_negotiations: Point surplus from multi-issue negotiations
        • resource_allocation: Resource value surplus from allocation decisions
        
        Value Interpretation:
        • Positive values: Player negotiated better outcome than their best alternative
        • Zero values: Player achieved exactly their BATNA (or no agreement reached)
        • Negative values: Player accepted worse outcome than their BATNA (poor negotiation)
        
        Applications:
        - Measure individual negotiation effectiveness
        - Compare performance across different negotiation sessions
        - Identify win-win vs. win-lose negotiation patterns
        - Evaluate strategy effectiveness in value creation and claiming
        
        Note: Higher surplus values indicate more successful negotiations, but optimal
        outcomes often involve balanced surplus distribution between parties.
        """