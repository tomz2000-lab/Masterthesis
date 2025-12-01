"""
Company Car Price Bargaining Game
=================================

Bilateral negotiation game for vehicle price negotiations with realistic market dynamics.

This module implements a structured car buying/selling negotiation where a buyer and seller
negotiate the price of a company vehicle. The game includes realistic elements such as:

- Time-decaying BATNAs (Best Alternatives to Negotiated Agreement)
- Strategic guidance and structured prompts
- Market-realistic price ranges and constraints
- JSON-based proposal system for structured interactions

Key Features:
    - Bilateral negotiation between buyer and seller roles
    - Dynamic BATNA values that decay over time to simulate urgency
    - Structured proposal system with validation
    - Strategic prompts to guide realistic negotiation behavior
    - Win-win outcome detection and scoring

Game Parameters:
    - Starting Price: Initial vehicle price for negotiation
    - Buyer Budget: Maximum amount buyer can afford
    - Seller Cost: Minimum acceptable amount for seller  
    - BATNA Values: Alternative options for both parties
    - Time Decay: How BATNAs change over negotiation rounds

Example:
    >>> config = {
    ...     "starting_price": 42000,
    ...     "buyer_budget": 45000,
    ...     "seller_cost": 38000,
    ...     "buyer_batna": 41000,
    ...     "seller_batna": 39000,
    ...     "rounds": 5,
    ...     "batna_decay": 0.02
    ... }
    >>> game = CompanyCarGame(config)
    >>> game.initialize_game(["buyer_agent", "seller_agent"])
"""

import random
import re
import json
from typing import Dict, List, Any, Optional
from .base_game import BaseGame, PlayerAction


class CompanyCarGame(BaseGame):
    """
    Company car price negotiation game implementing realistic bilateral bargaining dynamics.
    
    This class manages structured price negotiations between a buyer and seller for
    a company vehicle purchase. The game incorporates realistic market dynamics,
    time pressure through BATNA decay, and strategic guidance to create authentic
    negotiation experiences.
    
    The game simulates a common business scenario where organizations must negotiate
    vehicle purchases, balancing budget constraints with value optimization. Both
    parties have alternatives (BATNAs) that become less attractive over time,
    encouraging timely agreement.
    
    Key Features:
        - Realistic price negotiation with market constraints
        - Dynamic BATNA values that decay to simulate time pressure
        - Strategic guidance system with contextual prompts
        - JSON-based structured proposal system for clear communication
        - Win-win outcome detection based on surplus above BATNA values
        - Comprehensive utility tracking and performance analysis
    
    Game Mechanics:
        1. Buyer and seller alternate making price proposals
        2. Each proposal is validated against budget and cost constraints
        3. Players can make offers, counteroffers, or accept/reject proposals
        4. BATNA values decay each round to encourage timely resolution
        5. Game ends on acceptance or maximum rounds reached
        6. Final utilities calculated based on achieved price vs. BATNA
    
    Attributes:
        starting_price (int): Initial vehicle price for negotiation reference.
        buyer_budget (int): Maximum amount buyer can afford to pay.
        seller_cost (int): Minimum amount seller needs to receive.
        buyer_batna (float): Buyer's best alternative option value.
        seller_batna (float): Seller's best alternative option value.
        batna_decay (float): Per-round decay rate for BATNA values.
        current_price (Optional[int]): Most recent proposed price.
        agreement_reached (bool): Whether parties have reached agreement.
        final_price (Optional[int]): Agreed upon price if deal completed.
    
    Example:
        >>> config = {
        ...     "starting_price": 42000,
        ...     "buyer_budget": 45000,
        ...     "seller_cost": 38000,
        ...     "buyer_batna": 41000,
        ...     "seller_batna": 39000,
        ...     "rounds": 5,
        ...     "batna_decay": 0.02
        ... }
        >>> game = CompanyCarGame(config)
        >>> game.initialize_game(["buyer_agent", "seller_agent"])
        >>> 
        >>> # Buyer makes initial offer
        >>> action = {"type": "offer", "price": 40000}
        >>> valid = game.is_valid_action("buyer_agent", action)
        >>> if valid:
        ...     game.process_actions({"buyer_agent": action}, game_state)
    
    Strategic Considerations:
        - Buyer Strategy: Start low but stay above seller's likely cost
        - Seller Strategy: Start high but consider buyer's budget limits
        - Both: Monitor BATNA decay and time pressure effects
        - Optimal: Find price range where both parties gain vs. their BATNAs
    
    Outcome Analysis:
        Game success measured by:
        - Agreement reached within round limit
        - Both parties achieve positive surplus above BATNA
        - Efficient price discovery within feasible range
        - Balanced utility distribution between parties
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize company car negotiation game with configuration parameters.
        
        Sets up the bilateral price negotiation environment with buyer and seller
        roles, BATNA values, budget constraints, and time decay mechanisms.
        Validates that all required configuration parameters are provided.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - starting_price (int): Initial vehicle price for reference
                - buyer_budget (int): Maximum amount buyer can afford
                - seller_cost (int): Minimum amount seller needs to receive
                - buyer_batna (float): Buyer's best alternative value
                - seller_batna (float): Seller's best alternative value
                - rounds (int): Maximum negotiation rounds allowed
                - batna_decay (float): Per-round BATNA decay rate (0.0-1.0)
        
        Raises:
            ValueError: If any required configuration field is missing.
        
        Example:
            >>> config = {
            ...     "starting_price": 42000,
            ...     "buyer_budget": 45000,
            ...     "seller_cost": 38000,
            ...     "buyer_batna": 41000,
            ...     "seller_batna": 39000,
            ...     "rounds": 5,
            ...     "batna_decay": 0.02
            ... }
            >>> game = CompanyCarGame(config)
        """
        # Initialize base class with dummy game_id - will be set by game engine
        super().__init__(game_id="company_car", config=config)
        # Require all parameters from config, raise error if missing
        required_fields = [
            "starting_price", "buyer_budget", "seller_cost",
            "buyer_batna", "seller_batna", "rounds", "batna_decay"
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        self.starting_price = config["starting_price"]
        self.buyer_budget = config["buyer_budget"]
        self.seller_cost = config["seller_cost"]
        self.buyer_batna = config["buyer_batna"]
        self.seller_batna = config["seller_batna"]
        self.max_rounds = config["rounds"]
        self.batna_decay = config["batna_decay"]

    def validate_json_response(self, response: str) -> bool:
        """
        Validate that a response string contains properly formatted JSON.
        
        Checks if the provided response can be parsed as valid JSON and
        contains the required "type" field for action identification.
        Used for input validation before processing player responses.
        
        Args:
            response (str): Raw response string from player to validate.
        
        Returns:
            bool: True if response is valid JSON with "type" field,
                 False otherwise.
        
        Example:
            >>> valid_response = '{"type": "offer", "price": 42000}'
            >>> game.validate_json_response(valid_response)
            True
            >>> invalid_response = 'I want to offer 42000'
            >>> game.validate_json_response(invalid_response)
            False
        """
        try:
            data = json.loads(response.strip())
            return isinstance(data, dict) and "type" in data
        except (json.JSONDecodeError, TypeError):
            return False

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and normalize JSON response from players into standard format.
        
        Extracts decision data from various JSON response formats, handling
        both direct action format and structured response format. Provides
        robust error recovery with fallback parsing for malformed responses.
        
        Args:
            response (str): Raw JSON response string from player.
        
        Returns:
            Dict[str, Any]: Parsed response containing:
                - decision (Dict[str, Any]): Extracted action data with "type" field
                - raw_response (str): Original response for debugging
        
        Example:
            >>> response = '{"type": "offer", "price": 42000}'
            >>> parsed = game.parse_json_response(response)
            >>> print(parsed["decision"]["type"])
            offer
            >>> print(parsed["decision"]["price"])
            42000
        
        Note:
            Falls back to {"type": "reject"} for unparseable responses
            to ensure graceful handling of malformed input.
        """
        try:
            # Clean the response by removing common instruction patterns
            cleaned_response = response.strip()
            
            # Remove any surrounding text that isn't JSON
            json_match = re.search(r'\{[^{}]*"type"[^{}]*\}', cleaned_response)
            if json_match:
                json_str = json_match.group(0)
                decision_data = json.loads(json_str)
                
                # Validate that decision has required type field
                if not decision_data.get("type"):
                    print(f"⚠️ Decision missing 'type' field: {decision_data}")
                    decision_data = {"type": "reject"}
                    
                return {
                    "decision": decision_data,
                    "raw_response": response
                }
            else:
                # Try to parse the entire response as JSON
                decision_data = json.loads(cleaned_response)
                if not decision_data.get("type"):
                    decision_data = {"type": "reject"}
                    
                return {
                    "decision": decision_data,
                    "raw_response": response
                }
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}")
            print(f"Raw response: {response[:200]}...")
            
            # Try to extract type and price manually as fallback
            type_match = re.search(r'"type":\s*"([^"]+)"', response)
            price_match = re.search(r'"price":\s*(\d+)', response)
            if type_match:
                decision_data = {"type": type_match.group(1)}
                if price_match:
                    decision_data["price"] = int(price_match.group(1))
                return {
                    "decision": decision_data,
                    "raw_response": response
                }
            
            # Ultimate fallback
            return {
                "decision": {"type": "reject"},
                "raw_response": response
            }
        except Exception as e:
            print(f"⚠️ Failed to parse JSON response: {e}")
            return {
                "decision": {"type": "reject"}, 
                "raw_response": response
            }

    def initialize_game(self, players: List[str]) -> Dict[str, Any]:
        """
        Process a single player action and update the game state accordingly.
        
        Handles individual player actions by adding them to history and
        delegating to the batch process_actions method. Required by BaseGame
        interface for single-action processing.
        
        Args:
            action (PlayerAction): Player action to process containing player_id,
                action_type, action_data, timestamp, and round_number.
        
        Returns:
            Dict[str, Any]: Updated game state after processing the action.
        
        Example:
            >>> action = PlayerAction(
            ...     player_id="buyer",
            ...     action_type="offer",
            ...     action_data={"price": 42000},
            ...     timestamp=1609459200.0,
            ...     round_number=3
            ... )
            >>> new_state = game.process_action(action)
        """
        if len(players) != 2:
            raise ValueError("Company car game requires exactly 2 players")

        self.players = players
        
        # Randomly assign buyer and seller roles to eliminate role-based bias
        if random.choice([True, False]):
            self.buyer = players[0]
            self.seller = players[1]
            print(f"🎲 [ROLE ASSIGNMENT] {players[0]} = BUYER, {players[1]} = SELLER")
        else:
            self.buyer = players[1] 
            self.seller = players[0]
            print(f"🎲 [ROLE ASSIGNMENT] {players[1]} = BUYER, {players[0]} = SELLER")
            
        self.state = self.state.__class__.ACTIVE  # Set to active state

        self.game_data = {
            "game_type": "company_car",
            "players": self.players,
            "rounds": self.max_rounds,
            "current_round": 1,
            "role_assignments": {
                "buyer": self.buyer,
                "seller": self.seller
            },
            "round_by_round_surplus": {
                self.buyer: [],
                self.seller: []
            },
            "private_info": {
                self.buyer: {
                    "role": "buyer",
                    "budget": self.buyer_budget,
                    "batna": self.buyer_batna,
                    "starting_price": self.starting_price
                },
                self.seller: {
                    "role": "seller",
                    "cost": self.seller_cost,
                    "batna": self.seller_batna,
                    "starting_price": self.starting_price
                }
            },
            "public_info": {
                "deadline": self.max_rounds,
                "starting_price": self.starting_price
            }
        }
        return self.game_data

    def get_current_batna(self, player: str, round_num: int) -> float:
        """
        Calculate time-adjusted BATNA value for specified player and round.
        
        Applies exponential decay to the player's initial BATNA value based on
        the current round number, simulating decreasing value of alternatives
        over time. This creates time pressure encouraging earlier agreements.
        
        Args:
            player (str): Player identifier (buyer or seller).
            round_num (int): Current round number (1-based).
        
        Returns:
            float: Time-adjusted BATNA value for the specified round.
        
        Example:
            >>> # Initial buyer BATNA: 41000, decay rate: 0.02
            >>> round_1_batna = game.get_current_batna("buyer", 1)
            >>> print(f"Round 1 BATNA: €{round_1_batna:,.0f}")
            Round 1 BATNA: €40,180
            >>> round_3_batna = game.get_current_batna("buyer", 3)
            >>> print(f"Round 3 BATNA: €{round_3_batna:,.0f}")
            Round 3 BATNA: €39,572
        
        Note:
            BATNA decay formula: initial_batna * (1 - decay_rate)^round_num
        """
        if player == self.buyer:
            decay_rate = self.batna_decay["buyer"]
            base_batna = self.buyer_batna
        else:
            decay_rate = self.batna_decay["seller"]
            base_batna = self.seller_batna

        return base_batna * ((1 - decay_rate) ** (round_num-1))

    def is_valid_action(self, player: str, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """
        Validate player action against game rules with flexible format support.
        
        Validates negotiation actions including offers, acceptances, and rejections.
        Supports both direct action format and structured response format with
        "decision" wrapper. Ensures actions comply with game constraints including
        proposal limits and valid action types.
        
        Args:
            player (str): Identifier of the player taking the action.
                Must be registered buyer or seller.
            action (Dict[str, Any]): Action data to validate. Supported formats:
                - Direct: {"type": "offer", "price": 42000}
                - Structured: {"decision": {"type": "offer", "price": 42000}}
            game_state (Dict[str, Any]): Current game state containing round
                information, proposal counts, and negotiation history.
        
        Returns:
            bool: True if action is valid and can be processed, False otherwise.
        
        Validation Rules:
            - Action must have valid "type" field (offer, accept, reject)
            - Offers must include numeric "price" field
            - Player must not exceed proposal limits
            - Price offers must be positive numeric values
        
        Example:
            >>> # Valid offer action
            >>> action = {"type": "offer", "price": 42000}
            >>> is_valid = game.is_valid_action("buyer", action, game_state)
            >>> print(is_valid)
            True
            
            >>> # Valid structured format
            >>> structured = {"decision": {"type": "accept"}}
            >>> is_valid = game.is_valid_action("seller", structured, game_state)
            >>> print(is_valid)
            True
        
        Note:
            Invalid actions are logged but do not raise exceptions, allowing
            graceful handling of malformed AI model responses.
        """
        # Handle structured response format
        if isinstance(action, dict) and "decision" in action:
            action_data = action["decision"]
        else:
            action_data = action
            
        action_type = action_data.get("type", "")
        
        # Handle empty or invalid action types by treating as reject
        if not action_type or action_type == "":
            print(f"⚠️ Player {player} provided empty action type, treating as reject")
            return True  # Allow but will be processed as reject
            
        max_proposals = self.max_rounds-1
        player_proposals = game_state.get(f"{player}_proposal_count", 0)

        if action_type == "offer":
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} tried to make offer but exceeded proposal limit ({player_proposals}/{max_proposals})")
                return False
                
            price = action_data.get("price", 0)
            if price <= 0:
                return False

            # Basic range validation - offers should be reasonable
            if price > 100000 or price < 10000:  # Sanity check
                return False

            return True

        elif action_type in ["accept", "reject"]:
            return True

        elif action_type in ["counter", "counteroffer"]:
            # Check proposal limit for counters too
            if player_proposals >= max_proposals:
                return False
                
            # Treat counter/counteroffer as regular offers
            price = action_data.get("price", 0)
            return price > 0 and 10000 <= price <= 100000

        elif action_type in ["offer_accepted", "offer_response"]:
            # Treat these as accept actions
            return True

        elif action_type == "noop":
            # Allow no-op actions as fallback
            return True

        return False

    def process_actions(self, actions: Dict[str, Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process simultaneous player actions with proposal tracking and validation.
        
        Handles bilateral negotiation actions including offers, acceptances, and
        rejections. Enforces proposal limits, validates action compatibility,
        and determines negotiation outcomes. Updates game state with action
        results and manages agreement detection.
        
        Args:
            actions (Dict[str, Dict[str, Any]]): Mapping of player identifiers
                to their action data. Expected format:
                {"player_id": {"type": "offer", "price": 42000}}
            game_state (Dict[str, Any]): Current game state containing:
                - current_round: Round number for tracking
                - proposal counts: Limits for each player
                - negotiation history: Previous actions and offers
        
        Returns:
            Dict[str, Any]: Updated game state containing:
                - agreement_reached: Boolean indicating successful negotiation
                - final_price: Agreed price if agreement reached
                - final_utilities: Utility scores for each player
                - game_over: Boolean indicating termination
                - Updated proposal counts and action history
        
        Processing Logic:
            1. Initialize and validate proposal counters
            2. Process each player's action with validation
            3. Check for mutual acceptances (agreement)
            4. Handle offers and update last offer tracking
            5. Update proposal counts and game state
            6. Determine if negotiation should continue
        
        Agreement Detection:
            - Mutual acceptance: Both players accept in same round
            - Offer acceptance: One player accepts other's previous offer
            - Price agreement: Players converge on acceptable price
        
        Example:
            >>> actions = {
            ...     "buyer": {"type": "offer", "price": 41000},
            ...     "seller": {"type": "offer", "price": 43000}
            ... }
            >>> new_state = game.process_actions(actions, current_state)
            >>> print(new_state["agreement_reached"])
            False  # No agreement, continue negotiating
        
        Note:
            Players exceeding proposal limits receive rejection responses.
            Invalid actions are logged and may result in negotiation failure.
        """
        current_round = game_state["current_round"]
        max_proposals = self.max_rounds-1  # Use rounds from YAML config

        # Initialize proposal counters if not present
        for player in [self.buyer, self.seller]:
            if f"{player}_proposal_count" not in game_state:
                game_state[f"{player}_proposal_count"] = 0

        # Process JSON responses and extract decision data
        processed_actions = {}
        for player, raw_action in actions.items():
            if isinstance(raw_action, str):
                # If it's a string, try to parse it as JSON response
                parsed = self.parse_json_response(raw_action)
                action_data = parsed["decision"]
            elif isinstance(raw_action, dict) and "decision" in raw_action:
                # Already structured
                action_data = raw_action["decision"]
            else:
                # Regular action format
                action_data = raw_action
            
            processed_actions[player] = action_data

        # Normalize action types - treat counter/counteroffer as offers, handle empty types
        normalized_actions = {}
        for player, action in processed_actions.items():
            action_type = action.get("type", "")
            
            # Handle empty or invalid action types
            if not action_type or action_type == "":
                print(f"⚠️ Player {player} provided empty action type, treating as reject")
                normalized_actions[player] = {"type": "reject"}
            elif action_type in ["counter", "counteroffer"]:
                # Convert to offer
                normalized_actions[player] = {"type": "offer", "price": action.get("price")}
            elif action_type in ["offer_accepted", "offer_response"]:
                # Convert to accept
                normalized_actions[player] = {"type": "accept"}
            else:
                normalized_actions[player] = action

        # Check for offers and responses
        offers = {player: action for player, action in normalized_actions.items()
                  if action.get("type") == "offer"}
        responses = {player: action for player, action in normalized_actions.items()
                     if action.get("type") in ["accept", "reject"]}

        # Process rejections - only end if proposal limit reached
        for player, action in responses.items():
            if action.get("type") == "reject":
                player_proposals = game_state.get(f"{player}_proposal_count", 0)
                if player_proposals >= max_proposals:
                    print(f"❌ Player {player} rejected after reaching proposal limit ({player_proposals}/{max_proposals})")
                    return self._create_no_agreement(game_state)
                else:
                    print(f"⚠️ Player {player} rejected but still has proposals remaining ({player_proposals}/{max_proposals}). Continuing negotiation.")

        # Process offers with proposal limit validation
        for player, action in offers.items():
            player_proposals = game_state.get(f"{player}_proposal_count", 0)
            
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} exceeded proposal limit ({player_proposals}/{max_proposals}). Ignoring additional offers.")
                # Don't process this offer, but don't end negotiation unless they also rejected
                continue
            
            # Valid offer - process it
            price = action.get("price")
            game_state[f"{player}_last_offer"] = price
            game_state[f"{player}_last_offer_round"] = current_round  # Track when offer was made
            game_state[f"{player}_proposal_count"] = player_proposals + 1
            print(f"💡 Player {player} made offer €{price:,.0f} (proposal {player_proposals + 1}/{max_proposals})")

        # Check for convergence: if both players made identical offers, create agreement
        if len(offers) == 2:  # Both players made offers this round
            offer_prices = [action.get("price") for action in offers.values()]
            if len(set(offer_prices)) == 1 and offer_prices[0] is not None:  # All offers are identical and valid
                agreed_price = offer_prices[0]
                print(f"🎉 CONVERGENCE! Both players offered €{agreed_price:,.0f} - Creating automatic agreement!")
                return self._create_agreement(agreed_price, current_round, game_state)

        # Process acceptances (rejections already handled above)
        for player, action in responses.items():
            if action.get("type") == "accept":
                # Find the offer being accepted
                other_player = self.seller if player == self.buyer else self.buyer
                if f"{other_player}_last_offer" in game_state:
                    agreed_price = game_state[f"{other_player}_last_offer"]
                    # Get the round when the accepted offer was made
                    offer_round = game_state.get(f"{other_player}_last_offer_round", current_round)
                    print(f"✅ Player {player} accepted offer of €{agreed_price:,.0f} (made in round {offer_round})")

                    # Use the BATNA from the current round (when acceptance happens), not when offer was made
                    return self._create_agreement(agreed_price, current_round, game_state)
                else:
                    print(f"⚠️ Player {player} tried to accept but no offer exists")

        # Update round
        game_state["current_round"] += 1

        # Check if deadline reached - but allow extra rounds for final responses
        # Players should have a chance to accept/reject final proposals
        max_total_rounds = self.max_rounds
        
        if game_state["current_round"] > max_total_rounds:
            return self._create_no_agreement(game_state)

        return game_state

    def _create_agreement(self, price: float, current_round: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create final agreement result with utility calculations and surplus analysis.
        
        Generates comprehensive agreement outcome including final utilities,
        surplus calculations above BATNA values, and detailed round-by-round
        tracking. Determines winner based on positive utility surplus.
        
        Args:
            price (float): Agreed upon final price.
            current_round (int): Round when agreement was reached.
            game_state (Dict[str, Any]): Current game state dictionary.
        
        Returns:
            Dict[str, Any]: Complete agreement result containing:
                - agreement_reached (bool): True
                - final_price (float): Agreed price
                - final_utilities (Dict[str, float]): Player utility values
                - batnas_at_agreement (Dict[str, float]): BATNA values at agreement
                - utility_surplus (Dict[str, float]): Surplus above BATNA
                - winner (Optional[str]): Player with highest positive surplus
        
        Example:
            >>> result = game._create_agreement(42000, 3, game_state)
            >>> print(f"Price: €{result['final_price']:,.0f}")
            >>> print(f"Buyer utility: {result['final_utilities']['buyer']:.1f}")
            Price: €42,000
            Buyer utility: 8.2
        """
        buyer_batna = self.get_current_batna(self.buyer, current_round)
        seller_batna = self.get_current_batna(self.seller, current_round)

        # For the company car game:
        # - The car has intrinsic value equal to the buyer's BATNA (alternative car cost)
        # - The seller's minimum cost is their BATNA
        
        # Calculate absolute utilities (not used any more, as it is too confusing)
        buyer_absolute_utility = buyer_batna - (price - buyer_batna)  # Value - net cost = BATNA - (price - BATNA) = 2*BATNA - price
        seller_absolute_utility = price - seller_batna + seller_batna  # Revenue - cost + BATNA = price
        
        
        # The surplus we want is:
        buyer_surplus = buyer_batna - price  # Money saved vs BATNA
        seller_surplus = price - seller_batna  # Profit over BATNA
        
        # Convert to absolute utilities for session manager
        buyer_utility = buyer_surplus + buyer_batna  # This will give surplus when session manager subtracts BATNA
        seller_utility = seller_surplus + seller_batna  # This will give surplus when session manager subtracts BATNA
        
        # DEBUG: Log the exact calculation values
        print(f"🔍 [BATNA DEBUG] Round {current_round}: price={price}")
        print(f"🔍 [BATNA DEBUG] Config BATNAs: buyer={self.buyer_batna}, seller={self.seller_batna}")
        print(f"🔍 [BATNA DEBUG] Decay rate: {self.batna_decay}")
        print(f"🔍 [BATNA DEBUG] Calculated BATNAs: buyer={buyer_batna:.2f}, seller={seller_batna:.2f}")
        print(f"🔍 [SURPLUS DEBUG] Buyer surplus: {buyer_surplus:.2f} (saved €{buyer_surplus:.2f} vs BATNA)")
        print(f"🔍 [SURPLUS DEBUG] Seller surplus: {seller_surplus:.2f} (profit €{seller_surplus:.2f} over BATNA)")
        print(f"🔍 [UTILITY DEBUG] Final utilities (for session manager): buyer={buyer_utility:.2f}, seller={seller_utility:.2f}")
        print(f"🎲 [ROLE DEBUG] Buyer={self.buyer}, Seller={self.seller}")
        print(f"🎲 [ROLE DEBUG] {self.buyer} surplus={buyer_surplus:.2f}, {self.seller} surplus={seller_surplus:.2f}")

        game_state.update({
            "agreement_reached": True,
            "game_ended": True,  # Explicitly mark game as ended
            "agreed_price": price,
            "agreement_round": current_round,
            "role_assignments": {
                "buyer": self.buyer,
                "seller": self.seller
            },
            "final_utilities": {
                self.buyer: buyer_utility,
                self.seller: seller_utility
            },
            "batnas_at_agreement": {
                self.buyer: buyer_batna,
                self.seller: seller_batna
            }
        })

        return game_state

    def _create_no_agreement(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create final result when no agreement is reached between parties.
        
        Generates comprehensive failure outcome with BATNA-based utilities,
        indicating both parties resort to their best alternatives.
        No surplus is generated as negotiation failed to create mutual value.
        
        Args:
            game_state (Dict[str, Any]): Current game state dictionary.
        
        Returns:
            Dict[str, Any]: No agreement result containing:
                - agreement_reached (bool): False
                - final_price (None): No agreed price
                - final_utilities (Dict[str, float]): BATNA-based utilities
                - batnas_at_agreement (Dict[str, float]): Final BATNA values
                - utility_surplus (Dict[str, float]): Zero surplus for both
                - winner (None): No winner in failed negotiations
        
        Example:
            >>> result = game._create_no_agreement(game_state)
            >>> print(f"Agreement: {result['agreement_reached']}")
            >>> print(f"Final utilities: {result['final_utilities']}")
            Agreement: False
            Final utilities: {'buyer': 0.0, 'seller': 0.0}
        """
        print(f"🎲 [ROLE DEBUG] No Agreement - Buyer={self.buyer}, Seller={self.seller}")
        print(f"🎲 [ROLE DEBUG] {self.buyer} utility=0, {self.seller} utility=0")
        
        game_state.update({
            "agreement_reached": False,
            "game_ended": True,  # Explicitly mark game as ended
            "role_assignments": {
                "buyer": self.buyer,
                "seller": self.seller
            },
            "final_utilities": {
                self.buyer: 0,  # No deal utility
                self.seller: 0
            }
        })

        return game_state

    def is_game_over(self, game_state: Dict[str, Any]) -> bool:
        """
        Determine if the negotiation game has reached a terminal state.
        
        Checks for various end conditions including agreement reached,
        maximum rounds exceeded, or explicit rejections that end negotiation.
        
        Args:
            game_state (Dict[str, Any]): Current game state to evaluate.
        
        Returns:
            bool: True if game should terminate, False if negotiation continues.
        
        Example:
            >>> # Agreement reached
            >>> game_state = {"agreement_reached": True}
            >>> game.is_game_over(game_state)
            True
            >>> # Maximum rounds exceeded  
            >>> game_state = {"current_round": 6, "agreement_reached": False}
            >>> game.is_game_over(game_state)  # max_rounds = 5
            True
        """
        return (game_state.get("agreement_reached", False) or
                game_state.get("game_ended", False) or
                game_state.get("current_round", 1) > self.max_rounds)

    def get_winner(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        Determine the winning player based on positive utility surplus analysis.
        
        Evaluates final utilities and surplus values to identify the player
        who achieved the highest positive surplus above their BATNA. Only
        players with positive surplus can be considered winners.
        
        Args:
            game_state (Dict[str, Any]): Final game state with utility data.
        
        Returns:
            Optional[str]: Player identifier of winner, or None if:
                - No agreement was reached
                - No player has positive surplus
                - Utilities are tied or unavailable
        
        Example:
            >>> # Buyer achieved higher surplus
            >>> game_state = {
            ...     "agreement_reached": True,
            ...     "utility_surplus": {"buyer": 1000, "seller": 500}
            ... }
            >>> winner = game.get_winner(game_state)
            >>> print(f"Winner: {winner}")
            Winner: buyer
        """
        if not game_state.get("agreement_reached", False):
            return None

        utilities = game_state.get("final_utilities", {})
        batnas = game_state.get("batnas_at_agreement", {})
        
        if not utilities or not batnas:
            return None

        # Calculate utility surplus for each player (utility - BATNA)
        surpluses = {}
        for player in utilities.keys():
            utility = utilities[player]
            batna = batnas[player]
            surpluses[player] = utility - batna

        # Only consider players with positive surplus
        positive_surplus_players = {player: surplus for player, surplus in surpluses.items() if surplus > 0}
        
        if not positive_surplus_players:
            # No player has positive surplus - no winner
            return None
        elif len(positive_surplus_players) == 1:
            # Only one player has positive surplus - they win
            return list(positive_surplus_players.keys())[0]
        else:
            # Multiple players with positive surplus - highest surplus wins
            return max(positive_surplus_players, key=positive_surplus_players.get)

    # Required abstract methods from BaseGame
    def process_action(self, action: PlayerAction) -> Dict[str, Any]:
        """
        Process a single player action and update the game state accordingly.
        
        Handles individual player actions by adding them to history and
        delegating to the batch process_actions method. Required by BaseGame
        interface for single-action processing.
        
        Args:
            action (PlayerAction): Player action to process containing player_id,
                action_type, action_data, timestamp, and round_number.
        
        Returns:
            Dict[str, Any]: Updated game state after processing the action.
        
        Example:
            >>> action = PlayerAction(
            ...     player_id="buyer",
            ...     action_type="offer",
            ...     action_data={"price": 42000},
            ...     timestamp=1609459200.0,
            ...     round_number=3
            ... )
            >>> new_state = game.process_action(action)
        """
        # This method should process individual actions
        # For now, delegate to process_actions with single action
        player_id = action.player_id if hasattr(action, 'player_id') else 'unknown'
        action_data = action.action_data if hasattr(action, 'action_data') else action
        
        return self.process_actions({player_id: action_data}, self.game_data)

    def process_action(self, action: PlayerAction) -> Dict[str, Any]:
        """
        Process a single player action and update the game state accordingly.
        
        Handles individual player actions by adding them to history and
        delegating to the batch process_actions method. Required by BaseGame
        interface for single-action processing.
        
        Args:
            action (PlayerAction): Player action to process containing player_id,
                action_type, action_data, timestamp, and round_number.
        
        Returns:
            Dict[str, Any]: Updated game state after processing the action.
        
        Example:
            >>> action = PlayerAction(
            ...     player_id="buyer",
            ...     action_type="offer",
            ...     action_data={"price": 42000},
            ...     timestamp=1609459200.0,
            ...     round_number=3
            ... )
            >>> new_state = game.process_action(action)
        """
        # Add action to history
        self.add_action(action)
        
        # Process the action and update game state
        action_data = action.action_data if hasattr(action, 'action_data') else action
        player = action.player_id if hasattr(action, 'player_id') else action.get('player', '')
        
        # Update game state based on action
        actions_dict = {player: action_data}
        self.game_data = self.process_actions(actions_dict, self.game_data)
        
        return self.game_data

    def check_end_conditions(self) -> bool:
        """
        Check if the negotiation game should terminate based on current state.
        
        Evaluates termination conditions by delegating to the is_game_over
        method. Required by BaseGame interface for consistent end condition
        checking across all game implementations.
        
        Returns:
            bool: True if game should end, False if negotiation continues.
        
        Example:
            >>> game.check_end_conditions()
            True  # If agreement reached or max rounds exceeded
        """
        return self.is_game_over(self.game_data)

    def calculate_scores(self) -> Dict[str, float]:
        """
        Calculate final utility scores for all participating players.
        
        Computes final negotiation outcomes based on whether agreement was reached.
        If successful agreement occurred, returns actual utility values achieved
        by each player. If negotiation failed, returns zero scores for all players
        to reflect lack of value creation.
        
        Returns:
            Dict[str, float]: Mapping of player identifiers to final utility scores.
                For successful negotiations, contains actual utility values.
                For failed negotiations, contains 0.0 for all players.
        
        Example:
            >>> # Successful price agreement at $42,000
            >>> scores = game.calculate_scores()
            >>> print(scores)
            {'buyer_player': 7.5, 'seller_player': 8.2}
            
            >>> # Failed negotiation (no agreement)
            >>> scores = game.calculate_scores()
            >>> print(scores)
            {'buyer_player': 0.0, 'seller_player': 0.0}
        
        Note:
            Required by BaseGame interface for consistent scoring across all
            negotiation game implementations. Scores reflect relative success
            in achieving negotiation objectives.
        """
        if self.game_data.get("agreement_reached", False):
            return self.game_data.get("final_utilities", {})
        else:
            return {player: 0.0 for player in self.players}

    def _get_neutral_role_label(self, player_id: str) -> str:
        """
        Map player identifier to neutral role label to reduce cognitive bias.
        
        Provides neutral terminology ("Party A"/"Party B") instead of loaded
        terms ("buyer"/"seller") to minimize role-based behavioral biases
        in prompts and communications.
        
        Args:
            player_id (str): Player identifier to map.
        
        Returns:
            str: Neutral role label ("Party A" or "Party B").
        
        Example:
            >>> # If player1 is buyer, player2 is seller
            >>> game._get_neutral_role_label("player1")
            'Party A'
            >>> game._get_neutral_role_label("player2")
            'Party B'
        """
        if player_id == self.buyer:
            return "ROLE A"
        else:
            return "ROLE B"

    def get_game_prompt(self, player_id: str) -> str:
        """
        Generate comprehensive negotiation prompt with neutral role terminology.
        
        Creates detailed, contextual prompts for car price negotiations using
        neutral role labels ("Party A"/"Party B") instead of loaded terms
        ("buyer"/"seller") to reduce cognitive bias and role-based behavioral
        influences. Includes current game state, strategic guidance, and
        structured action formatting requirements.
        
        Args:
            player_id (str): Identifier of the player requesting the prompt.
                Must be one of the registered players (buyer or seller).
        
        Returns:
            str: Comprehensive negotiation prompt containing:
                - Neutral role context and scenario description
                - Current round and proposal count information
                - BATNA thresholds and acceptance criteria
                - Opponent's latest offer (if available)
                - Available actions and JSON formatting requirements
                - Strategic guidance for decision making
        
        Prompt Elements:
            - Bias-reduced role terminology (Party A/B vs buyer/seller)
            - Current negotiation state and round tracking
            - BATNA-based decision criteria and thresholds
            - Offer history and opponent action visibility
            - Structured JSON response format requirements
            - Strategic guidance for proposal and acceptance decisions
        
        Example:
            >>> prompt = game.get_game_prompt("player1")
            >>> print("Party A" in prompt)  # Neutral terminology
            True
            >>> print("JSON" in prompt)     # Format requirements
            True
            >>> print("BATNA" in prompt)    # Strategic guidance
            True
        
        Note:
            Returns error message if game is not properly initialized with
            buyer and seller assignments. Prompts adapt to current game
            state including round limits and proposal constraints.
        """
        if not hasattr(self, 'buyer') or not hasattr(self, 'seller'):
            return "Game not initialized properly"

        private_info = self.game_data.get("private_info", {}).get(player_id, {})
        current_round = self.game_data.get("current_round", 1)
        other_player = self.seller if player_id == self.buyer else self.buyer
        other_offer = self.game_data.get(f"{other_player}_last_offer", None)
        my_offer = self.game_data.get(f"{player_id}_last_offer", None)
        batna = self.get_current_batna(player_id, current_round)
        
        # Track proposals made by this player
        player_proposals = self.game_data.get(f"{player_id}_proposal_count", 0)
        max_proposals = self.max_rounds-1  # Use rounds from YAML config = 4
        
        # Map internal roles to neutral display roles to reduce bias
        internal_role = "buyer" if player_id == self.buyer else "seller"
        neutral_role = self._get_neutral_role_label(player_id)
        
        # Create neutral goal description
        if internal_role == "buyer":
            goal = f"Acquire the item for €{batna:,.0f} or less"
        else:
            goal = f"Transfer the item for €{batna:,.0f} or more"
        
        offer_history = []
        if my_offer:
            offer_history.append(f"- Your last offer: €{my_offer:,.0f}")
        if other_offer:
            offer_history.append(f"- Opponent's last offer: €{other_offer:,.0f}")
        offer_status = "\n".join(offer_history) if offer_history else "No offers made yet."

        # Enhanced acceptance guidance with round awareness
        acceptance_guidance = ""
        can_propose = player_proposals < max_proposals
        rounds_remaining = max_proposals - player_proposals
        
        if other_offer is not None:
            is_within_batna = ((player_id == self.buyer and other_offer <= batna) or 
                              (player_id == self.seller and other_offer >= batna))
            
            if is_within_batna:
                if rounds_remaining == 0:  # No proposals left - encourage acceptance
                    acceptance_guidance = (
                        f"🎯 FINAL ANALYSIS: The opponent's offer (€{other_offer:,.0f}) is within your BATNA (€{batna:,.0f}). "
                        f"You have no proposals left - ACCEPT to secure this beneficial deal!\n"
                    )
                elif rounds_remaining == 1:  # Last proposal - be more encouraging
                    acceptance_guidance = (
                        f"🎯 ANALYSIS: The opponent's offer (€{other_offer:,.0f}) is within your BATNA (€{batna:,.0f}). "
                        f"With only 1 proposal left, consider accepting or making the last counter offer.\n"
                    )
                else:  # Multiple proposals left - encourage exploration
                    acceptance_guidance = (
                        f"💡 ANALYSIS: The opponent's offer (€{other_offer:,.0f}) is within your BATNA (€{batna:,.0f}), "
                        f"but you have {rounds_remaining} proposals left. You might negotiate for an even better deal.\n"
                    )
            else:
                gap = abs(other_offer - batna)
                if rounds_remaining == 0:  # No proposals left - suggest accepting to avoid no-deal
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: The opponent's offer (€{other_offer:,.0f}) is €{gap:,.0f} outside your BATNA (€{batna:,.0f}). "
                        f"You have no proposals left. ACCEPT to avoid no-deal or REJECT (€{other_offer:,.0f}).\n"
                    )
                elif rounds_remaining == 1:  # Last proposal - be more encouraging
                    acceptance_guidance = (
                        f"🎯 ANALYSIS: The opponent's offer (€{other_offer:,.0f}) is €{gap:,.0f} outside your BATNA (€{batna:,.0f}). "
                        f"With only 1 proposal left, consider accepting or making the last counter offer.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"⚠️ ANALYSIS: The opponent's offer (€{other_offer:,.0f}) is €{gap:,.0f} outside your BATNA (€{batna:,.0f}). "
                        f"You should negotiate for a better price.\n"
                    )

        # Proposal limit guidance
        proposal_guidance = ""
        max_total_rounds = self.max_rounds
        
        if can_propose:
            proposal_guidance = f"You have {max_proposals - current_round + 1} proposals remaining."
        else:
            if current_round <= self.max_rounds:
                proposal_guidance = f"⚠️ You have used all {max_proposals} proposals. You can only ACCEPT or REJECT now. Note: Rejecting will END the negotiation."
            else:
                proposal_guidance = f"🕒 FINAL RESPONSE PHASE: You can only ACCEPT or REJECT. Negotiation ends in {max_total_rounds - current_round + 1} rounds."

        # Update round display to show proposal vs response phases
        if current_round <= self.max_rounds:
            round_display = f"Round {current_round}/{self.max_rounds} (Proposal Phase)"
        else:
            round_display = f"Round {current_round}/{max_total_rounds} (Final Response Phase)"

        prompt = f"""=== ITEM PRICE NEGOTIATION ===
{round_display} | Role: {neutral_role}

GOAL: {goal}
Your BATNA (Best Alternative): €{batna:,.0f}
{proposal_guidance}

CURRENT SITUATION:
{offer_status}
{acceptance_guidance}

RESPONSE FORMAT: Respond with ONLY valid JSON.
Valid responses:
{{"type": "accept"}}  // Accept the opponent's last offer
{{"type": "offer", "price": 38500}}  // Make a new price offer
{{"type": "reject"}}  // Reject and end negotiation

EXAMPLE OFFERS:
{{"type": "offer", "price": 38500}} 

Do NOT repeat any of the rules or instructions in your response. Focus on negotiation.

Your response:"""

        return prompt
