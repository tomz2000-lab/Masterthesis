import random
from typing import Dict, List, Any, Optional, Tuple
from .base_game import BaseGame, PlayerAction


class CompanyCarGame(BaseGame):
    """
    Bilateral car negotiation game based on document specifications.
    - Company car worth 45,000€ starting price
    - Buyer max budget 40,000€, seller cost 38,000€
    - BATNAs: Buyer 44,000€ (alternative car cost), Seller 39,000€ (minimum acceptable)
    - 5 rounds with time-adjusted BATNA decay
    """

    def __init__(self, config: Dict[str, Any]):
        # Initialize base class with dummy game_id - will be set by game engine
        super().__init__(game_id="company_car", config=config)
        self.starting_price = config.get("starting_price", 45000)
        self.buyer_budget = config.get("buyer_budget", 40000)
        self.seller_cost = config.get("seller_cost", 38000)
        # Fixed BATNA logic: Buyer BATNA is what they'd pay elsewhere (higher than their desired price)
        self.buyer_batna = config.get("buyer_batna", 44000)  # Alternative car costs more
        self.seller_batna = config.get("seller_batna", 39000)  # Minimum they'll accept
        self.max_rounds = config.get("rounds", 5)
        self.batna_decay = config.get("batna_decay", {"buyer": 0.02, "seller": 0.01})

    def initialize_game(self, players: List[str]) -> Dict[str, Any]:
        """Initialize bilateral car negotiation."""
        if len(players) != 2:
            raise ValueError("Company car game requires exactly 2 players")

        self.players = players
        self.buyer = players[0]  # First player is buyer
        self.seller = players[1]  # Second player is seller
        self.state = self.state.__class__.ACTIVE  # Set to active state

        self.game_data = {
            "game_type": "company_car",
            "players": self.players,
            "rounds": self.max_rounds,
            "current_round": 1,
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
        """Calculate time-adjusted BATNA for current round."""
        if player == self.buyer:
            decay_rate = self.batna_decay["buyer"]
            base_batna = self.buyer_batna
        else:
            decay_rate = self.batna_decay["seller"]
            base_batna = self.seller_batna

        return base_batna * ((1 - decay_rate) ** (round_num - 1))

    def is_valid_action(self, player: str, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Validate player action."""
        action_type = action.get("type", "")

        if action_type == "offer":
            price = action.get("price", 0)
            if price <= 0:
                return False

            # Basic range validation - offers should be reasonable
            if price > 100000 or price < 10000:  # Sanity check
                return False

            # For now, accept any reasonable price offer
            # The economic logic will be handled in the AI's decision making
            return True

        elif action_type in ["accept", "reject"]:
            return True

        elif action_type in ["counter", "counteroffer"]:
            # Treat counter/counteroffer as regular offers
            price = action.get("price", 0)
            return price > 0 and 10000 <= price <= 100000

        elif action_type in ["offer_accepted", "offer_response"]:
            # Treat these as accept actions
            return True

        elif action_type == "noop":
            # Allow no-op actions as fallback
            return True

        return False

    def process_actions(self, actions: Dict[str, Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process player actions and update game state."""
        current_round = game_state["current_round"]

        # Normalize action types - treat counter/counteroffer as offers
        normalized_actions = {}
        for player, action in actions.items():
            action_type = action.get("type", "")
            if action_type in ["counter", "counteroffer"]:
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

        # Process offers
        for player, action in offers.items():
            price = action.get("price")
            game_state[f"{player}_last_offer"] = price

        # Process acceptances
        for player, action in responses.items():
            if action.get("type") == "accept":
                # Find the offer being accepted
                other_player = self.seller if player == self.buyer else self.buyer
                if f"{other_player}_last_offer" in game_state:
                    agreed_price = game_state[f"{other_player}_last_offer"]

                    # Validate agreement against BATNAs
                    buyer_batna = self.get_current_batna(self.buyer, current_round)
                    seller_batna = self.get_current_batna(self.seller, current_round)

                    if agreed_price <= buyer_batna and agreed_price >= seller_batna:
                        return self._create_agreement(agreed_price, current_round, game_state)

        # Update round
        game_state["current_round"] += 1

        # Check if deadline reached
        if game_state["current_round"] > self.max_rounds:
            return self._create_no_agreement(game_state)

        return game_state

    def _create_agreement(self, price: float, round_num: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create agreement result."""
        buyer_batna = self.get_current_batna(self.buyer, round_num)
        seller_batna = self.get_current_batna(self.seller, round_num)

        buyer_utility = buyer_batna - price  # Saved money vs BATNA
        seller_utility = price - seller_batna  # Profit over BATNA

        game_state.update({
            "agreement_reached": True,
            "agreed_price": price,
            "agreement_round": round_num,
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
        """Create no agreement result."""
        game_state.update({
            "agreement_reached": False,
            "final_utilities": {
                self.buyer: 0,  # No deal utility
                self.seller: 0
            }
        })

        return game_state

    def is_game_over(self, game_state: Dict[str, Any]) -> bool:
        """Check if game is finished."""
        return (game_state.get("agreement_reached", False) or
                game_state.get("current_round", 1) > self.max_rounds)

    def get_winner(self, game_state: Dict[str, Any]) -> Optional[str]:
        """Determine winner based on utilities."""
        if not game_state.get("agreement_reached", False):
            return None

        utilities = game_state.get("final_utilities", {})
        if not utilities:
            return None

        return max(utilities, key=utilities.get)

    # Required abstract methods from BaseGame
    def process_action(self, action: PlayerAction) -> Dict[str, Any]:
        """Process a player action and update game state"""
        # This method should process individual actions
        # For now, delegate to process_actions with single action
        player_id = action.player_id if hasattr(action, 'player_id') else 'unknown'
        action_data = action.action_data if hasattr(action, 'action_data') else action
        
        return self.process_actions({player_id: action_data}, self.game_data)

    def process_action(self, action) -> Dict[str, Any]:
        """Process a player action and update game state"""
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
        """Check if the game should end"""
        return self.is_game_over(self.game_data)

    def calculate_scores(self) -> Dict[str, float]:
        """Calculate final scores for all players"""
        if self.game_data.get("agreement_reached", False):
            return self.game_data.get("final_utilities", {})
        else:
            return {player: 0.0 for player in self.players}

    def get_game_prompt(self, player_id: str) -> str:
        """Get the current game prompt for a specific player"""
        if not hasattr(self, 'buyer') or not hasattr(self, 'seller'):
            return "Game not initialized properly"
            
        private_info = self.game_data.get("private_info", {}).get(player_id, {})
        current_round = self.game_data.get("current_round", 1)
        
        # Get previous offers from game state
        other_player = self.seller if player_id == self.buyer else self.buyer
        other_offer = self.game_data.get(f"{other_player}_last_offer", None)
        my_offer = self.game_data.get(f"{player_id}_last_offer", None)
        
        # Build offer history with explicit accept/reject guidance  
        offer_history = []
        accept_guidance = ""
        
        if my_offer:
            role_name = "You"
            offer_history.append(f"- Your last offer: €{my_offer:,.0f}")
            
        if other_offer:
            other_role = "Seller" if other_player == self.seller else "Buyer"  
            offer_history.append(f"- {other_role}'s last offer: €{other_offer:,.0f}")
            
            # Generate explicit accept/reject guidance
            if player_id == self.buyer:
                batna = self.get_current_batna(player_id, current_round)
                if other_offer <= batna:
                    accept_guidance = f"\n→ ACCEPT this offer (saves you €{batna - other_offer:,.0f})"
                else:
                    accept_guidance = f"\n→ Offer is too high, counter-offer below €{batna:,.0f}"
                    
            else:  # seller
                batna = self.get_current_batna(player_id, current_round)
                if other_offer >= batna:
                    accept_guidance = f"\n→ ACCEPT this offer (profit €{other_offer - batna:,.0f})"
                else:
                    accept_guidance = f"\n→ Offer is too low, counter-offer above €{batna:,.0f}"
        
        offer_status = "\n".join(offer_history) if offer_history else "No offers made yet."
        offer_status += accept_guidance
        
        if player_id == self.buyer:
            budget = private_info.get("budget", self.buyer_budget)
            batna = self.get_current_batna(player_id, current_round)
            
            if other_offer:
                decision_guidance = "ACCEPT IT" if other_offer <= batna else "COUNTER-OFFER"
                return f"""You are a BUYER in a car negotiation. Round {current_round}/{self.max_rounds}.
Your maximum budget: €{batna:,.0f}
Seller's offer: €{other_offer:,.0f}
Decision: {decision_guidance} (offer is {'acceptable' if other_offer <= batna else 'too high'})

TASK: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "accept"}}
{{"type": "offer", "price": 40000}}
{{"type": "reject"}}

Your response:"""
            else:
                return f"""You are a BUYER in a car negotiation. Round {current_round}/{self.max_rounds}.
Your maximum budget: €{batna:,.0f}
No offers yet. Make a reasonable offer below your budget.

TASK: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "offer", "price": 40000}}
{{"type": "reject"}}

Your response:"""

        else:  # seller
            cost = private_info.get("cost", self.seller_cost)
            batna = self.get_current_batna(player_id, current_round)
            
            if other_offer:
                decision_guidance = "ACCEPT IT" if other_offer >= batna else "COUNTER-OFFER"
                return f"""You are a SELLER in a car negotiation. Round {current_round}/{self.max_rounds}.
Your minimum price: €{batna:,.0f}
Buyer's offer: €{other_offer:,.0f}
Decision: {decision_guidance} (offer is {'acceptable' if other_offer >= batna else 'too low'})

TASK: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "accept"}}
{{"type": "offer", "price": 42000}}
{{"type": "reject"}}

Your response:"""
            else:
                return f"""You are a SELLER in a car negotiation. Round {current_round}/{self.max_rounds}.
Your minimum price: €{batna:,.0f}
No offers yet. Make a reasonable offer above your minimum.

TASK: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "offer", "price": 42000}}
{{"type": "reject"}}

Your response:"""
