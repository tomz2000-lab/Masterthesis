import random
from typing import Dict, List, Any, Optional, Tuple
from .base_game import BaseGame


class CompanyCarGame(BaseGame):
    """
    Bilateral car negotiation game based on document specifications.
    - Company car worth 45,000€ starting price
    - Buyer max budget 40,000€, seller cost 38,000€
    - BATNAs: Buyer 41,000€, Seller 39,000€
    - 5 rounds with time-adjusted BATNA decay
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.starting_price = config.get("starting_price", 45000)
        self.buyer_budget = config.get("buyer_budget", 40000)
        self.seller_cost = config.get("seller_cost", 38000)
        self.buyer_batna = config.get("buyer_batna", 41000)
        self.seller_batna = config.get("seller_batna", 39000)
        self.max_rounds = config.get("rounds", 5)
        self.batna_decay = config.get("batna_decay", {"buyer": 0.02, "seller": 0.01})

    def initialize_game(self, players: List[str]) -> Dict[str, Any]:
        """Initialize bilateral car negotiation."""
        if len(players) != 2:
            raise ValueError("Company car game requires exactly 2 players")

        self.players = players
        self.buyer = players[0]  # First player is buyer
        self.seller = players[1]  # Second player is seller

        return {
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

            # Check if offer meets BATNA requirement
            current_batna = self.get_current_batna(player, game_state["current_round"])

            if player == self.buyer:
                return price <= current_batna  # Buyer won't pay more than BATNA
            else:
                return price >= current_batna  # Seller won't accept less than BATNA

        elif action_type in ["accept", "reject"]:
            return True

        return False

    def process_actions(self, actions: Dict[str, Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process player actions and update game state."""
        current_round = game_state["current_round"]

        # Check for offers and responses
        offers = {player: action for player, action in actions.items()
                  if action.get("type") == "offer"}
        responses = {player: action for player, action in actions.items()
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
