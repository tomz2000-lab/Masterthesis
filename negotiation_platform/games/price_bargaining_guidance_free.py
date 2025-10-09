import random
import re
import json
from typing import Dict, List, Any, Optional, Tuple

from .base_game import BaseGame, PlayerAction
from .negotiation_tools import calculate_percentage_difference, is_offer_above_batna, is_offer_below_batna, suggest_next_offer


class CompanyCarGame(BaseGame):
    """
    Bilateral car negotiation game based on document specifications.
    - Company car worth 45,000€ starting price
    - Buyer max budget 40,000€, seller cost 38,000€
    - BATNAs: Buyer 44,000€ (alternative car cost), Seller 39,000€ (minimum acceptable)
    - 5 rounds with time-adjusted BATNA decay
    - Enhanced with structured prompts, proposal limits, and strategic guidance
    """

    def __init__(self, config: Dict[str, Any]):
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
        """Check if response is valid JSON with proper structure."""
        try:
            data = json.loads(response.strip())
            return isinstance(data, dict) and "type" in data
        except (json.JSONDecodeError, TypeError):
            return False

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse pure JSON response format similar to integrative negotiation game."""
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
        """Validate player action with enhanced structured format support."""
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
            
        max_proposals = self.max_rounds
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
        """Process player actions with proposal limits and enhanced validation."""
        current_round = game_state["current_round"]
        max_proposals = self.max_rounds  # Use rounds from YAML config

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
            game_state[f"{player}_proposal_count"] = player_proposals + 1
            print(f"💡 Player {player} made offer €{price:,.0f} (proposal {player_proposals + 1}/{max_proposals})")

        # Process acceptances (rejections already handled above)
        for player, action in responses.items():
            if action.get("type") == "accept":
                # Find the offer being accepted
                other_player = self.seller if player == self.buyer else self.buyer
                if f"{other_player}_last_offer" in game_state:
                    agreed_price = game_state[f"{other_player}_last_offer"]
                    print(f"✅ Player {player} accepted offer of €{agreed_price:,.0f}")

                    # Validate agreement against BATNAs (more lenient for acceptance)
                    buyer_batna = self.get_current_batna(self.buyer, current_round)
                    seller_batna = self.get_current_batna(self.seller, current_round)

                    # Accept the agreement even if slightly outside BATNA to encourage deals
                    return self._create_agreement(agreed_price, current_round, game_state)
                else:
                    print(f"⚠️ Player {player} tried to accept but no offer exists")

        # Update round
        game_state["current_round"] += 1

        # Check if deadline reached - but allow extra rounds for final responses
        # Players should have a chance to accept/reject final proposals
        grace_rounds = 1  # Allow 1 extra round for final responses after proposal exhaustion
        max_total_rounds = self.max_rounds + grace_rounds
        
        if game_state["current_round"] > max_total_rounds:
            return self._create_no_agreement(game_state)

        return game_state

    def _create_agreement(self, price: float, round_num: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create agreement result."""
        buyer_batna = self.get_current_batna(self.buyer, round_num)
        seller_batna = self.get_current_batna(self.seller, round_num)

        buyer_utility = buyer_batna - price  # Saved money vs BATNA
        seller_utility = price - seller_batna  # Profit over BATNA
        
        # DEBUG: Log the exact calculation values
        print(f"🔍 [BATNA DEBUG] Round {round_num}: price={price}")
        print(f"🔍 [BATNA DEBUG] Config BATNAs: buyer={self.buyer_batna}, seller={self.seller_batna}")
        print(f"🔍 [BATNA DEBUG] Decay rate: {self.batna_decay}")
        print(f"🔍 [BATNA DEBUG] Calculated BATNAs: buyer={buyer_batna:.2f}, seller={seller_batna:.2f}")
        print(f"🔍 [BATNA DEBUG] Utilities: buyer={buyer_utility:.2f}, seller={seller_utility:.2f}")

        game_state.update({
            "agreement_reached": True,
            "game_ended": True,  # Explicitly mark game as ended
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
            "game_ended": True,  # Explicitly mark game as ended
            "final_utilities": {
                self.buyer: 0,  # No deal utility
                self.seller: 0
            }
        })

        return game_state

    def is_game_over(self, game_state: Dict[str, Any]) -> bool:
        """Check if game is finished."""
        return (game_state.get("agreement_reached", False) or
                game_state.get("game_ended", False) or
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
        """Enhanced prompt with structured format, proposal limits, and strategic guidance."""
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
        max_proposals = self.max_rounds  # Use rounds from YAML config
        
        role = "buyer" if player_id == self.buyer else "seller"
        goal = f"Buy car for less than €{batna:,.0f}" if role == "buyer" else f"Sell car for more than €{batna:,.0f}"
        
        offer_history = []
        if my_offer:
            offer_history.append(f"- Your last offer: €{my_offer:,.0f}")
        if other_offer:
            offer_history.append(f"- Opponent's last offer: €{other_offer:,.0f}")
        offer_status = "\n".join(offer_history) if offer_history else "No offers made yet."

        # Enhanced acceptance guidance
        acceptance_guidance = ""
        can_propose = player_proposals < max_proposals
        
        if other_offer is not None:
            percent_diff = calculate_percentage_difference(other_offer, batna)
            is_within_batna = ((player_id == self.buyer and other_offer < batna) or 
                              (player_id == self.seller and other_offer >= batna))
            
            if is_within_batna:
                acceptance_guidance = (
                    f"🎯 STRATEGIC ANALYSIS: The opponent's offer (€{other_offer:,.0f}) is WITHIN your BATNA (€{batna:,.0f}). "
                    f"This is a GOOD DEAL for you! Consider accepting to secure a beneficial agreement, but also try to maximize your gain.\n"
                )
            else:
                if can_propose:
                    acceptance_guidance = (
                        f"⚠️ STRATEGIC ANALYSIS: The opponent's offer (€{other_offer:,.0f}) is OUTSIDE your BATNA (€{batna:,.0f}). "
                        f"You should negotiate for a better price, try to maximize your payoff.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: You've used all {max_proposals} proposals. The opponent's offer is outside your BATNA. "
                        f"You can ACCEPT (even if not ideal) or REJECT (which will END the negotiation).\n"
                    )

        # Proposal limit guidance
        proposal_guidance = ""
        grace_rounds = 1  # Same as in process_actions
        max_total_rounds = self.max_rounds + grace_rounds
        
        if can_propose:
            proposal_guidance = f"You have {max_proposals - player_proposals} proposals remaining."
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

        prompt = f"""=== CAR PRICE NEGOTIATION ===
{round_display} | Role: {role.upper()}

GOAL: {goal}
Your BATNA (Best Alternative): €{batna:,.0f}
{proposal_guidance}

NEGOTIATION STRATEGY:
• Start with ambitious offers but be ready to compromise
• Accept any offer BETTER than your current BATNA, but also try to maximize your surplus
• Use early rounds to explore, later rounds to close the deal
• Don't be afraid to make bold moves if the situation calls for it

CURRENT SITUATION:
{offer_status}
{acceptance_guidance}

RESPONSE FORMAT: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "accept"}}  // Accept the opponent's last offer
{{"type": "offer", "price": [amount]}}  // Make a new price offer
{{"type": "reject"}}  // Reject and end negotiation

EXAMPLE OFFERS:
{{"type": "offer", "price": 38500}}
{{"type": "offer", "price": 42758}}

Your response:"""

        return prompt
