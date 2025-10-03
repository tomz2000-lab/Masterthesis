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
        self.starting_price = config.get("starting_price", 45000)
        self.buyer_budget = config.get("buyer_budget", 40000)
        self.seller_cost = config.get("seller_cost", 38000)
        # Use configuration values with balanced fallbacks
        self.buyer_batna = config.get("buyer_batna", 42000)  # Balanced fallback
        self.seller_batna = config.get("seller_batna", 40000)  # Balanced fallback
        self.max_rounds = config.get("rounds", 5)
        # Use configuration decay rates with balanced fallbacks
        self.batna_decay = config.get("batna_decay", {"buyer": 0.015, "seller": 0.015})

    def parse_structured_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response with REASONING, DECISION, and MESSAGE tags."""
        try:
            # Extract reasoning
            reasoning_match = re.search(r'<REASONING>(.*?)</REASONING>', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
            # Extract decision (JSON)
            decision_match = re.search(r'<DECISION>(.*?)</DECISION>', response, re.DOTALL)
            decision_text = decision_match.group(1).strip() if decision_match else ""
            
            # Extract message
            message_match = re.search(r'<MESSAGE>(.*?)</MESSAGE>', response, re.DOTALL)
            message = message_match.group(1).strip() if message_match else ""
            
            # Parse JSON decision
            # Clean up the decision text and extract JSON
            json_match = re.search(r'\{[^}]*\}', decision_text)
            if json_match:
                json_str = json_match.group(0)
                decision_data = json.loads(json_str)
            else:
                # Fallback: try to parse the entire decision as JSON
                decision_data = json.loads(decision_text)
            
            return {
                "reasoning": reasoning,
                "decision": decision_data,
                "message": message,
                "raw_response": response
            }
            
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"⚠️ Failed to parse structured response: {e}")
            print(f"Raw response: {response}")
            
            # Fallback: try to extract JSON from anywhere in the response
            json_match = re.search(r'\{[^}]*"type"[^}]*\}', response)
            if json_match:
                try:
                    fallback_decision = json.loads(json_match.group(0))
                    return {
                        "reasoning": "Failed to parse structured format",
                        "decision": fallback_decision,
                        "message": "",
                        "raw_response": response
                    }
                except json.JSONDecodeError:
                    pass
            
            # Ultimate fallback: return a rejection
            return {
                "reasoning": "Failed to parse response",
                "decision": {"type": "reject"},
                "message": "Could not parse response",
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
        max_proposals = 3
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
        max_proposals = 3

        # Initialize proposal counters if not present
        for player in [self.buyer, self.seller]:
            if f"{player}_proposal_count" not in game_state:
                game_state[f"{player}_proposal_count"] = 0

        # Process structured responses and extract decision data
        processed_actions = {}
        for player, raw_action in actions.items():
            if isinstance(raw_action, str):
                # If it's a string, try to parse it as structured response
                parsed = self.parse_structured_response(raw_action)
                action_data = parsed["decision"]
                # Store reasoning and message for logging
                if parsed["reasoning"]:
                    print(f"🧠 Player {player} reasoning: {parsed['reasoning'][:100]}...")
                if parsed["message"]:
                    print(f"💬 Player {player} message: {parsed['message']}")
            elif isinstance(raw_action, dict) and "decision" in raw_action:
                # Already structured
                action_data = raw_action["decision"]
            else:
                # Regular action format
                action_data = raw_action
            
            processed_actions[player] = action_data

        # Normalize action types - treat counter/counteroffer as offers
        normalized_actions = {}
        for player, action in processed_actions.items():
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

        # Process offers with proposal limit validation
        for player, action in offers.items():
            player_proposals = game_state.get(f"{player}_proposal_count", 0)
            
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} exceeded proposal limit ({player_proposals}/{max_proposals}). Treating as reject.")
                # Convert to reject if they've exceeded proposals
                if player not in responses:
                    responses[player] = {"type": "reject"}
                continue
            
            # Valid offer - process it
            price = action.get("price")
            game_state[f"{player}_last_offer"] = price
            game_state[f"{player}_proposal_count"] = player_proposals + 1
            print(f"💡 Player {player} made offer €{price:,.0f} (proposal {player_proposals + 1}/{max_proposals})")

        # Process acceptances and rejections
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
            
            elif action.get("type") == "reject":
                print(f"❌ Player {player} rejected the negotiation")
                return self._create_no_agreement(game_state)

        # Check if both players have exhausted proposals
        buyer_proposals = game_state.get(f"{self.buyer}_proposal_count", 0)
        seller_proposals = game_state.get(f"{self.seller}_proposal_count", 0)
        
        if buyer_proposals >= max_proposals and seller_proposals >= max_proposals:
            print(f"⏰ Both players exhausted proposals ({buyer_proposals}/{max_proposals}, {seller_proposals}/{max_proposals}). Ending negotiation.")
            return self._create_no_agreement(game_state)

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
        max_proposals = 3  # Limit proposals to force decision making
        
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
                    f"This is a GOOD DEAL for you! Consider accepting to secure a beneficial agreement.\n"
                )
            else:
                if can_propose:
                    acceptance_guidance = (
                        f"⚠️ STRATEGIC ANALYSIS: The opponent's offer (€{other_offer:,.0f}) is OUTSIDE your BATNA (€{batna:,.0f}). "
                        f"You should negotiate for a better price.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: You've used all {max_proposals} proposals. The opponent's offer is outside your BATNA. "
                        f"You must now ACCEPT (if acceptable) or REJECT and end the negotiation.\n"
                    )

        # Proposal limit guidance
        proposal_guidance = ""
        if can_propose:
            proposal_guidance = f"You have {max_proposals - player_proposals} proposals remaining."
        else:
            proposal_guidance = f"⚠️ You have used all {max_proposals} proposals. You can only ACCEPT or REJECT now."

        # Build the offer option text based on proposal limits
        offer_option = f'{{"type": "offer", "price": [amount]}}  // Make a new price offer'
        if can_propose:
            offer_option += " // Only if you have proposals left"
        else:
            offer_option += " // NOT ALLOWED - no proposals left"

        prompt = f"""=== CAR PRICE NEGOTIATION ===
Round {current_round}/{self.max_rounds} | Role: {role.upper()}

GOAL: {goal}
Your BATNA (Best Alternative): €{batna:,.0f}
{proposal_guidance}

CURRENT SITUATION:
{offer_status}
{acceptance_guidance}
You must respond with a structured format that includes your reasoning and decision.

RESPONSE FORMAT:
```
<REASONING>
[Explain your strategic thinking: Why are you making this decision? How does it relate to your BATNA and goals?]
</REASONING>

<DECISION>
[Choose exactly ONE of the following JSON responses:]
{{"type": "accept"}}  // Accept the opponent's last offer
{offer_option}
{{"type": "reject"}}  // Reject and end negotiation
</DECISION>

<MESSAGE>
[Optional: Send a message to your opponent explaining your position or trying to persuade them]
</MESSAGE>
```

EXAMPLE RESPONSE:
```
<REASONING>
The opponent's offer of €39,000 is within my BATNA of €44,000, meaning I would save €5,000 compared to my alternative. This is a good deal that achieves my goal of buying below my BATNA.
</REASONING>

<DECISION>
{{"type": "accept"}}
</DECISION>

<MESSAGE>
I accept your offer of €39,000. This works well for both of us!
</MESSAGE>
```

Your response:"""
        return prompt
