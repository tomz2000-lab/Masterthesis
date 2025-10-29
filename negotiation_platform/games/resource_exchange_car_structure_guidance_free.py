import random
import re
import json
import sys
from typing import Dict, List, Any, Optional
from .base_game import BaseGame, PlayerAction


class ResourceAllocationGame(BaseGame):
    """
    Resource allocation negotiation game between Development and Marketing teams.
    
    ADAPTED FROM INTEGRATIVE NEGOTIATIONS: Uses similar structure to integrative negotiation
    and price bargaining games but focuses on resource allocation:
    - Two resources: GPU hours and CPU hours 
    - Utility functions: Development (8x + 6y), Marketing (6x + 8y)
    - Constraints: total resources, GPU-bandwidth limit, minimum allocations
    - 5 rounds with time-adjusted BATNA decay
    - Enhanced with structured prompts, proposal limits, and strategic guidance
    """

    def __init__(self, config: Dict[str, Any]):
        # Initialize base class with game type as game_id  
        super().__init__(game_id="resource_allocation", config=config)
        required_fields = [
            "batnas", "rounds", "batna_decay", "total_resources", "constraints"
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Extract BATNAs from the batnas dictionary
        batnas = config["batnas"]
        self.development_batna = batnas["development"]
        self.marketing_batna = batnas["marketing"]
        self.max_rounds = config["rounds"]
        self.batna_decay = config["batna_decay"]
        
        # Resource allocation specific configuration
        self.total_resources = config["total_resources"]
        self.constraints = config["constraints"]
        
        # Uncertainty parameters (optional)
        self.uncertainty = config.get("uncertainty", {})

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
            
            # Try to extract type and resource allocation manually as fallback
            type_match = re.search(r'"type":\s*"([^"]+)"', response)
            gpu_match = re.search(r'"gpu_hours":\s*(\d+(?:\.\d+)?)', response)
            cpu_match = re.search(r'"cpu_hours":\s*(\d+(?:\.\d+)?)', response)
            if type_match:
                decision_data = {"type": type_match.group(1)}
                if gpu_match and cpu_match:
                    decision_data["gpu_hours"] = float(gpu_match.group(1))
                    decision_data["cpu_hours"] = float(cpu_match.group(1))
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
        """Initialize resource allocation negotiation with randomized role assignment."""
        if len(players) != 2:
            raise ValueError("Resource allocation game requires exactly 2 players")

        self.players = players
        
        # Randomly assign development and marketing roles to eliminate role-based bias
        if random.choice([True, False]):
            self.development = players[0]
            self.marketing = players[1]
            print(f"🎲 [ROLE ASSIGNMENT] {players[0]} = DEVELOPMENT, {players[1]} = MARKETING")
        else:
            self.development = players[1] 
            self.marketing = players[0]
            print(f"🎲 [ROLE ASSIGNMENT] {players[1]} = DEVELOPMENT, {players[0]} = MARKETING")
            
        self.state = self.state.__class__.ACTIVE  # Set to active state

        self.game_data = {
            "game_type": "resource_allocation",
            "players": self.players,
            "rounds": self.max_rounds,
            "current_round": 1,
            "role_assignments": {
                "development": self.development,
                "marketing": self.marketing
            },
            "private_info": {
                self.development: {
                    "role": "development",
                    "team": "development",
                    "utility_function": "8x + 6y + ε",
                    "batna": self.development_batna,
                    "constraints": self.constraints
                },
                self.marketing: {
                    "role": "marketing", 
                    "team": "marketing",
                    "utility_function": "6x + 8y + ι", 
                    "batna": self.marketing_batna,
                    "constraints": self.constraints
                }
            },
            "public_info": {
                "deadline": self.max_rounds,
                "total_resources": self.total_resources,
                "constraints": self.constraints
            }
        }
        return self.game_data

    def get_current_batna(self, player: str, round_num: int) -> float:
        """Calculate time-adjusted BATNA for current round."""
        if player == self.development:
            decay_rate = self.batna_decay["development"]
            base_batna = self.development_batna
        else:
            decay_rate = self.batna_decay["marketing"]
            base_batna = self.marketing_batna

        return base_batna * ((1 - decay_rate) ** (round_num-1))

    def calculate_utility(self, player: str, gpu_hours: float, cpu_hours: float, round_num: int) -> float:
        """Calculate utility for a player given resource allocation."""
        if player == self.development:
            # Development team utility: 9x + 6y + ε (epsilon for uncertainty)
            base_utility = 8 * gpu_hours + 6 * cpu_hours
            # Add small random uncertainty factor
            epsilon = random.uniform(-2, 2)
            return base_utility + epsilon
        else:
            # Marketing team utility: 6x + 9y + i (iota for market volatility)
            base_utility = 6 * gpu_hours + 8 * cpu_hours
            # Add small random uncertainty factor
            iota = random.uniform(-2, 2)
            return base_utility + iota

    def _validate_resource_constraints(self, gpu_hours: float, cpu_hours: float) -> bool:
        """Validate resource allocation against constraints."""
        # Total resource constraint: x + y <= total_resources
        if gpu_hours + cpu_hours > self.total_resources:
            return False

        # GPU-CPU constraint: 4x + 4y <= gpu_bandwidth
        if 4 * gpu_hours + 4 * cpu_hours > self.constraints["gpu_bandwidth"]:
            return False
            
        # Minimum allocation constraints
        if gpu_hours < self.constraints["min_gpu"]:
            return False
        if cpu_hours < self.constraints["min_cpu"]:
            return False
            
        return True

    def check_constraints_and_update(self, gpu_hours: float, cpu_hours: float) -> None:
        """Check constraints and update game data with the result."""
        constraints_met = True
        messages = []

        # Check total resource constraint
        if gpu_hours + cpu_hours > self.total_resources:
            constraints_met = False
            messages.append(f"Total resources exceeded: {gpu_hours + cpu_hours} > {self.total_resources}")

        # Check GPU-CPU constraint
        if 4 * gpu_hours + 4 * cpu_hours > self.constraints["gpu_bandwidth"]:
            constraints_met = False
            messages.append(f"GPU-Bandwidth limit exceeded: {4 * gpu_hours + 4 * cpu_hours} > {self.constraints['gpu_bandwidth']}")

        # Check minimum allocation constraints
        if gpu_hours < self.constraints["min_gpu"]:
            constraints_met = False
            messages.append(f"GPU hours below minimum: {gpu_hours} < {self.constraints['min_gpu']}")
        if cpu_hours < self.constraints["min_cpu"]:
            constraints_met = False
            messages.append(f"CPU hours below minimum: {cpu_hours} < {self.constraints['min_cpu']}")

        # Update game data
        self.game_data["constraints_met"] = constraints_met

        # Print results
        if constraints_met:
            print("✅ All constraints are satisfied.")
        else:
            print("❌ Constraints violated:")
            for message in messages:
                print(f"   - {message}")

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
            
        max_proposals = self.max_rounds-1
        player_proposals = game_state.get(f"{player}_proposal_count", 0)

        if action_type in ["offer", "propose"]:  # Accept both "offer" and "propose"
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} tried to make offer but exceeded proposal limit ({player_proposals}/{max_proposals})")
                return False
                
            gpu_hours = action_data.get("gpu_hours", 0)
            cpu_hours = action_data.get("cpu_hours", 0)

            if gpu_hours <= 0 or cpu_hours <= 0:
                return False

            # Validate constraints
            if not self._validate_resource_constraints(gpu_hours, cpu_hours):
                print(f"⚠️ Player {player} offer violates constraints: GPU={gpu_hours}, CPU={cpu_hours}")
                return False

            return True

        elif action_type in ["accept", "reject"]:
            return True

        elif action_type in ["counter", "counteroffer"]:
            # Check proposal limit for counters too
            if player_proposals >= max_proposals:
                return False
                
            # Treat counter/counteroffer as regular offers
            gpu_hours = action_data.get("gpu_hours", 0)
            cpu_hours = action_data.get("cpu_hours", 0)
            return (gpu_hours > 0 and cpu_hours > 0 and
                   self._validate_resource_constraints(gpu_hours, cpu_hours))

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
        max_proposals = self.max_rounds-1  # Use rounds from YAML config
        
        print(f"🔍 Processing actions for round {current_round}: {actions}", file=sys.stderr)

        # Initialize proposal counters if not present
        for player in [self.development, self.marketing]:
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
                normalized_actions[player] = {
                    "type": "offer", 
                    "gpu_hours": action.get("gpu_hours"),
                    "cpu_hours": action.get("cpu_hours")
                }
            elif action_type == "propose":
                # Convert "propose" to "offer" for consistency
                normalized_actions[player] = {
                    "type": "offer", 
                    "gpu_hours": action.get("gpu_hours"),
                    "cpu_hours": action.get("cpu_hours")
                }
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

        # Process acceptances FIRST (before new offers) to ensure only previous round offers can be accepted
        for player, action in responses.items():
            if action.get("type") == "accept":
                # Find the offer being accepted
                other_player = self.marketing if player == self.development else self.development
                if f"{other_player}_last_offer" in game_state:
                    offer_data = game_state[f"{other_player}_last_offer"]
                    gpu_hours = offer_data["gpu_hours"]
                    cpu_hours = offer_data["cpu_hours"]
                    # Get the round when the accepted offer was made
                    offer_round = game_state.get(f"{other_player}_last_offer_round", current_round)
                    
                    # Validate that the offer being accepted was made in a previous round
                    if offer_round >= current_round:
                        print(f"⚠️ Player {player} tried to accept offer made in same round {offer_round}. Offers can only be accepted from previous rounds.")
                        continue  # Skip this acceptance, don't end the game

                    print(f"✅ Player {player} accepted offer of GPU={gpu_hours}, CPU={cpu_hours} (made in round {offer_round})")

                    # Use the BATNA from when the offer was made, but record agreement as happening in current round
                    return self._create_agreement(gpu_hours, cpu_hours, current_round, game_state, offer_round_for_batna=offer_round)
                else:
                    print(f"⚠️ Player {player} tried to accept but no offer exists")

        # Process offers with proposal limit validation (AFTER acceptances)
        for player, action in offers.items():
            player_proposals = game_state.get(f"{player}_proposal_count", 0)
            
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} exceeded proposal limit ({player_proposals}/{max_proposals}). Ignoring additional offers.")
                # Don't process this offer, but don't end negotiation unless they also rejected
                continue
            
            # Valid offer - process it
            gpu_hours = action.get("gpu_hours")
            cpu_hours = action.get("cpu_hours")
            game_state[f"{player}_last_offer"] = {"gpu_hours": gpu_hours, "cpu_hours": cpu_hours}
            game_state[f"{player}_last_offer_round"] = current_round  # Track when offer was made
            game_state[f"{player}_proposal_count"] = player_proposals + 1
            print(f"💡 Player {player} made offer GPU={gpu_hours}, CPU={cpu_hours} (proposal {player_proposals + 1}/{max_proposals})")

        # Check for convergence: if both players made identical offers, create agreement
        if len(offers) == 2:  # Both players made offers this round
            offer_data = [action for action in offers.values()]
            if (len(offer_data) == 2 and 
                offer_data[0].get("gpu_hours") == offer_data[1].get("gpu_hours") and
                offer_data[0].get("cpu_hours") == offer_data[1].get("cpu_hours") and
                offer_data[0].get("gpu_hours") is not None):  # All offers are identical and valid
                gpu_hours = offer_data[0].get("gpu_hours")
                cpu_hours = offer_data[0].get("cpu_hours")
                print(f"🎉 CONVERGENCE! Both players offered GPU={gpu_hours}, CPU={cpu_hours} - Creating automatic agreement!")
                return self._create_agreement(gpu_hours, cpu_hours, current_round, game_state)

        # Update round
        game_state["current_round"] += 1
        
        # Debug logging for round progression
        print(f"🔄 Updated to round {game_state['current_round']}/{self.max_rounds}", file=sys.stderr)
        
        # Check proposal status for both players
        dev_proposals = game_state.get(f"{self.development}_proposal_count", 0)
        mkt_proposals = game_state.get(f"{self.marketing}_proposal_count", 0)
        print(f"🔍 Proposal status: {self.development}={dev_proposals}/4, {self.marketing}={mkt_proposals}/4", file=sys.stderr)

        # Check if deadline reached - but allow extra rounds for final responses
        # Players should have a chance to accept/reject final proposals
        max_total_rounds = self.max_rounds
        
        if game_state["current_round"] > max_total_rounds:
            print(f"⏰ Maximum rounds ({max_total_rounds}) reached - Ending negotiation")
            return self._create_no_agreement(game_state)

        return game_state

    def _create_agreement(self, gpu_hours: float, cpu_hours: float, current_round: int, game_state: Dict[str, Any], offer_round_for_batna: int = None) -> Dict[str, Any]:
        """Create agreement result.
        
        Args:
            gpu_hours: GPU hours in the agreement
            cpu_hours: CPU hours in the agreement
            current_round: Round when agreement was reached (for recording)
            game_state: Current game state
            offer_round_for_batna: Round when the accepted offer was made (for BATNA calculation)
        """
        # Use offer_round_for_batna if provided, otherwise use current_round
        batna_round = offer_round_for_batna if offer_round_for_batna is not None else current_round
        
        dev_batna = self.get_current_batna(self.development, batna_round)
        mkt_batna = self.get_current_batna(self.marketing, batna_round)

        # Calculate utility using the resource allocation utility functions
        # Development: 8x + 6x, Marketing: 6x + 8x
        dev_utility = 8 * gpu_hours + 6 * cpu_hours
        mkt_utility = 6 * gpu_hours + 8 * cpu_hours

        # Calculate utility surplus (utility - BATNA)
        dev_surplus = dev_utility - dev_batna
        mkt_surplus = mkt_utility - mkt_batna
        
        # DEBUG: Log the exact calculation values
        print(f"🔍 [RESOURCE DEBUG] Agreement in round {current_round}: GPU={gpu_hours}, CPU={cpu_hours}")
        if offer_round_for_batna is not None and offer_round_for_batna != current_round:
            print(f"🔍 [BATNA DEBUG] Using BATNA from offer round {batna_round} (offer made), agreement in round {current_round}")
        else:
            print(f"🔍 [BATNA DEBUG] Using BATNA from round {batna_round}")
        print(f"🔍 [BATNA DEBUG] Config BATNAs: development={self.development_batna}, marketing={self.marketing_batna}")
        print(f"🔍 [BATNA DEBUG] Decay rates: development={self.batna_decay['development']}, marketing={self.batna_decay['marketing']}")
        print(f"🔍 [BATNA DEBUG] Calculated BATNAs: development={dev_batna:.2f}, marketing={mkt_batna:.2f}")
        print(f"🔍 [UTILITY DEBUG] Utilities: development={dev_utility:.2f}, marketing={mkt_utility:.2f}")
        print(f"🔍 [SURPLUS DEBUG] Surpluses: development={dev_surplus:.2f}, marketing={mkt_surplus:.2f}")
        print(f"🎲 [ROLE DEBUG] Development={self.development}, Marketing={self.marketing}")

        game_state.update({
            "agreement_reached": True,
            "game_ended": True,  # Explicitly mark game as ended
            "agreed_allocation": {
                "gpu_hours": gpu_hours,
                "cpu_hours": cpu_hours
            },
            "agreement_round": current_round,
            "role_assignments": {
                "development": self.development,
                "marketing": self.marketing
            },
            "final_utilities": {
                self.development: dev_utility,
                self.marketing: mkt_utility
            },
            "utility_surpluses": {
                self.development: dev_surplus,
                self.marketing: mkt_surplus
            },
            "batnas_at_agreement": {
                self.development: dev_batna,
                self.marketing: mkt_batna
            }
        })

        return game_state

    def _create_no_agreement(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create no agreement result."""
        print(f"🎲 [ROLE DEBUG] No Agreement - Development={self.development}, Marketing={self.marketing}")
        print(f"🎲 [ROLE DEBUG] {self.development} utility=0, {self.marketing} utility=0")
        
        game_state.update({
            "agreement_reached": False,
            "game_ended": True,  # Explicitly mark game as ended
            "role_assignments": {
                "development": self.development,
                "marketing": self.marketing
            },
            "final_utilities": {
                self.development: 0.0,  # No deal utility
                self.marketing: 0.0
            }
        })
        return game_state

    def is_game_over(self, game_state: Dict[str, Any]) -> bool:
        """Check if game is finished."""
        current_round = game_state.get("current_round", 1)
        agreement_reached = game_state.get("agreement_reached", False)
        game_ended = game_state.get("game_ended", False)
        
        print(f"🔍 is_game_over check: round={current_round}/{self.max_rounds}, agreement={agreement_reached}, ended={game_ended}", file=sys.stderr)
        
        result = (agreement_reached or game_ended or current_round > self.max_rounds)
        print(f"🔍 is_game_over result: {result}", file=sys.stderr)
        
        return result

    def get_game_prompt(self, player_id: str) -> str:
        """Enhanced prompt with structured format, proposal limits, and strategic guidance."""
        if not hasattr(self, 'development') or not hasattr(self, 'marketing'):
            return "Game not initialized properly"

        private_info = self.game_data.get("private_info", {}).get(player_id, {})
        current_round = self.game_data.get("current_round", 1)
        other_player = self.marketing if player_id == self.development else self.development
        other_offer = self.game_data.get(f"{other_player}_last_offer", None)
        my_offer = self.game_data.get(f"{player_id}_last_offer", None)
        batna = self.get_current_batna(player_id, current_round)
        
        # Track proposals made by this player
        player_proposals = self.game_data.get(f"{player_id}_proposal_count", 0)
        max_proposals = self.max_rounds - 1
        can_propose = player_proposals < max_proposals
        
        role = private_info.get("role", "unknown")
        team_name = "Development Team" if role == "development" else "Marketing Team"
        utility_func = private_info.get("utility_function", "unknown")
        preference = "GPU-heavy tasks" if role == "development" else "CPU-intensive operations"
        
        # Offer status
        offer_history = []
        if my_offer:
            gpu = my_offer.get("gpu_hours", 0)
            cpu = my_offer.get("cpu_hours", 0)
            offer_history.append(f"- Your last offer: GPU={gpu}, CPU={cpu}")
        if other_offer:
            gpu = other_offer.get("gpu_hours", 0)
            cpu = other_offer.get("cpu_hours", 0)
            offer_history.append(f"- Opponent's offer: GPU={gpu}, CPU={cpu}")
        offer_status = "\n".join(offer_history) if offer_history else "No offers made yet."

        # Role-specific configuration
        role_priorities = ""
        if role == "development":
            role_priorities = (
                f"Your priority is to maximize GPU hours for development tasks.\n"
            )

        else:  # Marketing
            role_priorities = (
                f"Your priority is to maximize CPU hours for marketing tasks.\n"
            )

        # Acceptance guidance
        acceptance_guidance = ""
        proposal_guidance = f"📊 You have **{max_proposals - player_proposals}** proposals remaining out of {max_proposals} total."
        rounds_remaining = max_proposals - player_proposals
        
        if other_offer is not None:
            # Calculate utility for the proposed offer
            gpu_hours = other_offer.get("gpu_hours", 0)
            cpu_hours = other_offer.get("cpu_hours", 0)
            proposed_utility = self.calculate_utility(player_id, gpu_hours, cpu_hours, current_round)
            is_within_batna = proposed_utility >= batna
            
            if is_within_batna:
                if rounds_remaining == 0:  # No proposals left - encourage acceptance
                    acceptance_guidance = (
                        f"🎯 FINAL ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is better than your BATNA ({batna:.1f}). You have no proposals left - ACCEPT to secure this beneficial deal!\n"
                    )
                elif rounds_remaining == 1:  # Last proposal - be more encouraging
                    acceptance_guidance = (
                        f"🎯 ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is better than your BATNA ({batna:.1f}). With only 1 proposal left, consider accepting or making the last counter offer.\n"
                    )
                else:  # Multiple proposals left - encourage exploration
                    acceptance_guidance = (
                        f"💡 ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is better than your BATNA ({batna:.1f}), but you have {rounds_remaining} proposals left. You might negotiate for an even better deal.\n"
                    )
            else:
                utility_gap = batna - proposed_utility
                if rounds_remaining == 0:  # No proposals left - suggest accepting to avoid no-deal
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is {utility_gap:.1f} below your BATNA ({batna:.1f}). You have no proposals left. ACCEPT to avoid no-deal or REJECT.\n"
                    )
                elif rounds_remaining == 1:  # Last proposal - be more encouraging
                    acceptance_guidance = (
                        f"🎯 ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is {utility_gap:.1f} below your BATNA ({batna:.1f}). With only 1 proposal left, consider accepting or making the last counter offer.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"⚠️ ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is {utility_gap:.1f} below your BATNA ({batna:.1f}). You should negotiate for a better allocation.\n"
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

        prompt = f"""=== RESOURCE ALLOCATION NEGOTIATION ===
{round_display} | Role: {role.upper()}

YOUR PRIORITIES: {role_priorities}
GOAL: Maximize your utility: {utility_func}
Your BATNA (Best Alternative): {batna:.1f}
{proposal_guidance}

CURRENT SITUATION:
{offer_status}
{acceptance_guidance}

CONSTRAINTS & RULES:
- x is the total GPU hours in system
- y is the total CPU hours in system
- Total resources: x + y ≤ {self.total_resources}
- GPU-Bandwidth limit: 4x + 4y ≤ {self.constraints['gpu_bandwidth']}
- Minimum allocations: x ≥ {self.constraints['min_gpu']}, y ≥ {self.constraints['min_cpu']}

RESPONSE FORMAT: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "accept"}}  // Accept the opponent's last offer
{{"type": "offer", "gpu_hours": 30, "cpu_hours": 25}}  // Propose new allocation (if proposals remain)
{{"type": "reject"}}  // Reject and end negotiation

EXAMPLE OFFERS:
{{"type": "offer", "gpu_hours": 30, "cpu_hours": 25}}

Do NOT repeat any of the rules or instructions in your response. Focus on negotiation.

Your response:"""

        return prompt

    # Abstract methods required by BaseGame interface
    def process_action(self, action: PlayerAction) -> Dict[str, Any]:
        """Process a single player action - required by BaseGame interface."""
        # For compatibility with BaseGame interface, delegate to process_actions
        actions_dict = {action.player_id: {"type": action.action_type, **action.action_data}}
        return self.process_actions(actions_dict, self.game_data)

    def check_end_conditions(self) -> bool:
        """Check if the game should end - required by BaseGame interface."""
        return self.is_game_over(self.game_data)

    def calculate_scores(self) -> Dict[str, float]:
        """Calculate final scores for all players - required by BaseGame interface."""
        if self.game_data.get("agreement_reached", False):
            return self.game_data.get("final_utilities", {})
        else:
            # Return BATNA values if no agreement
            return {
                self.development: self.development_batna,
                self.marketing: self.marketing_batna
            }