import random
import re
import json
from typing import Dict, List, Any, Optional, Tuple

from .base_game import BaseGame, PlayerAction
from .negotiation_tools import calculate_percentage_difference, is_offer_above_batna, is_offer_below_batna, suggest_next_offer


class ResourceAllocationGame(BaseGame):
    """
    Resource allocation negotiation game between Development and Marketing teams.
    - Development Team needs GPU hours (x) and bandwidth (y) - utility: 12x + 3y + ε  
    - Marketing Team needs GPU hours (x) and bandwidth (y) - utility: 3x + 12y + i
    - Constraints: x + y ≤ 100, 3x + 4y ≤ 300, x ≥ 5, y ≥ 5
    - BATNAs: Development 300 (external provider), Marketing 360 (alternative SaaS)
    - 5 rounds with time-adjusted BATNA decay
    """

    def __init__(self, config: Dict[str, Any]):
        # Initialize base class - exact same pattern as car game
        super().__init__(game_id="resource_allocation", config=config)
        self.total_resources = config.get("total_resources", 100)
        self.constraints = config.get("constraints", {
            "gpu_bandwidth": 300,  # 3x + 4y <= 300 (relaxed to allow more solutions)
            "min_gpu": 5,          # x >= 5 (reduced to be more flexible)
            "min_bandwidth": 5     # y >= 5 (reduced to be more flexible)
        })
        
        print(f"🔧 CONSTRAINT INIT: total_resources={self.total_resources}, constraints={self.constraints}")
        
        # BATNA configuration - same pattern as car game
        batnas = config.get("batnas", {})
        self.development_batna = batnas.get("development", 300)
        self.marketing_batna = batnas.get("marketing", 360)
        print(f"🎯 [BATNA CONFIG] Development={self.development_batna}, Marketing={self.marketing_batna}")
        self.max_rounds = config.get("rounds", 5)
        self.batna_decay = config.get("batna_decay", {"development": 0.025, "marketing": 0.025})

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
        """Initialize resource allocation negotiation - same structure as car game."""
        if len(players) != 2:
            raise ValueError("Resource allocation game requires exactly 2 players")

        # Randomize first mover like in integrative negotiation and car game
        import random
        shuffled_players = players.copy()
        random.shuffle(shuffled_players)
        
        self.players = shuffled_players
        self.development = shuffled_players[0]  # First player is development
        self.marketing = shuffled_players[1]    # Second player is marketing
        self.state = self.state.__class__.ACTIVE  # Set to active state
        
        print(f"🎲 [FIRST MOVER] Randomized player order: Development={self.development}, Marketing={self.marketing}")

        self.game_data = {
            "game_type": "resource_allocation",
            "players": self.players,
            "rounds": self.max_rounds,
            "current_round": 1,
            "private_info": {
                self.development: {
                    "role": "development",
                    "team": "development",
                    "utility_function": "12x + 3y + ε",
                    "batna": self.development_batna,
                    "constraints": self.constraints
                },
                self.marketing: {
                    "role": "marketing", 
                    "team": "marketing",
                    "utility_function": "3x + 12y + i", 
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
        """Calculate time-adjusted BATNA for current round - same as car game."""
        if player == self.development:
            decay_rate = self.batna_decay["development"]
            base_batna = self.development_batna
        else:
            decay_rate = self.batna_decay["marketing"]
            base_batna = self.marketing_batna

        return base_batna * ((1 - decay_rate) ** (round_num - 1))

    def calculate_utility(self, player: str, x: float, y: float, round_num: int) -> float:
        """Calculate utility for resource allocation with balanced stochastic components."""
        if player == self.development:
            # Development: 12x + 3y + ε (stochastic demand)
            epsilon = random.gauss(0, 1)  # Reduced and balanced variance
            base_utility = 12 * x + 3 * y
            return base_utility + epsilon
        else:
            # Marketing: 3x + 12y + ε (same additive noise for fairness)  
            epsilon = random.gauss(0, 1)  # Same noise pattern as development
            base_utility = 3 * x + 12 * y
            return base_utility + epsilon

    def is_valid_allocation(self, x: float, y: float) -> bool:
        """Check if allocation satisfies constraints."""
        # Force explicit constraint values in case config loading failed
        total_resources = getattr(self, 'total_resources', 100)
        gpu_bandwidth_limit = self.constraints.get("gpu_bandwidth", 300)  # Default to 300
        min_gpu = self.constraints.get("min_gpu", 5)  # Default to 5
        min_bandwidth = self.constraints.get("min_bandwidth", 5)  # Default to 5
        
        total_check = x + y <= total_resources
        gpu_bandwidth_check = 4 * x + 4 * y <= gpu_bandwidth_limit  # BALANCED: 4x + 4y <= 300
        min_gpu_check = x >= min_gpu
        min_bandwidth_check = y >= min_bandwidth
        positive_check = x >= 0 and y >= 0
        
        is_valid = total_check and gpu_bandwidth_check and min_gpu_check and min_bandwidth_check and positive_check
        
        # Simplified constraint logging for cleaner output
        import logging
        logger = logging.getLogger(__name__)
        if not is_valid:
            logger.warning(f"🔍 CONSTRAINT VIOLATION for ({x},{y}):")
            if not total_check:
                logger.warning(f"  ❌ Total: {x}+{y}={x+y} > {total_resources}")
            if not gpu_bandwidth_check:
                logger.warning(f"  ❌ GPU-BW: 4×{x}+4×{y}={4*x + 4*y} > {gpu_bandwidth_limit}")
            if not min_gpu_check:
                logger.warning(f"  ❌ Min GPU: {x} < {min_gpu}")
            if not min_bandwidth_check:
                logger.warning(f"  ❌ Min BW: {y} < {min_bandwidth}")

        logger.warning(f"🔍 CONSTRAINT CHECK for ({x},{y}): {is_valid}")
        print(f"  Positive: {x}≥0 and {y}≥0 → {'✅' if positive_check else '❌'}")
        print(f"  FINAL RESULT: {'VALID ✅' if is_valid else 'INVALID ❌'}")
        
        return is_valid

    def is_valid_action(self, player: str, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Validate player action - same pattern as car game."""
        action_type = action.get("type", "")

        # Handle different type formats from models
        if action_type == 1 or action_type == "1":
            action["type"] = "offer"  # Modify the action in place
            action_type = "offer"
        elif action_type in ["propose", "suggestion"]:
            action["type"] = "offer"  # Modify the action in place
            action_type = "offer"

        # Reduced validation logging for cleaner output
        import logging
        logger = logging.getLogger(__name__)
        if action.get("type") == "noop":
            logger.warning(f"⚠️ [PARSE ERROR] Player {player}: failed to parse response")

        if action_type == "offer":
            # Check proposal limit
            player_proposals = game_state.get(f"{player}_proposal_count", 0)
            max_proposals = 3
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} tried to make offer but exceeded proposal limit ({player_proposals}/{max_proposals})")
                return False
                
            gpu_hours = action.get("gpu_hours", 0)
            bandwidth = action.get("bandwidth", 0)
            
            # SANITIZE INPUT: Handle cases where LLM returns lists instead of single values
            if isinstance(gpu_hours, list):
                print(f"🔧 [SANITIZE] GPU hours is list {gpu_hours}, taking first value")
                gpu_hours = gpu_hours[0] if len(gpu_hours) > 0 else 0
                action["gpu_hours"] = gpu_hours  # Update the action
            
            if isinstance(bandwidth, list):
                print(f"🔧 [SANITIZE] Bandwidth is list {bandwidth}, taking first value")
                bandwidth = bandwidth[0] if len(bandwidth) > 0 else 0
                action["bandwidth"] = bandwidth  # Update the action
            
            # Convert to float to handle any string inputs
            try:
                gpu_hours = float(gpu_hours)
                bandwidth = float(bandwidth)
                action["gpu_hours"] = gpu_hours
                action["bandwidth"] = bandwidth
            except (ValueError, TypeError) as e:
                print(f"⚠️ [INVALID] Cannot convert to numbers: gpu_hours={gpu_hours}, bandwidth={bandwidth}, error={e}")
                return False
            
            # Handle empty offers - treat as invalid
            if gpu_hours <= 0 or bandwidth <= 0:
                print(f"⚠️ [INVALID] Empty offer: gpu_hours={gpu_hours}, bandwidth={bandwidth}")
                return False

            # Validate allocation constraints with detailed logging
            print(f"🧮 [CALC] Checking (gpu_hours={gpu_hours}, bandwidth={bandwidth}):")
            print(f"  - Total: {gpu_hours}+{bandwidth}={gpu_hours+bandwidth} ≤ {self.total_resources}?")
            print(f"  - GPU-BW: 4×{gpu_hours}+4×{bandwidth}={4*gpu_hours + 4*bandwidth} ≤ {self.constraints.get('gpu_bandwidth', 'MISSING')}?")
            print(f"  - Min GPU: {gpu_hours} ≥ {self.constraints.get('min_gpu', 'MISSING')}?")
            print(f"  - Min BW: {bandwidth} ≥ {self.constraints.get('min_bandwidth', 'MISSING')}?")
            
            # First check physical constraints
            is_physically_valid = self.is_valid_allocation(gpu_hours, bandwidth)
            
            if not is_physically_valid:
                print(f"🎯 [RESULT] Offer ({gpu_hours},{bandwidth}) → INVALID (violates physical constraints)")
                return False
            
            # Then check BATNA constraint - players cannot make offers worse than their own BATNA
            current_round = game_state.get("current_round", 1)
            player_utility = self.calculate_utility(player, gpu_hours, bandwidth, current_round)
            player_batna = self.get_current_batna(player, current_round)
            
            print(f"  - BATNA Check: utility={player_utility:.1f} ≥ BATNA={player_batna:.1f}?")
            
            if player_utility < player_batna:
                print(f"🎯 [RESULT] Offer ({gpu_hours},{bandwidth}) → INVALID (utility {player_utility:.1f} < BATNA {player_batna:.1f})")
                print(f"⚠️ [RATIONAL] Players cannot make offers worse than their BATNA - this prevents irrational behavior")
                return False
            
            print(f"🎯 [RESULT] Offer ({gpu_hours},{bandwidth}) → VALID (all constraints satisfied)")
            return True

        elif action_type in ["accept", "ACCEPT"]:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"✅ [ACCEPT] Player {player} accepts current offer")
            return True

        elif action_type in ["reject"]:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"❌ [REJECT] Player {player} rejects current offer")
            return True
            
        elif action_type in ["reject", "REJECT"]:
            print(f"❌ [REJECT] Player {player} rejects")
            return True

        print(f"⚠️ [UNKNOWN] Unknown action type '{action_type}' for {player}")
        return False

    def process_actions(self, actions: Dict[str, Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process player actions with proposal limits and enhanced validation."""
        current_round = game_state["current_round"]
        max_proposals = 3

        # Initialize proposal counters if not present
        for player in [self.development, self.marketing]:
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
                normalized_actions[player] = {"type": "offer", "gpu_hours": action.get("gpu_hours"), "bandwidth": action.get("bandwidth")}
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
            gpu_hours = action.get("gpu_hours")
            bandwidth = action.get("bandwidth")
            game_state[f"{player}_last_offer"] = {"gpu_hours": gpu_hours, "bandwidth": bandwidth}
            game_state[f"{player}_proposal_count"] = player_proposals + 1
            print(f"💡 Player {player} made offer gpu_hours={gpu_hours}, bandwidth={bandwidth} (proposal {player_proposals + 1}/{max_proposals})")

        # Process acceptances and rejections
        for player, action in responses.items():
            if action.get("type") == "accept":
                # Find the offer being accepted
                other_player = self.marketing if player == self.development else self.development
                if f"{other_player}_last_offer" in game_state:
                    accepted_offer = game_state[f"{other_player}_last_offer"]
                    gpu_hours = accepted_offer["gpu_hours"]
                    bandwidth = accepted_offer["bandwidth"]
                    print(f"✅ Player {player} accepted offer of gpu_hours={gpu_hours}, bandwidth={bandwidth}")

                    # Validate agreement against BATNAs (more lenient for acceptance)
                    dev_batna = self.get_current_batna(self.development, current_round)
                    marketing_batna = self.get_current_batna(self.marketing, current_round)

                    # Accept the agreement even if slightly outside BATNA to encourage deals
                    return self._create_agreement(gpu_hours, bandwidth, current_round, game_state)
                else:
                    print(f"⚠️ Player {player} tried to accept but no offer exists")
            
            elif action.get("type") == "reject":
                print(f"❌ Player {player} rejected the negotiation")
                return self._create_no_agreement(game_state)

        # Check if both players have exhausted proposals
        dev_proposals = game_state.get(f"{self.development}_proposal_count", 0)
        marketing_proposals = game_state.get(f"{self.marketing}_proposal_count", 0)
        
        if dev_proposals >= max_proposals and marketing_proposals >= max_proposals:
            print(f"⏰ Both players exhausted proposals ({dev_proposals}/{max_proposals}, {marketing_proposals}/{max_proposals}). Ending negotiation.")
            return self._create_no_agreement(game_state)

        # Update round
        game_state["current_round"] += 1

        # Check if deadline reached
        if game_state["current_round"] > self.max_rounds:
            return self._create_no_agreement(game_state)

        return game_state

    def _create_agreement(self, gpu_hours: float, bandwidth: float, round_num: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create agreement result."""
        dev_batna = self.get_current_batna(self.development, round_num)
        marketing_batna = self.get_current_batna(self.marketing, round_num)

        dev_utility = self.calculate_utility(self.development, gpu_hours, bandwidth, round_num)
        marketing_utility = self.calculate_utility(self.marketing, gpu_hours, bandwidth, round_num)

        game_state.update({
            "agreement_reached": True,
            "final_allocation": {"gpu_hours": gpu_hours, "bandwidth": bandwidth},
            "agreement_round": round_num,
            "final_utilities": {
                self.development: dev_utility,
                self.marketing: marketing_utility
            },
            "batnas_at_agreement": {
                self.development: dev_batna,
                self.marketing: marketing_batna
            }
        })
        return game_state

    def _create_no_agreement(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create no agreement result."""
        game_state.update({
            "agreement_reached": False,
            "final_utilities": {
                self.development: 0.0,
                self.marketing: 0.0
            }
        })
        return game_state

    def is_game_over(self, game_state: Dict[str, Any]) -> bool:
        """Check if game should end - same pattern as car game."""
        if game_state.get("agreement_reached", False):
            return True
            
        if game_state["current_round"] >= self.max_rounds:
            return True
            
        return False

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
        max_proposals = 3  # Limit proposals to force decision making
        
        role = "development" if player_id == self.development else "marketing"
        team_name = "Development Team" if role == "development" else "Marketing Team"
        utility_func = "12x + 3y + ε" if role == "development" else "3x + 12y + i"
        preference = "GPU hours" if role == "development" else "bandwidth"
        goal = f"Maximize GPU hours allocation" if role == "development" else f"Maximize bandwidth allocation"
        
        offer_history = []
        if my_offer:
            offer_history.append(f"- Your last offer: GPU={my_offer['gpu_hours']}, Bandwidth={my_offer['bandwidth']}")
        if other_offer:
            offer_history.append(f"- Opponent's last offer: GPU={other_offer['gpu_hours']}, Bandwidth={other_offer['bandwidth']}")
        offer_status = "\n".join(offer_history) if offer_history else "No offers made yet."

        # Enhanced acceptance guidance
        acceptance_guidance = ""
        can_propose = player_proposals < max_proposals
        
        if other_offer is not None:
            gpu_hours = other_offer['gpu_hours']
            bandwidth = other_offer['bandwidth']
            proposed_utility = self.calculate_utility(player_id, gpu_hours, bandwidth, current_round)
            is_within_batna = proposed_utility >= batna
            
            if is_within_batna:
                acceptance_guidance = (
                    f"🎯 STRATEGIC ANALYSIS: The opponent's offer (GPU={gpu_hours}, Bandwidth={bandwidth}) gives you utility {proposed_utility:.1f} which is ABOVE your BATNA ({batna:.1f}). "
                    f"This is a GOOD DEAL for you! Consider accepting to secure a beneficial agreement.\n"
                )
            else:
                if can_propose:
                    acceptance_guidance = (
                        f"⚠️ STRATEGIC ANALYSIS: The opponent's offer (GPU={gpu_hours}, Bandwidth={bandwidth}) gives you utility {proposed_utility:.1f} which is BELOW your BATNA ({batna:.1f}). "
                        f"You should negotiate for a better allocation.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: You've used all {max_proposals} proposals. The opponent's offer gives utility {proposed_utility:.1f} vs BATNA {batna:.1f}. "
                        f"You must now ACCEPT (if acceptable) or REJECT and end the negotiation.\n"
                    )

        # Proposal limit guidance
        proposal_guidance = ""
        if can_propose:
            proposal_guidance = f"You have {max_proposals - player_proposals} proposals remaining."
        else:
            proposal_guidance = f"⚠️ You have used all {max_proposals} proposals. You can only ACCEPT or REJECT now."

        # Build the offer option text based on proposal limits
        offer_option = f'{{"type": "offer", "gpu_hours": [X], "bandwidth": [Y]}}  // Make a new allocation offer'
        if can_propose:
            offer_option += " // Only if you have proposals left"
        else:
            offer_option += " // NOT ALLOWED - no proposals left"

        prompt = f"""=== RESOURCE ALLOCATION NEGOTIATION ===

You are the **{team_name}** in a resource allocation negotiation.

**ROUND {current_round}/{self.max_rounds} | TIME PRESSURE INCREASING**

**YOUR OBJECTIVES:**
- Maximize your utility: {utility_func}
- You strongly prefer {preference}
- Your current BATNA: {batna:.1f} (decreases each round)

**CONSTRAINTS & RULES:**
- Total resources: x + y ≤ {self.total_resources}
- GPU-Bandwidth limit: 4x + 4y ≤ {self.constraints['gpu_bandwidth']}
- Minimum allocations: x ≥ {self.constraints['min_gpu']}, y ≥ {self.constraints['min_bandwidth']}

{offer_status}

📊 You have **{max_proposals - player_proposals}** proposals remaining out of {max_proposals} total.

{acceptance_guidance}{proposal_guidance}

**RESPONSE FORMAT:**
```
<REASONING>
[Analyze the current situation, your BATNA, the opponent's likely preferences, and explain your strategic thinking]
</REASONING>

<DECISION>
{{"type": "accept"}}  // Accept the opponent's last offer
{offer_option}
{{"type": "reject"}}  // Reject and end negotiation
</DECISION>

<MESSAGE>
[Optional: Communicate with the opponent to build rapport or explain your reasoning]
</MESSAGE>
```

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
        max_proposals = 3  # Limit proposals to force decision making
        
        role = "development" if player_id == self.development else "marketing"
        team_name = "Development Team" if role == "development" else "Marketing Team"
        utility_func = "12x + 3y + ε" if role == "development" else "3x + 12y + i"
        preference = "GPU hours" if role == "development" else "bandwidth"
        goal = f"Maximize GPU hours allocation" if role == "development" else f"Maximize bandwidth allocation"
        
        offer_history = []
        if my_offer:
            offer_history.append(f"- Your last offer: GPU={my_offer['gpu_hours']}, Bandwidth={my_offer['bandwidth']}")
        if other_offer:
            offer_history.append(f"- Opponent's last offer: GPU={other_offer['gpu_hours']}, Bandwidth={other_offer['bandwidth']}")
        offer_status = "\n".join(offer_history) if offer_history else "No offers made yet."

        # Enhanced acceptance guidance
        acceptance_guidance = ""
        can_propose = player_proposals < max_proposals
        
        if other_offer is not None:
            gpu_hours = other_offer['gpu_hours']
            bandwidth = other_offer['bandwidth']
            proposed_utility = self.calculate_utility(player_id, gpu_hours, bandwidth, current_round)
            is_within_batna = proposed_utility >= batna
            
            if is_within_batna:
                acceptance_guidance = (
                    f"🎯 STRATEGIC ANALYSIS: The opponent's offer (GPU={gpu_hours}, Bandwidth={bandwidth}) gives you utility {proposed_utility:.1f} which is ABOVE your BATNA ({batna:.1f}). "
                    f"This is a GOOD DEAL for you! Consider accepting to secure a beneficial agreement.\n"
                )
            else:
                if can_propose:
                    acceptance_guidance = (
                        f"⚠️ STRATEGIC ANALYSIS: The opponent's offer (GPU={gpu_hours}, Bandwidth={bandwidth}) gives you utility {proposed_utility:.1f} which is BELOW your BATNA ({batna:.1f}). "
                        f"You should negotiate for a better allocation.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: You've used all {max_proposals} proposals. The opponent's offer gives utility {proposed_utility:.1f} vs BATNA {batna:.1f}. "
                        f"You must now ACCEPT (if acceptable) or REJECT and end the negotiation.\n"
                    )

        # Proposal limit guidance
        proposal_guidance = ""
        if can_propose:
            proposal_guidance = f"You have {max_proposals - player_proposals} proposals remaining."
        else:
            proposal_guidance = f"⚠️ You have used all {max_proposals} proposals. You can only ACCEPT or REJECT now."

        # Build the offer option text based on proposal limits
        offer_option = f'{{"type": "offer", "gpu_hours": [X], "bandwidth": [Y]}}  // Make a new allocation offer'
        if can_propose:
            offer_option += " // Only if you have proposals left"
        else:
            offer_option += " // NOT ALLOWED - no proposals left"

        prompt = f"""=== RESOURCE ALLOCATION NEGOTIATION ===

You are the **{team_name}** in a resource allocation negotiation.

**ROUND {current_round}/{self.max_rounds} | TIME PRESSURE INCREASING**

**YOUR OBJECTIVES:**
- Maximize your utility: {utility_func}
- You strongly prefer {preference}
- Your current BATNA: {batna:.1f} (decreases each round)

**CONSTRAINTS & RULES:**
- Total resources: x + y ≤ {self.total_resources}
- GPU-Bandwidth limit: 4x + 4y ≤ {self.constraints['gpu_bandwidth']}
- Minimum allocations: x ≥ {self.constraints['min_gpu']}, y ≥ {self.constraints['min_bandwidth']}

{offer_status}

📊 You have **{max_proposals - player_proposals}** proposals remaining out of {max_proposals} total.

{acceptance_guidance}{proposal_guidance}

**RESPONSE FORMAT:**
```
<REASONING>
[Analyze the current situation, your BATNA, the opponent's likely preferences, and explain your strategic thinking]
</REASONING>

<DECISION>
{{"type": "accept"}}  // Accept the opponent's last offer
{offer_option}
{{"type": "reject"}}  // Reject and end negotiation
</DECISION>

<MESSAGE>
[Optional: Communicate with the opponent to build rapport or explain your reasoning]
</MESSAGE>
```

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