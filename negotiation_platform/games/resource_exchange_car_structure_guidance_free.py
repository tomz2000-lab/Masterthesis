import random
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
        self.development_batna = config.get("batnas", {}).get("development", 300)
        self.marketing_batna = config.get("batnas", {}).get("marketing", 360)
        self.max_rounds = config.get("rounds", 5)
        self.batna_decay = config.get("batna_decay", {"development": 0.03, "marketing": 0.02})

    def initialize_game(self, players: List[str]) -> Dict[str, Any]:
        """Initialize resource allocation negotiation - same structure as car game."""
        if len(players) != 2:
            raise ValueError("Resource allocation game requires exactly 2 players")

        self.players = players
        self.development = players[0]  # First player is development
        self.marketing = players[1]    # Second player is marketing
        self.state = self.state.__class__.ACTIVE  # Set to active state

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
        """Calculate utility for resource allocation."""
        if player == self.development:
            # Development: 12x + 3y + ε (stochastic demand)
            epsilon = random.gauss(0, 2)  # Reduced variance for stability
            base_utility = 12 * x + 3 * y
            return base_utility + epsilon
        else:
            # Marketing: 3x + 12y + i (market volatility)  
            i_factor = random.uniform(-0.02, 0.02)  # Reduced volatility
            base_utility = 3 * x + 12 * y
            return base_utility * (1 + i_factor)

    def is_valid_allocation(self, x: float, y: float) -> bool:
        """Check if allocation satisfies constraints."""
        # Force explicit constraint values in case config loading failed
        total_resources = getattr(self, 'total_resources', 100)
        gpu_bandwidth_limit = self.constraints.get("gpu_bandwidth", 300)  # Default to 300
        min_gpu = self.constraints.get("min_gpu", 5)  # Default to 5
        min_bandwidth = self.constraints.get("min_bandwidth", 5)  # Default to 5
        
        total_check = x + y <= total_resources
        gpu_bandwidth_check = 3 * x + 4 * y <= gpu_bandwidth_limit
        min_gpu_check = x >= min_gpu
        min_bandwidth_check = y >= min_bandwidth
        positive_check = x >= 0 and y >= 0
        
        is_valid = total_check and gpu_bandwidth_check and min_gpu_check and min_bandwidth_check and positive_check
        
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"🔍 DETAILED CONSTRAINT CHECK for ({x},{y}):")
        logger.warning(f"  Total: {x}+{y}={x+y} ≤ {total_resources} → {'✅' if total_check else '❌'}")
        logger.warning(f"  GPU-BW: 3×{x}+4×{y}={3*x + 4*y} ≤ {gpu_bandwidth_limit} → {'✅' if gpu_bandwidth_check else '❌'}")
        logger.warning(f"  Min GPU: {x} ≥ {min_gpu} → {'✅' if min_gpu_check else '❌'}")
        logger.warning(f"  Min BW: {y} ≥ {min_bandwidth} → {'✅' if min_bandwidth_check else '❌'}")
        logger.warning(f"  Positive: x≥0 and y≥0 → {'✅' if positive_check else '❌'}")
        logger.warning(f"  FINAL RESULT: {is_valid}")

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

        # Always log validation attempts for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"🔍 [VALIDATION] Player {player}: {action}")

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
            print(f"  - GPU-BW: 3×{gpu_hours}+4×{bandwidth}={3*gpu_hours + 4*bandwidth} ≤ {self.constraints.get('gpu_bandwidth', 'MISSING')}?")
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
        
        for player_id, action in actions.items():
            if not self.is_valid_action(player_id, action, game_state):
                continue
                
            action_type = action.get("type", "")
            
            if action_type == "offer":
                # Check proposal limit
                player_proposals = game_state.get(f"{player_id}_proposal_count", 0)
                if player_proposals >= max_proposals:
                    print(f"⚠️ Player {player_id} tried to make offer but exceeded proposal limit ({player_proposals}/{max_proposals})")
                    continue
                
                gpu_hours = action.get("gpu_hours", 0)
                bandwidth = action.get("bandwidth", 0)
                
                # DEFENSIVE: Additional sanitization in case action bypassed validation
                if isinstance(gpu_hours, list):
                    gpu_hours = gpu_hours[0] if len(gpu_hours) > 0 else 0
                if isinstance(bandwidth, list):
                    bandwidth = bandwidth[0] if len(bandwidth) > 0 else 0
                
                # Ensure numeric values
                gpu_hours = float(gpu_hours)
                bandwidth = float(bandwidth)
                
                # Increment proposal count
                game_state[f"{player_id}_proposal_count"] = player_proposals + 1
                
                # Store the offer
                game_state.setdefault("offers", []).append({
                    "player": player_id,
                    "round": current_round,
                    "gpu_hours": gpu_hours,
                    "bandwidth": bandwidth,
                    "utility": self.calculate_utility(player_id, gpu_hours, bandwidth, current_round)
                })
                
            elif action_type == "accept":
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"🎯 [ACCEPT PROCESSING] {player_id} accepting offer...")
                
                # Find the most recent offer from the other player
                other_player = self.marketing if player_id == self.development else self.development
                recent_offers = [o for o in game_state.get("offers", []) if o["player"] == other_player]
                
                if recent_offers:
                    accepted_offer = recent_offers[-1]
                    x, y = accepted_offer["gpu_hours"], accepted_offer["bandwidth"]
                    logger.warning(f"🎯 [AGREEMENT REACHED] Accepting offer ({x},{y})")
                    
                    # Calculate final utilities
                    dev_utility = self.calculate_utility(self.development, x, y, current_round)
                    marketing_utility = self.calculate_utility(self.marketing, x, y, current_round)
                    
                    # Calculate BATNAs at agreement time for metrics
                    dev_batna = self.get_current_batna(self.development, current_round)
                    marketing_batna = self.get_current_batna(self.marketing, current_round)
                    
                    game_state.update({
                        "agreement_reached": True,
                        "final_allocation": {"gpu_hours": x, "bandwidth": y},
                        "agreement_round": current_round,
                        "final_utilities": {
                            self.development: dev_utility,
                            self.marketing: marketing_utility
                        },
                        "batnas_at_agreement": {
                            self.development: dev_batna,
                            self.marketing: marketing_batna
                        }
                    })
                    logger.warning(f"🎉 [GAME COMPLETE] Agreement: ({x},{y}), Round: {current_round}")
                    return game_state
                else:
                    logger.warning(f"⚠️ [ACCEPT ERROR] No recent offers found from {other_player}")
                    
            elif action_type == "reject":
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"❌ [REJECT] Player {player_id} rejects current offer")
                # Continue to next round
                    
            elif action_type == "reject":
                game_state.setdefault("rejections", []).append({
                    "player": player_id,
                    "round": current_round
                })

        return game_state

    def is_game_over(self, game_state: Dict[str, Any]) -> bool:
        """Check if game should end - same pattern as car game."""
        if game_state.get("agreement_reached", False):
            return True
            
        if game_state["current_round"] >= self.max_rounds:
            return True
            
        return False

    def get_system_prompt(self, player_id: str, game_state: Dict[str, Any]) -> str:
        """Generate role-specific system prompt - adapted from car game pattern."""
        current_round = game_state["current_round"]
        current_batna = self.get_current_batna(player_id, current_round)
        
        if player_id == self.development:
            return f"""You are the **Development Team** negotiating resource allocation with the Marketing Team.

**YOUR ROLE & OBJECTIVES:**
- You need GPU hours (x) and bandwidth (y) for product development
- Your utility function: 12x + 3y + ε (you heavily prefer GPU hours)
- Your current BATNA: {current_batna:.1f} (decreases each round due to time pressure)

**CONSTRAINTS:**
- Total resources: x + y ≤ {self.total_resources}
- GPU-Bandwidth limit: 3x + 4y ≤ {self.constraints['gpu_bandwidth']}
- Minimum GPU hours: x ≥ {self.constraints['min_gpu']}
- Minimum bandwidth: y ≥ {self.constraints['min_bandwidth']}

**CONSTRAINT EXAMPLES:**
- Valid: (25,40) → 3×25 + 4×40 = 235 ≤ 300 ✅
- Valid: (30,45) → 3×30 + 4×45 = 270 ≤ 300 ✅  
- Invalid: (40,50) → 3×40 + 4×50 = 320 > 300 ❌

**NEGOTIATION STRATEGY:**
- Focus on maximizing GPU hours (12x coefficient vs 3y)
- Accept offers that give you utility > {current_batna:.1f}
- Time pressure increases - your BATNA decreases each round
- Consider that Marketing Team prefers bandwidth (their utility: 3x + 12y + i)

**ROUND {current_round}/{self.max_rounds}** - Make strategic offers and respond to counteroffers.

**RESPONSE FORMAT:** Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "offer", "gpu_hours": 25, "bandwidth": 40}}
{{"type": "accept"}}
{{"type": "reject"}}

Your response:"""

        else:  # Marketing Team
            return f"""You are the **Marketing Team** negotiating resource allocation with the Development Team.

**YOUR ROLE & OBJECTIVES:**
- You need GPU hours (x) and bandwidth (y) for marketing campaigns
- Your utility function: 3x + 12y + i (you heavily prefer bandwidth)
- Your current BATNA: {current_batna:.1f} (decreases each round due to time pressure)

**CONSTRAINTS:**
- Total resources: x + y ≤ {self.total_resources}
- GPU-Bandwidth limit: 3x + 4y ≤ {self.constraints['gpu_bandwidth']}
- Minimum GPU hours: x ≥ {self.constraints['min_gpu']}
- Minimum bandwidth: y ≥ {self.constraints['min_bandwidth']}

**CONSTRAINT EXAMPLES:**
- Valid: (20,60) → 3×20 + 4×60 = 300 ≤ 300 ✅  
- Valid: (25,50) → 3×25 + 4×50 = 275 ≤ 300 ✅
- Invalid: (30,60) → 3×30 + 4×60 = 360 > 300 ❌

**NEGOTIATION STRATEGY:**
- Focus on maximizing bandwidth (12y coefficient vs 3x)
- Accept offers that give you utility > {current_batna:.1f}
- Time pressure increases - your BATNA decreases each round
- Consider that Development Team prefers GPU hours (their utility: 12x + 3y + ε)

**ROUND {current_round}/{self.max_rounds}** - Make strategic offers and respond to counteroffers.

**RESPONSE FORMAT:** Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "offer", "gpu_hours": 20, "bandwidth": 45}}
{{"type": "accept"}}
{{"type": "reject"}}

Your response:"""

    def get_human_prompt(self, player_id: str, game_state: Dict[str, Any]) -> str:
        """Generate context-aware prompt for human player - adapted from car game."""
        current_round = game_state["current_round"]
        current_batna = self.get_current_batna(player_id, current_round)
        
        prompt = f"\n**Round {current_round}/{self.max_rounds} - Resource Allocation Negotiation**\n"
        prompt += f"Your current BATNA: {current_batna:.1f}\n\n"
        
        # Show recent offers if any
        recent_offers = game_state.get("offers", [])
        proposed_utility = None  # Initialize to avoid undefined variable errors
        
        if recent_offers:
            last_offer = recent_offers[-1]
            x, y = last_offer["gpu_hours"], last_offer["bandwidth"]
            
            proposed_utility = self.calculate_utility(player_id, x, y, current_round)
            
            prompt += f"**Latest Proposal:**\n"
            prompt += f"- GPU Hours: {x}, Bandwidth: {y}\n"
            prompt += f"- Your utility from this: {proposed_utility:.1f}\n"
            
            # Add strategic guidance for decision making
            guidance_hint = ""
            if proposed_utility > current_batna:
                utility_advantage = proposed_utility - current_batna
                if current_round > 2:  # After initial rounds
                    guidance_hint = f" Consider accepting this beneficial offer (+{utility_advantage:.1f})."
                prompt += f"- Analysis: This offer gives {proposed_utility:.1f} utility vs BATNA of {current_batna:.1f} (+{utility_advantage:.1f}){guidance_hint}\n\n"
            else:
                utility_disadvantage = current_batna - proposed_utility
                guidance_hint = f" Try proposing an allocation that better meets your needs while being fair to the other team."
                prompt += f"- Analysis: This offer gives {proposed_utility:.1f} utility vs BATNA of {current_batna:.1f} (-{utility_disadvantage:.1f}){guidance_hint}\n\n"
        else:
            prompt += "No offers made yet.\n"

        prompt += f"Your BATNA: {current_batna:.1f}\n"
        if recent_offers and proposed_utility is not None:
            prompt += f"Opponent's last offer utility: {proposed_utility:.1f} ({'above' if proposed_utility > current_batna else 'below'} BATNA)\n"

        prompt += "\nTASK: Respond with ONLY valid JSON. No explanations.\n"
        prompt += "Valid responses:\n"
        prompt += "{\"type\": \"accept\"}\n"
        prompt += "{\"type\": \"offer\", \"gpu_hours\": X, \"bandwidth\": Y}\n"
        prompt += "{\"type\": \"reject\"}\n\n"
        prompt += "Your response:"
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
        """Enhanced prompt with structured format, proposal limits, and strategic guidance - aligned with NegotiationArena."""
        if not hasattr(self, 'development') or not hasattr(self, 'marketing'):
            return "Game not initialized properly"

        private_info = self.game_data.get("private_info", {}).get(player_id, {})
        current_round = self.game_data.get("current_round", 1)
        other_player = self.marketing if player_id == self.development else self.development
        offers = self.game_data.get("offers", [])
        my_last = next((o for o in reversed(offers) if o["player"] == player_id), None)
        other_last = next((o for o in reversed(offers) if o["player"] == other_player), None)
        batna = self.get_current_batna(player_id, current_round)
        
        # Track proposals made by this player
        player_proposals = self.game_data.get(f"{player_id}_proposal_count", 0)
        max_proposals = 3  # Limit proposals to force decision making
        
        role = "development" if player_id == self.development else "marketing"
        goal = f"Achieve utility greater than {batna:.1f}" if role == "development" else f"Achieve utility greater than {batna:.1f}"

        offer_history = []
        if my_last:
            offer_history.append(f"- Your last offer: gpu_hours={my_last['gpu_hours']}, bandwidth={my_last['bandwidth']}")
        if other_last:
            offer_history.append(f"- Opponent's last offer: gpu_hours={other_last['gpu_hours']}, bandwidth={other_last['bandwidth']}")
        offer_status = "\n".join(offer_history) if offer_history else "No offers made yet."

        # Enhanced acceptance guidance
        acceptance_guidance = ""
        can_propose = player_proposals < max_proposals
        
        if other_last is not None:
            utility = self.calculate_utility(player_id, other_last["gpu_hours"], other_last["bandwidth"], current_round)
            percent_diff = calculate_percentage_difference(utility, batna)
            is_within_batna = utility > batna  # Both teams should accept offers giving utility > BATNA
            
            if is_within_batna:
                acceptance_guidance = (
                    f"🎯 STRATEGIC ANALYSIS: The opponent's offer (utility={utility:.1f}) is ABOVE your BATNA ({batna:.1f}). "
                    f"This is a GOOD DEAL for you! Consider accepting to secure a beneficial agreement.\n"
                )
            else:
                if can_propose:
                    acceptance_guidance = (
                        f"⚠️ STRATEGIC ANALYSIS: The opponent's offer (utility={utility:.1f}) is BELOW your BATNA ({batna:.1f}). "
                        f"You should negotiate for better resource allocation.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: You've used all {max_proposals} proposals. The opponent's offer is below your BATNA. "
                        f"You must now ACCEPT (if acceptable) or REJECT and end the negotiation.\n"
                    )

        # Proposal limit guidance
        proposal_guidance = ""
        if can_propose:
            proposal_guidance = f"You have {max_proposals - player_proposals} proposals remaining."
        else:
            proposal_guidance = f"⚠️ You have used all {max_proposals} proposals. You can only ACCEPT or REJECT now."

        # Build the offer option text based on proposal limits
        offer_option = '{"type": "offer", "gpu_hours": X, "bandwidth": Y}  // Make a new resource allocation offer with SINGLE INTEGER VALUES'
        if can_propose:
            offer_option += " // Only if you have proposals left"
        else:
            offer_option += " // NOT ALLOWED - no proposals left"

        prompt = f"""=== RESOURCE ALLOCATION NEGOTIATION ===
Round {current_round}/{self.max_rounds} | Role: {role.upper()}

GOAL: {goal}
Your BATNA (Best Alternative): {batna:.1f}
{proposal_guidance}

CURRENT SITUATION:
{offer_status}
{acceptance_guidance}
You must respond with a structured format that includes your reasoning and decision.

IMPORTANT: Use SINGLE INTEGER VALUES only, not lists or ranges!
Example: "gpu_hours": 25, "bandwidth": 40 (NOT [25,30] or [40,50])

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
The opponent's offer gives me utility {batna + 10:.1f} which is above my BATNA of {batna:.1f}. This allocation meets most of my resource needs while allowing for a fair agreement.
</REASONING>

<DECISION>
{{"type": "offer", "gpu_hours": 25, "bandwidth": 40}}
</DECISION>

<MESSAGE>
I propose gpu_hours=25 and bandwidth=40. This allocation balances both our needs fairly.
</MESSAGE>
```

Your response:"""
        return prompt