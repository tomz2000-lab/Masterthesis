import random
from typing import Dict, List, Any, Optional, Tuple
from .base_game import BaseGame, PlayerAction


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
            gpu_hours = action.get("gpu_hours", 0)
            bandwidth = action.get("bandwidth", 0)
            
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
            
            is_valid = self.is_valid_allocation(gpu_hours, bandwidth)
            print(f"🎯 [RESULT] Offer ({gpu_hours},{bandwidth}) → {'VALID' if is_valid else 'INVALID'}")
            return is_valid

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
        """Process player actions and update game state - same pattern as car game."""
        current_round = game_state["current_round"]
        
        for player_id, action in actions.items():
            if not self.is_valid_action(player_id, action, game_state):
                continue
                
            action_type = action.get("type", "")
            
            if action_type == "offer":
                gpu_hours = action.get("gpu_hours", 0)
                bandwidth = action.get("bandwidth", 0)
                
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
                    
                    game_state.update({
                        "agreement_reached": True,
                        "final_allocation": {"gpu_hours": x, "bandwidth": y},
                        "agreement_round": current_round,
                        "final_utilities": {
                            self.development: dev_utility,
                            self.marketing: marketing_utility
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
            
            # Make acceptance decision crystal clear
            if proposed_utility > current_batna:
                utility_advantage = proposed_utility - current_batna
                prompt += f"- 🎯 RECOMMENDATION: ACCEPT! ({proposed_utility:.1f} > {current_batna:.1f})\n\n"
                prompt += f"**ANALYSIS: This offer gives you {proposed_utility:.1f} utility, which is BETTER than your BATNA of {current_batna:.1f}. YOU SHOULD ACCEPT THIS OFFER.**\n\n"
                
                # Add extra emphasis for very good offers
                if utility_advantage > 50:
                    prompt += f"**🚨 CRITICAL: This is an EXCELLENT offer with +{utility_advantage:.1f} utility above your BATNA! ACCEPT immediately!**\n\n"
            else:
                prompt += f"- ⚠️ RECOMMENDATION: REJECT or COUNTER ({proposed_utility:.1f} < {current_batna:.1f})\n\n"
        else:
            # No offers yet - prompt to make an initial offer
            prompt += f"**No offers have been made yet. You should make the opening offer.**\n\n"
            
            # Add constraint information that models need to know
            prompt += f"**CONSTRAINTS - ALL OFFERS MUST SATISFY:**\n"
            prompt += f"1. **Total Resources:** gpu_hours + bandwidth ≤ {self.total_resources}\n"
            prompt += f"2. **GPU-Bandwidth Limit:** 3×gpu_hours + 4×bandwidth ≤ {self.constraints.get('gpu_bandwidth', 300)}\n"
            prompt += f"3. **Minimum GPU:** gpu_hours ≥ {self.constraints.get('min_gpu', 5)}\n"
            prompt += f"4. **Minimum Bandwidth:** bandwidth ≥ {self.constraints.get('min_bandwidth', 5)}\n\n"
            
            # Provide role-based guidance for initial offers
            if player_id == self.development:
                prompt += f"**ROLE:** You are the Development Team. You value GPU hours more highly.\n"
                prompt += f"**SUGGESTED OPENING:** Consider offering around (30, 45) - high GPU hours for your needs.\n"
                prompt += f"**CONSTRAINT CHECK:** (30,45) → Total: 75≤100 ✅, GPU-BW: 270≤300 ✅, Valid!\n"
                prompt += f"**EXACT JSON:** {{\"type\": \"offer\", \"gpu_hours\": 30, \"bandwidth\": 45}}\n"
            else:
                prompt += f"**ROLE:** You are the Marketing Team. You value bandwidth more highly.\n" 
                prompt += f"**SUGGESTED OPENING:** Consider offering around (25, 50) - high bandwidth for your needs.\n"
                prompt += f"**CONSTRAINT CHECK:** (25,50) → Total: 75≤100 ✅, GPU-BW: 275≤300 ✅, Valid!\n"
                prompt += f"**EXACT JSON:** {{\"type\": \"offer\", \"gpu_hours\": 25, \"bandwidth\": 50}}\n"
            
            prompt += f"**ACTION REQUIRED:** Make an opening offer with: {{\"type\": \"offer\", \"gpu_hours\": X, \"bandwidth\": Y}}\n"
            prompt += f"**CRITICAL:** Verify your offer satisfies ALL constraints above!\n\n"
        
        # Make the decision format more explicit for acceptance
        if proposed_utility and proposed_utility > current_batna:
            prompt += "**DECISION REQUIRED:** This offer is better than your BATNA.\n"
            prompt += "**RECOMMENDED ACTION:** Accept with: {\"type\": \"accept\"}\n\n"
        
        # Always show constraints so models know the rules
        prompt += f"**CONSTRAINTS - ALL OFFERS MUST SATISFY:**\n"
        prompt += f"1. **Total Resources:** gpu_hours + bandwidth ≤ {self.total_resources}\n"
        prompt += f"2. **GPU-Bandwidth Limit:** 3×gpu_hours + 4×bandwidth ≤ {self.constraints.get('gpu_bandwidth', 300)}\n"
        prompt += f"3. **Minimum GPU:** gpu_hours ≥ {self.constraints.get('min_gpu', 5)}\n"
        prompt += f"4. **Minimum Bandwidth:** bandwidth ≥ {self.constraints.get('min_bandwidth', 5)}\n\n"
        
        prompt += "**Your Options:**\n"
        prompt += "1. Make an offer: Specify allocation (gpu_hours=X, bandwidth=Y)\n"
        prompt += "2. Accept: Accept the current proposal\n"
        prompt += "3. Reject: Reject and continue negotiation\n\n"
        
        prompt += "**RESPONSE FORMAT:** Respond with ONLY valid JSON. No explanations.\n"
        prompt += "Valid responses:\n"
        prompt += '{"type": "offer", "gpu_hours": 25, "bandwidth": 40}\n'
        prompt += '{"type": "accept"}\n'
        prompt += '{"type": "reject"}\n'
        
        prompt += "\n**CRITICAL JSON REQUIREMENTS:**\n"
        prompt += "- For offers: MUST include both 'gpu_hours' and 'bandwidth' with numeric values\n"
        prompt += "- Example: {\"type\": \"offer\", \"gpu_hours\": 30, \"bandwidth\": 45}\n"
        prompt += "- DO NOT send incomplete JSON like {\"type\": \"offer\"}\n"
        prompt += "- Your offer MUST satisfy the constraints shown above!\n"
        prompt += "\nYour response:"
        
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
        """Get the current game prompt for a specific player - required by BaseGame interface."""
        if not hasattr(self, 'development') or not hasattr(self, 'marketing'):
            return "Game not initialized properly"
        
        # Use the human prompt method which has all the game logic
        return self.get_human_prompt(player_id, self.game_data)