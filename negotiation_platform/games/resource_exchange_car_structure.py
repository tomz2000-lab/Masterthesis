import random
from typing import Dict, List, Any, Optional, Tuple
from .base_game import BaseGame, PlayerAction


class ResourceAllocationGame(BaseGame):
    """
    Resource allocation negotiation game between Development and Marketing teams.
    - Development Team needs GPU hours (x) and bandwidth (y) - utility: 12x + 3y + ε  
    - Marketing Team needs GPU hours (x) and bandwidth (y) - utility: 3x + 12y + i
    - Constraints: x + y ≤ 100, 3x + 4y ≤ 240, x ≥ 20, y ≥ 15
    - BATNAs: Development 300 (external provider), Marketing 360 (alternative SaaS)
    - 5 rounds with time-adjusted BATNA decay
    """

    def __init__(self, config: Dict[str, Any]):
        # Initialize base class - exact same pattern as car game
        super().__init__(game_id="resource_allocation", config=config)
        self.total_resources = config.get("total_resources", 100)
        self.constraints = config.get("constraints", {
            "gpu_bandwidth": 240,  # 3x + 4y <= 240
            "min_gpu": 20,         # x >= 20  
            "min_bandwidth": 15    # y >= 15
        })
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
                    "utility_function": "12x + 3y + ε",
                    "batna": self.development_batna,
                    "constraints": self.constraints
                },
                self.marketing: {
                    "role": "marketing",
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
        return (x + y <= self.total_resources and
                3 * x + 4 * y <= self.constraints["gpu_bandwidth"] and
                x >= self.constraints["min_gpu"] and
                y >= self.constraints["min_bandwidth"] and
                x >= 0 and y >= 0)

    def is_valid_action(self, player: str, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Validate player action - same pattern as car game."""
        action_type = action.get("type", "")

        if action_type == "offer":
            gpu_hours = action.get("gpu_hours", 0)
            bandwidth = action.get("bandwidth", 0)
            
            if gpu_hours <= 0 or bandwidth <= 0:
                return False

            # Validate allocation constraints
            return self.is_valid_allocation(gpu_hours, bandwidth)

        elif action_type in ["accept", "reject"]:
            return True

        elif action_type in ["counter", "counteroffer"]:
            gpu_hours = action.get("gpu_hours", 0)
            bandwidth = action.get("bandwidth", 0)
            return gpu_hours > 0 and bandwidth > 0 and self.is_valid_allocation(gpu_hours, bandwidth)

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
                # Find the most recent offer from the other player
                other_player = self.marketing if player_id == self.development else self.development
                recent_offers = [o for o in game_state.get("offers", []) if o["player"] == other_player]
                
                if recent_offers:
                    accepted_offer = recent_offers[-1]
                    x, y = accepted_offer["gpu_hours"], accepted_offer["bandwidth"]
                    
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
                    return game_state
                    
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

**NEGOTIATION STRATEGY:**
- Focus on maximizing GPU hours (12x coefficient vs 3y)
- Accept offers that give you utility > {current_batna:.1f}
- Time pressure increases - your BATNA decreases each round
- Consider that Marketing Team prefers bandwidth (their utility: 3x + 12y + i)

**ROUND {current_round}/{self.max_rounds}** - Make strategic offers and respond to counteroffers."""

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

**NEGOTIATION STRATEGY:**
- Focus on maximizing bandwidth (12y coefficient vs 3x)
- Accept offers that give you utility > {current_batna:.1f}
- Time pressure increases - your BATNA decreases each round
- Consider that Development Team prefers GPU hours (their utility: 12x + 3y + ε)

**ROUND {current_round}/{self.max_rounds}** - Make strategic offers and respond to counteroffers."""

    def get_human_prompt(self, player_id: str, game_state: Dict[str, Any]) -> str:
        """Generate context-aware prompt for human player - adapted from car game."""
        current_round = game_state["current_round"]
        current_batna = self.get_current_batna(player_id, current_round)
        
        prompt = f"\n**Round {current_round}/{self.max_rounds} - Resource Allocation Negotiation**\n"
        prompt += f"Your current BATNA: {current_batna:.1f}\n\n"
        
        # Show recent offers if any
        recent_offers = game_state.get("offers", [])
        if recent_offers:
            last_offer = recent_offers[-1]
            x, y = last_offer["gpu_hours"], last_offer["bandwidth"]
            
            proposed_utility = self.calculate_utility(player_id, x, y, current_round)
            
            prompt += f"**Latest Proposal:**\n"
            prompt += f"- GPU Hours: {x}, Bandwidth: {y}\n"
            prompt += f"- Your utility from this: {proposed_utility:.1f}\n"
            prompt += f"- Accept (utility > BATNA)? Your BATNA: {current_batna:.1f}\n\n"
        
        prompt += "**Your Options:**\n"
        prompt += "1. Make an offer: Specify allocation (gpu_hours=X, bandwidth=Y)\n"
        prompt += "2. Accept: Accept the current proposal\n"
        prompt += "3. Reject: Reject and continue negotiation\n"
        
        return prompt