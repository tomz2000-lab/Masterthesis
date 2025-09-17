import random
from typing import Dict, List, Any, Optional, Tuple
from .base_game import BaseGame, PlayerAction

class ResourceAllocationGame(BaseGame):
    """
    Resource allocation game - MINIMAL VERSION LIKE WORKING STUB
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize exactly like car game stub that worked
        super().__init__(game_id="resource_allocation", config=config)
        self.starting_price = config.get("starting_price", 45000)
        self.buyer_budget = config.get("buyer_budget", 40000)
        self.seller_cost = config.get("seller_cost", 38000)
        self.buyer_batna = config.get("buyer_batna", 44000)
        self.seller_batna = config.get("seller_batna", 39000)
        self.max_rounds = config.get("rounds", 5)
        self.batna_decay = config.get("batna_decay", {"buyer": 0.02, "seller": 0.01})
        
    def initialize_game(self, players: List[str]) -> Dict[str, Any]:
        """Initialize resource allocation game - SAME STRUCTURE AS CAR GAME."""
        if len(players) != 2:
            raise ValueError("Resource allocation game requires exactly 2 players")

        self.players = players
        self.development = players[0]  # First player is development team  
        self.marketing = players[1]    # Second player is marketing team
        self.state = self.state.__class__.ACTIVE  # CRITICAL: Set to active state (like car game)

        # Build game data structure EXACTLY like car game pattern
        self.game_data = {
            "game_type": "resource_allocation",
            "players": self.players,
            "rounds": self.max_rounds,
            "current_round": 1,
            "private_info": {
                self.development: {
                    "role": "development",
                    "team": "Development Team",
                    "utility_function": "12x + 3y + ε",
                    "batna": self.development_batna,
                    "description": "GPU-intensive machine learning projects"
                },
                self.marketing: {
                    "role": "marketing",
                    "team": "Marketing Team", 
                    "utility_function": "3x + 12y + i",
                    "batna": self.marketing_batna,
                    "description": "bandwidth-intensive analytics and campaigns"
                }
            },
            "public_info": {
                "deadline": self.max_rounds,
                "total_resources": self.total_resources,
                "constraints": {
                    "gpu_bandwidth": self.gpu_bandwidth_constraint,
                    "min_gpu": self.min_gpu,
                    "min_bandwidth": self.min_bandwidth
                }
            }
        }
        return self.game_data
    
    def get_current_batna(self, player: str, round_num: int) -> float:
        """Calculate time-adjusted BATNA for current round - SAME PATTERN AS CAR GAME."""
        if player == self.development:
            decay_rate = self.batna_decay["development"]
            base_batna = self.development_batna
        else:
            decay_rate = self.batna_decay["marketing"]
            base_batna = self.marketing_batna
        
        current_batna = base_batna * ((1 - decay_rate) ** (round_num - 1))
        return current_batna
    
    def calculate_utility(self, player: str, x: float, y: float, round_num: int) -> float:
        """Calculate utility with uncertainty factors - SIMPLIFIED FOR STABILITY."""
        if player == self.development:
            # Development: 12x + 3y + ε (simplified: small random component)
            epsilon = random.gauss(0, 1)  # Reduced variance for stability
            base_utility = 12 * x + 3 * y
            final_utility = base_utility + epsilon
            return final_utility
        else:
            # Marketing: 3x + 12y + i (simplified: small random component)  
            i_factor = random.uniform(-0.02, 0.02)  # Reduced volatility for stability
            base_utility = 3 * x + 12 * y
            final_utility = base_utility * (1 + i_factor)
            return final_utility
    
    def is_valid_allocation(self, x: float, y: float) -> bool:
        """Check if allocation satisfies all constraints."""
        constraints = [
            x + y <= self.total_resources,           # x + y <= 100
            3 * x + 4 * y <= self.constraints["gpu_bandwidth"],  # 3x + 4y <= 240
            x >= self.constraints["min_gpu"],         # x >= 20
            y >= self.constraints["min_bandwidth"],   # y >= 15
            x >= 0,
            y >= 0
        ]
        return all(constraints)
    
    def is_valid_action(self, player: str, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Validate player action."""
        action_type = action.get("type", "")
        
        if action_type == "propose":
            x = action.get("gpu_hours", 0)
            y = action.get("bandwidth", 0)
            
            # Check constraints
            if not self.is_valid_allocation(x, y):
                return False
                
            # Check if proposal meets BATNA requirement
            current_batna = self.get_current_batna(player, game_state["current_round"])
            utility = self.calculate_utility(player, x, y, game_state["current_round"])
            
            return utility >= current_batna
            
        elif action_type in ["accept", "reject"]:
            return True
            
        return False
    
    def process_actions(self, actions: Dict[str, Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process player actions with turn-based logic."""
        current_round = game_state["current_round"]
        current_turn = game_state.get("current_turn", self.development)
        waiting_for_response = game_state.get("waiting_for_response", False)
        
        print(f"🎯 [DEBUG] Round {current_round}: Current turn = {current_turn}, Waiting for response = {waiting_for_response}")
        
        # Only process action from the player whose turn it is
        if current_turn not in actions:
            print(f"⚠️ [DEBUG] No action from {current_turn} (whose turn it is)")
            return game_state
            
        action = actions[current_turn]
        action_type = action.get("type", "")
        
        print(f"🎮 [DEBUG] {current_turn} action: {action}")
        
        if not waiting_for_response:
            # Expecting a proposal from current player
            if action_type == "propose":
                offer = {
                    "player": current_turn,
                    "round": current_round,
                    "gpu_hours": action.get("gpu_hours", 0),
                    "bandwidth": action.get("bandwidth", 0)
                }
                game_state["offers_history"].append(offer)
                
                # Switch turn to other player for response
                other_player = self.marketing if current_turn == self.development else self.development
                game_state["current_turn"] = other_player
                game_state["waiting_for_response"] = True
                
                print(f"✅ [DEBUG] {current_turn} proposed ({offer['gpu_hours']}, {offer['bandwidth']}). Now {other_player}'s turn to respond.")
                
            elif action_type == "accept":
                print(f"❌ [DEBUG] {current_turn} tried to accept but no proposal exists to respond to")
                # Invalid - ignore
                
            else:
                print(f"⚠️ [DEBUG] {current_turn} made invalid action {action_type} when proposal expected")
                
        else:
            # Waiting for response to a proposal
            if action_type == "accept":
                # Check if model incorrectly included values in accept action
                if any(key in action for key in ["gpu_hours", "bandwidth"]):
                    included_values = {k: v for k, v in action.items() if k in ["gpu_hours", "bandwidth"]}
                    print(f"⚠️ [DEBUG] {current_turn} included values in accept {included_values} - these are IGNORED")
                
                # Pure acceptance - find the last proposal (the one being accepted)
                if game_state["offers_history"]:
                    last_offer = game_state["offers_history"][-1]
                    x, y = last_offer["gpu_hours"], last_offer["bandwidth"]
                    proposer = last_offer["player"]
                    
                    if self.is_valid_allocation(x, y):
                        print(f"🤝 [DEBUG] {current_turn} accepted {proposer}'s proposal: ({x}, {y}) GPU hours/bandwidth - Agreement reached!")
                        return self._create_agreement(x, y, current_round, game_state)
                    else:
                        print(f"❌ [DEBUG] Proposed allocation ({x}, {y}) is invalid")
                        
            elif action_type == "propose":
                # Counter-proposal
                offer = {
                    "player": current_turn,
                    "round": current_round,
                    "gpu_hours": action.get("gpu_hours", 0),
                    "bandwidth": action.get("bandwidth", 0)
                }
                game_state["offers_history"].append(offer)
                
                # Switch turn back to other player
                other_player = self.marketing if current_turn == self.development else self.development
                game_state["current_turn"] = other_player
                # Still waiting for response to this new proposal
                
                print(f"🔄 [DEBUG] {current_turn} counter-proposed ({offer['gpu_hours']}, {offer['bandwidth']}). Now {other_player}'s turn to respond.")
                
            else:
                print(f"⚠️ [DEBUG] {current_turn} made invalid action {action_type} when response expected")
        
        # Check if we've reached max rounds
        if current_round >= self.max_rounds:
            print(f"⏰ [DEBUG] Max rounds ({self.max_rounds}) reached - No agreement")
            return self._create_no_agreement(game_state)
            
        return game_state
    
    def _create_agreement(self, x: float, y: float, round_num: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create agreement result with utilities."""
        dev_utility = self.calculate_utility(self.development, x, y, round_num)
        marketing_utility = self.calculate_utility(self.marketing, x, y, round_num)
        
        dev_batna = self.get_current_batna(self.development, round_num)
        marketing_batna = self.get_current_batna(self.marketing, round_num)
        
        game_state.update({
            "agreement_reached": True,
            "final_allocation": {"gpu_hours": x, "bandwidth": y},
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
                self.development: 0,
                self.marketing: 0
            }
        })
        
        return game_state
    
    def is_game_over(self, game_state: Dict[str, Any]) -> bool:
        """Check if game is finished."""
        return (game_state.get("agreement_reached", False) or 
                game_state.get("current_round", 1) > self.max_rounds)
    
    def get_winner(self, game_state: Dict[str, Any]) -> Optional[str]:
        """Determine winner based on utility surplus."""
        if not game_state.get("agreement_reached", False):
            return None
            
        utilities = game_state.get("final_utilities", {})
        batnas = game_state.get("batnas_at_agreement", {})
        
        if not utilities or not batnas:
            return None
            
        surpluses = {player: utilities[player] - batnas[player] 
                    for player in utilities.keys()}
        
        return max(surpluses, key=surpluses.get)
    
    def process_action(self, action) -> Dict[str, Any]:
        """Process a single player action (required by base class)"""
        # This method is required by the abstract base class but not used
        # in our current implementation since we use process_actions instead
        return {}
    
    def check_end_conditions(self) -> bool:
        """Check if the game should end (required by base class)"""
        return self.is_game_over(getattr(self, 'game_data', {}))
    
    def calculate_scores(self) -> Dict[str, float]:
        """Calculate final scores for all players (required by base class)"""
        if hasattr(self, 'game_data'):
            if self.game_data.get("agreement_reached", False):
                return self.game_data.get("final_utilities", {})
        return {player: 0.0 for player in getattr(self, 'players', [])}
    
    def get_game_prompt(self, player_id: str) -> str:
        """Get role-based game prompt with decision guidance (following successful car game design)"""
        if not hasattr(self, 'game_data'):
            return "Game not initialized properly"
            
        private_info = self.game_data.get("private_info", {}).get(player_id, {})
        current_round = self.game_data.get("current_round", 1)
        current_turn = self.game_data.get("current_turn", self.development)
        waiting_for_response = self.game_data.get("waiting_for_response", False)
        team = private_info.get("team", player_id).upper()
        offers_history = self.game_data.get("offers_history", [])
        
        # Calculate current BATNA with time decay
        current_batna = self.get_current_batna(player_id, current_round)
        
        # Role-based identity establishment
        if team == "DEVELOPMENT":
            role_desc = "DEVELOPMENT TEAM responsible for GPU-intensive machine learning projects"
            your_focus = "You need GPU hours for training AI models and processing large datasets"
            your_utility = "12x + 3y + ε (where ε represents stochastic demand uncertainty)"
        else:
            role_desc = "MARKETING TEAM responsible for bandwidth-intensive analytics and campaigns"  
            your_focus = "You need bandwidth for data analysis, customer analytics, and digital campaigns"
            your_utility = "3x + 12y + i (where i represents market volatility)"
        
        # Determine decision context and guidance
        is_my_turn = (current_turn == player_id)
        
        if not offers_history:
            # Initial state - no proposals yet
            if is_my_turn and not waiting_for_response:
                decision_guidance = "Decision: MAKE INITIAL PROPOSAL"
                action_reason = f"As {team}, you should propose a resource allocation that maximizes your utility while being acceptable to Marketing."
            else:
                decision_guidance = "Decision: WAIT FOR PROPOSAL"
                action_reason = "Wait for the other team to make the initial proposal."
        else:
            # There are existing proposals
            last_offer = offers_history[-1]
            last_proposer = last_offer["player"]
            last_gpu = last_offer["gpu_hours"] 
            last_bandwidth = last_offer["bandwidth"]
            
            if last_proposer == player_id:
                # I made the last proposal
                decision_guidance = "Decision: WAIT FOR RESPONSE"
                action_reason = f"Wait for the other team to respond to your proposal of {last_gpu} GPU hours and {last_bandwidth} GB/s bandwidth."
            else:
                # Other team made the last proposal - I need to respond
                if is_my_turn and waiting_for_response:
                    # Calculate utility and BATNA to determine if offer is acceptable
                    proposed_utility = self.calculate_utility(player_id, last_gpu, last_bandwidth, current_round)
                    
                    if proposed_utility >= current_batna:
                        decision_guidance = "Decision: ACCEPT IT"
                        action_reason = f"The proposed allocation ({last_gpu} GPU, {last_bandwidth} bandwidth) gives you utility {proposed_utility:.1f}, which is acceptable (above your BATNA of {current_batna:.1f})."
                    else:
                        decision_guidance = "Decision: COUNTER-OFFER"
                        action_reason = f"The proposed allocation ({last_gpu} GPU, {last_bandwidth} bandwidth) gives you utility {proposed_utility:.1f}, which is too low (below your BATNA of {current_batna:.1f})."
                else:
                    decision_guidance = "Decision: WAIT YOUR TURN"
                    action_reason = "Wait for your turn to respond to the other team's proposal."

        # Build the role-based prompt with explicit decision guidance
        prompt = f"""You are a {role_desc}.

{your_focus}. Your utility function is {your_utility}.

Current Situation (Round {current_round}/{self.max_rounds}):
- Your current BATNA (reservation value): {current_batna:.1f}
- Resource constraints: 3*GPU + 4*bandwidth ≤ 240, GPU ≥ 20, bandwidth ≥ 15
"""
        
        # Add offer history context
        if offers_history:
            prompt += f"\nLatest Proposal: {offers_history[-1]['gpu_hours']} GPU hours, {offers_history[-1]['bandwidth']} GB/s bandwidth (from {offers_history[-1]['player']})\n"
        else:
            prompt += "\nNo proposals have been made yet.\n"
            
        prompt += f"""
{decision_guidance}

Reasoning: {action_reason}

TASK: Respond with ONLY valid JSON (no explanations or additional text):"""

        # Add specific JSON format based on decision
        if "MAKE INITIAL PROPOSAL" in decision_guidance or "COUNTER-OFFER" in decision_guidance:
            prompt += '\n{"type": "propose", "gpu_hours": X, "bandwidth": Y}'
        elif "ACCEPT IT" in decision_guidance:
            prompt += '\n{"type": "accept"}'
        else:
            prompt += '\n{"type": "noop"}'
            
        return prompt
