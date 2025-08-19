import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .base_game import BaseGame

class ResourceAllocationGame(BaseGame):
    """
    Resource allocation game between Development and Marketing teams.
    Based on document specifications with stochastic demand and market volatility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(game_id="resource_allocation", config=config)
        self.total_resources = config.get("total_resources", 100)
        self.constraints = config.get("constraints", {
            "gpu_bandwidth": 240,  # 3x + 4y <= 240
            "min_gpu": 20,         # x >= 20
            "min_bandwidth": 15    # y >= 15
        })
        self.batnas = config.get("batnas", {"development": 300, "marketing": 360})
        self.batna_decay = config.get("batna_decay", {"development": 0.03, "marketing": 0.02})
        self.max_rounds = config.get("rounds", 5)
        
    def initialize_game(self, players: List[str]) -> Dict[str, Any]:
        """Initialize resource allocation game."""
        if len(players) != 2:
            raise ValueError("Resource allocation game requires exactly 2 players")
            
        self.players = players
        self.development = players[0]  # Development team
        self.marketing = players[1]    # Marketing team
        
        return {
            "game_type": "resource_allocation",
            "players": self.players,
            "rounds": self.max_rounds,
            "current_round": 1,
            "private_info": {
                self.development: {
                    "team": "development",
                    "utility_function": "12x + 3y + ε",
                    "batna": self.batnas["development"],
                    "batna_decay": self.batna_decay["development"],
                    "uncertainty": "stochastic_demand"  # ε ~ N(0,5)
                },
                self.marketing: {
                    "team": "marketing",
                    "utility_function": "3x + 12y + i", 
                    "batna": self.batnas["marketing"],
                    "batna_decay": self.batna_decay["marketing"],
                    "uncertainty": "market_volatility"  # i ~ U(-8%, +8%)
                }
            },
            "public_info": {
                "total_resources": self.total_resources,
                "constraints": self.constraints,
                "deadline": self.max_rounds
            },
            "offers_history": []
        }
    
    def get_current_batna(self, player: str, round_num: int) -> float:
        """Calculate time-adjusted BATNA for current round."""
        if player == self.development:
            decay_rate = self.batna_decay["development"]
            base_batna = self.batnas["development"]
        else:
            decay_rate = self.batna_decay["marketing"]
            base_batna = self.batnas["marketing"]
        
        current_batna = base_batna * ((1 - decay_rate) ** (round_num - 1))
        print(f"[DEBUG] {player} BATNA: base={base_batna}, decay_rate={decay_rate}, round={round_num}, current={current_batna:.4f}")
        return current_batna
    
    def calculate_utility(self, player: str, x: float, y: float, round_num: int) -> float:
        """Calculate utility with uncertainty factors."""
        print(f"[DEBUG] calculate_utility called with: player={player}, x={x}, y={y}, round={round_num}")
        if player == self.development:
            # Development: 12x + 3y + ε (stochastic demand N(0,5))
            epsilon = np.random.normal(0, 5)
            base_utility = 12 * x + 3 * y
            final_utility = base_utility + epsilon
            print(f"[DEBUG] Development utility: x={x}, y={y}, base=12*{x}+3*{y}={base_utility}, epsilon={epsilon:.4f}, final={final_utility:.4f}")
            return final_utility
        else:
            # Marketing: 3x + 12y + i (market volatility U(-8%, +8%))  
            i_factor = np.random.uniform(-0.08, 0.08)
            base_utility = 3 * x + 12 * y
            final_utility = base_utility * (1 + i_factor)
            print(f"[DEBUG] Marketing utility: x={x}, y={y}, base=3*{x}+12*{y}={base_utility}, i_factor={i_factor:.6f}, multiplier={1+i_factor:.6f}, final={final_utility:.4f}")
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
        """Process player actions and update game state."""
        current_round = game_state["current_round"]
        
        # Separate proposals and responses
        proposals = {player: action for player, action in actions.items() 
                    if action.get("type") == "propose"}
        responses = {player: action for player, action in actions.items() 
                    if action.get("type") in ["accept", "reject"]}
        
        # Record offers
        for player, action in proposals.items():
            offer = {
                "player": player,
                "round": current_round,
                "gpu_hours": action.get("gpu_hours", 0),
                "bandwidth": action.get("bandwidth", 0)
            }
            game_state["offers_history"].append(offer)
        
        # Check for mutual acceptance
        acceptances = [player for player, action in responses.items() 
                      if action.get("type") == "accept"]
        
        if len(acceptances) == 2:  # Both players accept
            # Find the most recent valid proposal
            if game_state["offers_history"]:
                last_offer = game_state["offers_history"][-1]
                x, y = last_offer["gpu_hours"], last_offer["bandwidth"]
                
                # Validate final allocation
                if self.is_valid_allocation(x, y):
                    return self._create_agreement(x, y, current_round, game_state)
        
        # Update round
        game_state["current_round"] += 1
        
        # Check deadline
        if game_state["current_round"] > self.max_rounds:
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
        """Get the current game prompt for a specific player (required by base class)"""
        if not hasattr(self, 'game_data'):
            return "Game not initialized properly"
            
        private_info = self.game_data.get("private_info", {}).get(player_id, {})
        current_round = self.game_data.get("current_round", 1)
        team = private_info.get("team", player_id)
        
        return f"""You are team {team.upper()} in a resource allocation negotiation.
Round {current_round} - Make your move by proposing a trade or accepting an offer.
Available actions: propose_trade, accept"""
