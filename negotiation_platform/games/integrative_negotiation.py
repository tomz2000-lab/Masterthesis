from typing import Dict, List, Any, Optional, Tuple
from .base_game import BaseGame, PlayerAction


class IntegrativeNegotiationsGame(BaseGame):
    """
    Integrative negotiations game between IT and Marketing teams.
    Based on document specifications for office space and collaborative responsibilities.

    Four issues with point values:
    - Server Room Size: 50 sqm (10), 100 sqm (30), 150 sqm (60)
    - Meeting Room Access: 2 days/week (10), 4 days/week (30), 7 days/week (60)
    - Cleaning Responsibility: IT handles (30), Shared (50), Outsourced (10)
    - Branding Visibility: Minimal (10), Moderate (30), Prominent (60)
    """

    def __init__(self, config: Dict[str, Any]):
        # Initialize base class with game type as game_id
        super().__init__(game_id="integrative_negotiations", config=config)
        self.max_rounds = config.get("rounds", 10)  # Increased from 5 to 10 for complex negotiations
        self.batna_decay = config.get("batna_decay", 0.02)  # 2% per round as specified

        # Issue configurations from document
        self.issues = {
            "server_room": {
                "options": [50, 100, 150],  # sqm
                "points": [10, 30, 60],
                "labels": ["50 sqm", "100 sqm", "150 sqm"]
            },
            "meeting_access": {
                "options": [2, 4, 7],  # days per week
                "points": [10, 30, 60],
                "labels": ["2 days/week", "4 days/week", "7 days/week"]
            },
            "cleaning": {
                "options": ["IT", "Shared", "Outsourced"],
                "points": [30, 50, 10],
                "labels": ["IT handles", "Shared responsibility", "Outsourced"]
            },
            "branding": {
                "options": ["Minimal", "Moderate", "Prominent"],
                "points": [10, 30, 60],
                "labels": ["Minimal visibility", "Moderate visibility", "Prominent visibility"]
            }
        }

        # Preference weights from document specifications
        self.weights = {
            "IT": {
                "server_room": 0.4,  # 40%
                "meeting_access": 0.1,  # 10%
                "cleaning": 0.3,  # 30%
                "branding": 0.2  # 20%
            },
            "Marketing": {
                "server_room": 0.1,  # 10%
                "meeting_access": 0.4,  # 40%
                "cleaning": 0.2,  # 20%
                "branding": 0.3  # 30%
            }
        }

        # Base BATNA values (to be adjusted with decay)
        self.base_batnas = config.get("batnas", {"IT": 200, "Marketing": 220})

    def initialize_game(self, players: List[str]) -> Dict[str, Any]:
        """Initialize integrative negotiations game."""
        if len(players) != 2:
            raise ValueError("Integrative negotiations game requires exactly 2 players")

        self.players = players
        self.it_team = players[0]  # First player is IT
        self.marketing_team = players[1]  # Second player is Marketing

        self.game_data = {
            "game_type": "integrative_negotiations",
            "players": self.players,
            "rounds": self.max_rounds,
            "current_round": 1,
            "private_info": {
                self.it_team: {
                    "role": "IT",
                    "weights": self.weights["IT"],
                    "base_batna": self.base_batnas["IT"],
                    "preferences": "Prioritizes server room size and cleaning responsibility"
                },
                self.marketing_team: {
                    "role": "Marketing",
                    "weights": self.weights["Marketing"],
                    "base_batna": self.base_batnas["Marketing"],
                    "preferences": "Prioritizes meeting room access and branding visibility"
                }
            },
            "public_info": {
                "issues": list(self.issues.keys()),
                "deadline": self.max_rounds,
                "issue_descriptions": {
                    "server_room": "Server room size allocation (50-150 sqm)",
                    "meeting_access": "Meeting room access days per week (2-7 days)",
                    "cleaning": "Cleaning responsibility assignment",
                    "branding": "Branding visibility level"
                }
            },
            "proposals_history": [],
            "current_proposal": None,
            "round_proposals": {}  # Track proposals by round to prevent overwriting
        }
        
        return self.game_data

    def get_current_batna(self, player: str, round_num: int) -> float:
        """Calculate time-adjusted BATNA for current round."""
        base_batna = self.base_batnas.get("IT" if "IT" in player or player == self.it_team else "Marketing", 200)
        return base_batna * ((1 - self.batna_decay) ** (round_num - 1))

    def calculate_utility(self, player: str, proposal: Dict[str, Any]) -> float:
        """Calculate weighted utility for a given proposal."""
        # Determine player role
        player_role = "IT" if (player == self.it_team or "IT" in player) else "Marketing"
        player_weights = self.weights[player_role]

        total_utility = 0.0

        for issue, selection in proposal.items():
            if issue in self.issues:
                issue_config = self.issues[issue]

                # Find the index of the selected option
                try:
                    if isinstance(selection, str):
                        option_index = issue_config["options"].index(selection)
                    else:
                        option_index = issue_config["options"].index(selection)

                    # Get points for this selection
                    points = issue_config["points"][option_index]

                    # Apply weight for this player
                    weight = player_weights.get(issue, 0)
                    weighted_points = points * weight

                    total_utility += weighted_points

                except (ValueError, IndexError):
                    # Invalid selection, contribute 0 utility
                    continue

        return total_utility

    def is_valid_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Check if proposal contains valid selections for all issues."""
        if not proposal:
            return False

        # Check that all issues are addressed
        required_issues = set(self.issues.keys())
        proposed_issues = set(proposal.keys())

        if not required_issues.issubset(proposed_issues):
            return False

        # Check that all selections are valid options
        for issue, selection in proposal.items():
            if issue in self.issues:
                valid_options = self.issues[issue]["options"]
                if selection not in valid_options:
                    return False

        return True

    def is_valid_action(self, player: str, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Validate player action."""
        action_type = action.get("type", "")

        if action_type == "propose":
            proposal = action.get("proposal", {})
            return self.is_valid_proposal(proposal)

        elif action_type in ["accept", "reject", "counter", "counter-offer"]:
            return True

        return False

    def process_actions(self, actions: Dict[str, Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process player actions and update game state with fixed proposal tracking."""
        current_round = game_state["current_round"]

        # Separate different action types
        proposals = {player: action for player, action in actions.items()
                     if action.get("type") == "propose"}
        acceptances = {player: action for player, action in actions.items()
                       if action.get("type") == "accept"}
        rejections = {player: action for player, action in actions.items()
                      if action.get("type") == "reject"}
        counters = {player: action for player, action in actions.items()
                    if action.get("type") in ["counter", "counter-offer"]}

        # Track proposals by round to prevent overwriting issue
        round_key = f"round_{current_round}"
        if round_key not in game_state["round_proposals"]:
            game_state["round_proposals"][round_key] = {}

        # Record all proposals for this round
        for player, action in proposals.items():
            proposal_record = {
                "player": player,
                "round": current_round,
                "proposal": action.get("proposal", {})
            }
            game_state["proposals_history"].append(proposal_record)
            game_state["round_proposals"][round_key][player] = proposal_record
            
            # Set as current proposal, but don't let it get overwritten
            if not game_state.get("current_proposal"):
                game_state["current_proposal"] = proposal_record

        # Handle counter-proposals
        for player, action in counters.items():
            if "proposal" in action:
                proposal_record = {
                    "player": player,
                    "round": current_round,
                    "proposal": action.get("proposal", {}),
                    "type": "counter"
                }
                game_state["proposals_history"].append(proposal_record)
                game_state["round_proposals"][round_key][player] = proposal_record
                game_state["current_proposal"] = proposal_record

        # Check for mutual acceptance
        if len(acceptances) >= 1:  # At least one player accepts
            if game_state.get("current_proposal"):
                final_proposal = game_state["current_proposal"]["proposal"]
                return self._create_agreement(final_proposal, current_round, game_state)

        # Update round
        game_state["current_round"] += 1

        # Check deadline
        if game_state["current_round"] > self.max_rounds:
            return self._create_no_agreement(game_state)

        return game_state

    def _create_no_agreement(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create no agreement result with BATNA values."""
        current_round = game_state["current_round"]
        
        it_batna = self.get_current_batna(self.it_team, current_round)
        marketing_batna = self.get_current_batna(self.marketing_team, current_round)
        
        game_state.update({
            "agreement_reached": False,
            "final_utilities": {
                self.it_team: it_batna,
                self.marketing_team: marketing_batna
            },
            "termination_reason": "deadline_reached",
            "final_round": current_round
        })
        
        return game_state

    def _create_agreement(self, final_proposal: Dict[str, Any], round_num: int, game_state: Dict[str, Any]) -> Dict[
        str, Any]:
        """Create agreement result with utilities."""
        it_utility = self.calculate_utility(self.it_team, final_proposal)
        marketing_utility = self.calculate_utility(self.marketing_team, final_proposal)

        it_batna = self.get_current_batna(self.it_team, round_num)
        marketing_batna = self.get_current_batna(self.marketing_team, round_num)

        # Create detailed agreement summary
        agreement_details = {}
        for issue, selection in final_proposal.items():
            if issue in self.issues:
                issue_config = self.issues[issue]
                try:
                    option_index = issue_config["options"].index(selection)
                    agreement_details[issue] = {
                        "selection": selection,
                        "description": issue_config["labels"][option_index]
                    }
                except (ValueError, IndexError):
                    agreement_details[issue] = {"selection": selection, "description": str(selection)}

        game_state.update({
            "agreement_reached": True,
            "final_proposal": final_proposal,
            "agreement_details": agreement_details,
            "agreement_round": round_num,
            "final_utilities": {
                self.it_team: it_utility,
                self.marketing_team: marketing_utility
            },
            "batnas_at_agreement": {
                self.it_team: it_batna,
                self.marketing_team: marketing_batna
            },
            "utility_breakdown": {
                self.it_team: self._calculate_utility_breakdown(self.it_team, final_proposal),
                self.marketing_team: self._calculate_utility_breakdown(self.marketing_team, final_proposal)
            }
        })

        return game_state

    def _calculate_utility_breakdown(self, player: str, proposal: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate detailed utility breakdown by issue."""
        player_role = "IT" if (player == self.it_team or "IT" in player) else "Marketing"
        player_weights = self.weights[player_role]

        breakdown = {}
        for issue, selection in proposal.items():
            if issue in self.issues:
                issue_config = self.issues[issue]
                try:
                    option_index = issue_config["options"].index(selection)
                    raw_points = issue_config["points"][option_index]
                    weight = player_weights.get(issue, 0)
                    weighted_utility = raw_points * weight

                    breakdown[issue] = {
                        "raw_points": raw_points,
                        "weight": weight,
                        "weighted_utility": weighted_utility,
                        "selection": selection
                    }
                except (ValueError, IndexError):
                    breakdown[issue] = {
                        "raw_points": 0,
                        "weight": player_weights.get(issue, 0),
                        "weighted_utility": 0,
                        "selection": selection
                    }

        return breakdown

    def is_game_over(self, game_state: Dict[str, Any]) -> bool:
        """Check if game is finished."""
        return (game_state.get("agreement_reached", False) or
                game_state.get("current_round", 1) > self.max_rounds)

    def get_winner(self, game_state: Dict[str, Any]) -> Optional[str]:
        """Determine winner based on utility surplus relative to BATNA."""
        if not game_state.get("agreement_reached", False):
            return None

        final_utilities = game_state.get("final_utilities", {})
        batnas = game_state.get("batnas_at_agreement", {})

        if not final_utilities or not batnas:
            return None

        # Calculate surplus for each player
        surpluses = {}
        for player in final_utilities.keys():
            utility = final_utilities[player]
            batna = batnas[player]
            surpluses[player] = utility - batna

        return max(surpluses, key=surpluses.get)

    def get_game_summary(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive game summary."""
        summary = {
            "game_type": "Integrative Negotiations",
            "players": {
                self.it_team: "IT Team",
                self.marketing_team: "Marketing Team"
            },
            "agreement_reached": game_state.get("agreement_reached", False)
        }

        if game_state.get("agreement_reached", False):
            summary.update({
                "agreement_round": game_state.get("agreement_round"),
                "final_agreement": game_state.get("agreement_details", {}),
                "utilities": game_state.get("final_utilities", {}),
                "utility_breakdown": game_state.get("utility_breakdown", {})
            })
        else:
            summary["failure_reason"] = game_state.get("reason", "Unknown")

        return summary

    # Required abstract methods from BaseGame
    def process_action(self, action) -> Dict[str, Any]:
        """Process a single player action (required by base class)"""
        # Add action to history if it has the required structure
        if hasattr(action, 'player_id') and hasattr(action, 'action_data'):
            self.add_action(action)
        
        # Extract action data and player info
        action_data = action.action_data if hasattr(action, 'action_data') else action
        player = action.player_id if hasattr(action, 'player_id') else action.get('player', '')
        
        # Delegate to process_actions method with single action
        actions_dict = {player: action_data}
        if hasattr(self, 'game_data'):
            self.game_data = self.process_actions(actions_dict, self.game_data)
            return self.game_data
        else:
            # Game not initialized properly
            return {}
    
    def check_end_conditions(self) -> bool:
        """Check if the game should end (required by base class)"""
        if hasattr(self, 'game_data'):
            return self.is_game_over(self.game_data)
        return False
    
    def calculate_scores(self) -> Dict[str, float]:
        """Calculate final scores for all players (required by base class)"""
        if hasattr(self, 'game_data'):
            if self.game_data.get("agreement_reached", False):
                return self.game_data.get("final_utilities", {})
        return {player: 0.0 for player in getattr(self, 'players', [])}
    
    def get_system_prompt(self, player_id: str, game_state: Dict[str, Any]) -> str:
        """Get system prompt for integrative negotiations - follows resource allocation pattern."""
        return """You are an expert negotiator participating in an integrative negotiation about office space allocation.
Focus on value creation and finding win-win solutions.
Your responses must be valid JSON only."""

    def get_human_prompt(self, player_id: str, game_state: Dict[str, Any]) -> str:
        """Get detailed human prompt with fixed proposal visibility."""
        private_info = game_state.get("private_info", {}).get(player_id, {})
        current_round = game_state.get("current_round", 1)
        role = private_info.get("role", "unknown")
        
        # Get current BATNA for this player and round
        batna = self.get_current_batna(player_id, current_round)
        
        # Check for opponent's proposal from PREVIOUS round (simultaneous game fix)
        current_proposal = None
        round_proposals = game_state.get("round_proposals", {})
        
        # In simultaneous games, look at previous round for opponent proposal
        if current_round > 1:
            previous_round_key = f"round_{current_round - 1}"
            if previous_round_key in round_proposals:
                previous_round_data = round_proposals[previous_round_key]
                for other_player, proposal_data in previous_round_data.items():
                    if other_player != player_id:  # It's from the opponent
                        current_proposal = proposal_data
                        break
        
        base_prompt = f"""**Round {current_round}/{self.max_rounds} - Office Space Negotiation**
You are participating as the {role} team in this negotiation.
Your current BATNA (fallback option): {batna:.1f} points

"""

        # Add proposal context with car-game-style clear decision guidance
        if current_proposal:
            proposal = current_proposal['proposal']
            # Calculate utility for current proposal
            if role == "IT":
                weights = {"server_room": 0.4, "cleaning": 0.3, "branding": 0.2, "meeting_access": 0.1}
            else:  # Marketing
                weights = {"meeting_access": 0.4, "branding": 0.3, "cleaning": 0.2, "server_room": 0.1}
            
            # Simple utility calculation for guidance
            utility_scores = {
                "server_room": {50: 10, 100: 30, 150: 60},
                "meeting_access": {2: 10, 4: 30, 7: 60},
                "cleaning": {"IT": 30, "Shared": 50, "Outsourced": 10},
                "branding": {"Minimal": 10, "Moderate": 30, "Prominent": 60}
            }
            
            total_utility = sum(weights[issue] * utility_scores[issue][proposal[issue]] for issue in weights)
            
            # Clear binary decision guidance like car game
            decision_guidance = "ACCEPT IT" if total_utility >= batna else "COUNTER-OFFER"
            offer_status = "acceptable" if total_utility >= batna else "below your BATNA"
            
            base_prompt += f"""**Other team's proposal:** {proposal}
Your utility from this proposal: {total_utility:.1f} points
Your BATNA (fallback): {batna:.1f} points
**Decision: {decision_guidance}** (proposal is {offer_status})

"""
        elif current_round == 1:
            base_prompt += "**This is the opening round.** Consider making an initial proposal to start the negotiation.\n\n"

        # Role-specific information
        if role == "IT":
            base_prompt += f"""**Your Role: IT Department**
You need office space that supports your technical operations and team productivity.

**Your Priorities (importance weights):**
- Server Room Size: 40% (you need adequate space for servers and equipment)
- Cleaning Responsibility: 30% (you prefer shared arrangements)
- Branding Visibility: 20% (moderate visibility works for you)
- Meeting Room Access: 10% (you need some meeting access but it's not critical)

**Available Options for Each Issue:**
- **Server Room Size:** 50 sqm (10 pts), 100 sqm (30 pts), or 150 sqm (60 pts)
- **Meeting Room Access:** 2 days/week (10 pts), 4 days/week (30 pts), or 7 days/week (60 pts)
- **Cleaning Responsibility:** "IT" (30 pts), "Shared" (50 pts), or "Outsourced" (10 pts)
- **Branding Visibility:** "Minimal" (10 pts), "Moderate" (30 pts), or "Prominent" (60 pts)

TASK: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "accept"}}
{{"type": "propose", "proposal": {{"server_room": 100, "meeting_access": 4, "cleaning": "Shared", "branding": "Moderate"}}}}

Your response:"""

        elif role == "Marketing":
            base_prompt += f"""**Your Role: Marketing Department**
You need office space that enhances your team's visibility and client interaction capabilities.

**Your Priorities (importance weights):**
- Meeting Room Access: 40% (critical for client meetings and presentations)
- Branding Visibility: 30% (important for company image)
- Cleaning Responsibility: 20% (you prefer outsourced cleaning)
- Server Room Size: 10% (not a priority for your team)

**Available Options for Each Issue:**
- **Server Room Size:** 50 sqm (10 pts), 100 sqm (30 pts), or 150 sqm (60 pts)
- **Meeting Room Access:** 2 days/week (10 pts), 4 days/week (30 pts), or 7 days/week (60 pts)
- **Cleaning Responsibility:** "IT" (30 pts), "Shared" (50 pts), or "Outsourced" (10 pts)
- **Branding Visibility:** "Minimal" (10 pts), "Moderate" (30 pts), or "Prominent" (60 pts)

TASK: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "accept"}}
{{"type": "propose", "proposal": {{"server_room": 100, "meeting_access": 7, "cleaning": "Outsourced", "branding": "Prominent"}}}}

Your response:"""

        return base_prompt

    def get_game_prompt(self, player_id: str, game_state: Dict[str, Any] = None) -> str:
        """Get the current game prompt for a specific player (required by base class)"""
        # Use provided game_state if available, otherwise fall back to self.game_data
        current_state = game_state if game_state is not None else getattr(self, 'game_data', {})
        if not current_state:
            return "Game not initialized properly"
        return self.get_human_prompt(player_id, current_state)
