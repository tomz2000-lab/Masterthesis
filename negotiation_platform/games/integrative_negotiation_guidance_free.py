from typing import Dict, List, Any, Optional, Tuple
import random
import re
import json

from .base_game import BaseGame, PlayerAction
from .negotiation_tools import calculate_percentage_difference, calculate_utility


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

        # Base BATNA values - must match point-based utility scale (max ~57 points)
        # Set BATNAs to create negotiation pressure but be achievable
        self.base_batnas = config.get("batnas", {"IT": 35, "Marketing": 30})

    def validate_json_response(self, response: str) -> bool:
        """Check if response is valid JSON with proper structure."""
        try:
            data = json.loads(response.strip())
            return isinstance(data, dict) and "type" in data
        except (json.JSONDecodeError, TypeError):
            return False

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse pure JSON response format similar to company car game."""
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
            
            # Try to extract type and proposal manually as fallback
            type_match = re.search(r'"type":\s*"([^"]+)"', response)
            proposal_match = re.search(r'"proposal":\s*(\{[^}]+\})', response)
            if type_match:
                decision_data = {"type": type_match.group(1)}
                if proposal_match:
                    try:
                        proposal_data = json.loads(proposal_match.group(1))
                        decision_data["proposal"] = proposal_data
                    except:
                        pass
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
        """Initialize integrative negotiations game with randomized role assignment."""
        if len(players) != 2:
            raise ValueError("Integrative negotiations game requires exactly 2 players")

        self.players = players
        
        # RANDOMIZE ROLE ASSIGNMENT - Fix for positional bias
        import random
        roles = ["IT", "Marketing"]
        random.shuffle(roles)
        
        # Assign randomized roles
        player_roles = {players[0]: roles[0], players[1]: roles[1]}
        
        # Set role mappings based on randomization
        if player_roles[players[0]] == "IT":
            self.it_team = players[0]
            self.marketing_team = players[1]
        else:
            self.it_team = players[1] 
            self.marketing_team = players[0]
        
        print(f"🎲 [ROLE ASSIGNMENT] {players[0]} -> {player_roles[players[0]]}, {players[1]} -> {player_roles[players[1]]}")

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
            "round_proposals": {},  # Track proposals by round to prevent overwriting
            "role_assignments": player_roles  # Track who got which role
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
        """Validate player action with enhanced structured format support and proposal limits."""
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

        if action_type == "propose":
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} tried to make proposal but exceeded limit ({player_proposals}/{max_proposals})")
                return False
                
            proposal = action_data.get("proposal", {})
            return self.is_valid_proposal(proposal)

        elif action_type in ["accept", "reject"]:
            return True

        elif action_type in ["counter", "counter-offer"]:
            # Check proposal limit for counters too
            if player_proposals >= max_proposals:
                return False
                
            # Treat counter/counter-offer as regular proposals
            proposal = action_data.get("proposal", {})
            return self.is_valid_proposal(proposal)

        elif action_type == "noop":
            # Allow no-op actions as fallback
            return True

        return False

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate the LLM response to ensure it contains a valid action."""
        required_keys = {"type", "proposal"}
        if not required_keys.issubset(response.keys()):
            print(f"Invalid response: Missing keys in {response}")
            return False

        if response["type"] not in {"propose", "accept", "reject"}:
            print(f"Invalid response type: {response['type']}")
            return False

        return True

    def process_actions(self, actions: Dict[str, Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process player actions with proposal limits and enhanced JSON validation."""
        current_round = game_state["current_round"]
        max_proposals = self.max_rounds  # Use rounds from YAML config

        # Initialize proposal counters if not present
        for player in [self.it_team, self.marketing_team]:
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

        # Normalize action types - treat counter/counter-offer as proposals, handle empty types
        normalized_actions = {}
        for player, action in processed_actions.items():
            action_type = action.get("type", "")
            
            # Handle empty or invalid action types
            if not action_type or action_type == "":
                print(f"⚠️ Player {player} provided empty action type, treating as reject")
                normalized_actions[player] = {"type": "reject"}
            elif action_type in ["counter", "counter-offer"]:
                # Convert to propose
                normalized_actions[player] = {"type": "propose", "proposal": action.get("proposal", {})}
            else:
                normalized_actions[player] = action

        # Check for offers and responses
        proposals = {player: action for player, action in normalized_actions.items()
                    if action.get("type") == "propose"}
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

        # Process proposals with proposal limit validation
        for player, action in proposals.items():
            player_proposals = game_state.get(f"{player}_proposal_count", 0)
            
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} exceeded proposal limit ({player_proposals}/{max_proposals}). Ignoring additional proposals.")
                # Don't process this proposal, but don't end negotiation unless they also rejected
                continue
            
            # Valid proposal - process it
            proposal = action.get("proposal", {})
            if self.is_valid_proposal(proposal):
                game_state[f"{player}_last_proposal"] = proposal
                game_state[f"{player}_last_proposal_round"] = current_round  # Track when proposal was made
                game_state[f"{player}_proposal_count"] = player_proposals + 1
                print(f"💡 Player {player} made proposal (#{player_proposals + 1}/{max_proposals}): {proposal}")
                
                # Track in proposals history for compatibility
                proposal_record = {
                    "player": player,
                    "round": current_round,
                    "proposal": proposal
                }
                game_state["proposals_history"].append(proposal_record)
                game_state["current_proposal"] = proposal_record
            else:
                print(f"⚠️ Player {player} made invalid proposal: {proposal}")

        # Process acceptances (rejections already handled above)
        for player, action in responses.items():
            if action.get("type") == "accept":
                # Find the proposal being accepted (from the other player)
                other_player = self.marketing_team if player == self.it_team else self.it_team
                if f"{other_player}_last_proposal" in game_state:
                    accepted_proposal = game_state[f"{other_player}_last_proposal"]
                    # Get the round when the accepted proposal was made
                    proposal_round = game_state.get(f"{other_player}_last_proposal_round", current_round)
                    print(f"✅ Player {player} accepted proposal: {accepted_proposal} (made in round {proposal_round})")
                    return self._create_agreement(accepted_proposal, proposal_round, game_state)
                else:
                    print(f"⚠️ Player {player} tried to accept but no proposal exists")

        # Update round
        game_state["current_round"] += 1

        # Check if deadline reached - but allow extra rounds for final responses
        # Players should have a chance to accept/reject final proposals
        grace_rounds = 1  # Allow 1 extra round for final responses after proposal exhaustion
        max_total_rounds = self.max_rounds + grace_rounds
        
        if game_state["current_round"] > max_total_rounds:
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
            "game_ended": True,  # Explicitly mark game as ended
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

    def _create_no_agreement(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create no agreement result."""
        game_state.update({
            "agreement_reached": False,
            "game_ended": True,  # Explicitly mark game as ended
            "final_utilities": {
                self.it_team: 0,  # No deal utility
                self.marketing_team: 0
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
                game_state.get("game_ended", False) or
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
            print(f"🔍 [DEBUG] Looking for opponent proposal in {previous_round_key}")
            print(f"🔍 [DEBUG] Available round_proposals keys: {list(round_proposals.keys())}")
            if previous_round_key in round_proposals:
                previous_round_data = round_proposals[previous_round_key]
                print(f"🔍 [DEBUG] Previous round data: {previous_round_data}")
                for other_player, proposal_data in previous_round_data.items():
                    if other_player != player_id:  # It's from the opponent
                        current_proposal = proposal_data
                        print(f"🔍 [DEBUG] Found opponent proposal from {other_player}: {proposal_data}")
                        break
            else:
                print(f"🔍 [DEBUG] No proposals found for {previous_round_key}")
        else:
            print(f"🔍 [DEBUG] Round 1 - no previous proposals to check")
        
        base_prompt = f"""**Round {current_round}/{self.max_rounds} - Office Space Negotiation**
You are participating as the {role} team in this negotiation.
Your current BATNA (fallback option): {batna:.1f} points

"""

        # Add proposal context with strategic decision guidance
        if current_proposal:
            proposal = current_proposal['proposal']
            # Calculate utility for current proposal
            if role == "IT":
                weights = {"server_room": 0.4, "cleaning": 0.3, "branding": 0.2, "meeting_access": 0.1}
            else:  # Marketing
                weights = {"meeting_access": 0.4, "branding": 0.3, "cleaning": 0.2, "server_room": 0.1}
            
            # Use the actual game utility calculation method for consistency
            total_utility = self.calculate_utility(player_id, proposal)
            
            # Add strategic guidance for decision making
            guidance_hint = ""
            if total_utility > batna:
                utility_advantage = total_utility - batna
                if current_round > 2:  # After initial rounds
                    guidance_hint = f" Consider accepting this beneficial proposal (+{utility_advantage:.1f})."
                elif utility_advantage > 10:  # Significant advantage
                    guidance_hint = f" This proposal provides substantial value above your fallback option."
            else:
                utility_shortfall = batna - total_utility
                guidance_hint = f" Try proposing terms that better align with your priorities while being fair to the other team."
            
            # Provide informational context with subtle guidance
            print(f"🔍 [INFO] Player {player_id} ({role}): Evaluating proposal {proposal}")
            print(f"🔍 [INFO] Utility: {total_utility:.1f}, BATNA: {batna:.1f}")
            
            base_prompt += f"""**Other team's proposal:** {proposal}
Your utility from this proposal: {total_utility:.1f} points
Your BATNA (fallback): {batna:.1f} points{guidance_hint}

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

TASK: Respond with ONLY valid JSON. No explanations."""

            # Provide response format with strategic guidance for IT
            base_prompt += """
Valid response types:
- Accept: {{"type": "accept"}}
- Propose: {{"type": "propose", "proposal": {{"server_room": [50/100/150], "meeting_access": [2/4/7], "cleaning": "[IT/Shared/Outsourced]", "branding": "[Minimal/Moderate/Prominent]"}}}}

Negotiate strategically: Accept proposals above your BATNA, or counter with balanced offers that meet both teams' needs.

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

TASK: Respond with ONLY valid JSON. No explanations."""

            # Provide response format with strategic guidance for Marketing
            base_prompt += """
Valid response types:
- Accept: {{"type": "accept"}}
- Propose: {{"type": "propose", "proposal": {{"server_room": [50/100/150], "meeting_access": [2/4/7], "cleaning": "[IT/Shared/Outsourced]", "branding": "[Minimal/Moderate/Prominent]"}}}}

Negotiate strategically: Accept proposals above your BATNA, or counter with balanced offers that meet both teams' needs.

Your response:"""

        return base_prompt

    def get_game_prompt(self, player_id: str, game_state: Dict[str, Any] = None) -> str:
        """Generate structured prompt for integrative negotiation - NegotiationArena style."""
        current_state = game_state if game_state is not None else getattr(self, 'game_data', {})
        if not current_state:
            return "Game not initialized properly"

        private_info = current_state.get("private_info", {}).get(player_id, {})
        current_round = current_state.get("current_round", 1)
        role = private_info.get("role", "unknown")
        
        # Get current BATNA for this player and round
        batna = self.get_current_batna(player_id, current_round)
        
        # Check for opponent's proposal from previous round
        round_proposals = current_state.get("round_proposals", {})
        current_proposal = None
        other_offer = None
        
        if current_round > 1:
            previous_round_key = f"round_{current_round - 1}"
            if previous_round_key in round_proposals:
                previous_round_data = round_proposals[previous_round_key]
                for other_player, proposal_data in previous_round_data.items():
                    if other_player != player_id:
                        current_proposal = proposal_data["proposal"]
                        other_offer = current_proposal
                        break

        # Role-specific configuration
        if role == "IT":
            team_name = "IT Team"
            opponent_team = "Marketing Team"
            priorities = [
                "Server Room Size (40% weight): Need adequate space for technical infrastructure",
                "Cleaning Responsibility (30% weight): Prefer shared arrangements",
                "Branding Visibility (20% weight): Moderate visibility acceptable", 
                "Meeting Room Access (10% weight): Basic access sufficient"
            ]
            strategy_focus = "Focus on server room size and cleaning arrangements"
        else:  # Marketing
            team_name = "Marketing Team"
            opponent_team = "IT Team"
            priorities = [
                "Meeting Room Access (40% weight): Critical for client presentations",
                "Branding Visibility (30% weight): Important for company image",
                "Cleaning Responsibility (20% weight): Prefer outsourced cleaning",
                "Server Room Size (10% weight): Not a priority for marketing"
            ]
            strategy_focus = "Focus on meeting access and branding visibility"

        # Track proposals made by this player
        player_proposals = current_state.get(f"{player_id}_proposal_count", 0)
        max_proposals = self.max_rounds  # Use rounds from YAML config
        
        # Enhanced acceptance guidance
        acceptance_guidance = ""
        can_propose = player_proposals < max_proposals
        
        if other_offer is not None:
            proposal_utility = self.calculate_utility(player_id, other_offer)
            is_above_batna = proposal_utility > batna
            
            if is_above_batna:
                acceptance_guidance = (
                    f"🎯 STRATEGIC ANALYSIS: The opponent's proposal gives you {proposal_utility:.1f} points, "
                    f"which is ABOVE your BATNA ({batna:.1f}). This is a BENEFICIAL deal! "
                    f"Consider accepting to secure a positive outcome.\n"
                )
            else:
                if can_propose:
                    acceptance_guidance = (
                        f"⚠️ STRATEGIC ANALYSIS: The opponent's proposal gives you {proposal_utility:.1f} points, "
                        f"which is BELOW your BATNA ({batna:.1f}). You should negotiate for better terms.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: You've used all {max_proposals} proposals. The opponent's proposal is below your BATNA. "
                        f"You can ACCEPT (even if not ideal) or REJECT (which will END the negotiation).\n"
                    )

        # Proposal limit guidance
        proposal_guidance = ""
        if can_propose:
            proposal_guidance = f"You have {max_proposals - player_proposals} proposals remaining."
        else:
            proposal_guidance = f"⚠️ You have used all {max_proposals} proposals. You can only ACCEPT or REJECT now. Note: Rejecting will END the negotiation."

        # Build offer history
        offer_history = []
        if other_offer:
            offer_str = ", ".join([f"{k}: {v}" for k, v in other_offer.items()])
            offer_history.append(f"- Opponent's last proposal: {offer_str}")
        offer_status = "\n".join(offer_history) if offer_history else "No proposals made yet."

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

        prompt = f"""=== OFFICE SPACE NEGOTIATION ===
,{round_display} | Role: {role.upper()}

GOAL: Reach agreement that maximizes your team's utility
Your BATNA (Best Alternative): {batna:.1f} points
{strategy_focus}
{proposal_guidance}

YOUR PRIORITIES:
{chr(10).join([f"• {p}" for p in priorities])}

💡 NEGOTIATION STRATEGY:
• Your BATNA decreases each round - time pressure is real!
• Focus on YOUR high-priority issues (highest weights)
• Be flexible on low-priority issues to gain on high-priority ones
• Look for win-win solutions where you both get what matters most

AVAILABLE OPTIONS:
• Server Room Size: 50 sqm (10 pts), 100 sqm (30 pts), 150 sqm (60 pts)
• Meeting Access: 2 days/week (10 pts), 4 days/week (30 pts), 7 days/week (60 pts)  
• Cleaning: "IT" handles (30 pts), "Shared" (50 pts), "Outsourced" (10 pts)
• Branding: "Minimal" (10 pts), "Moderate" (30 pts), "Prominent" (60 pts)

CURRENT SITUATION:
{offer_status}
{acceptance_guidance}

RESPONSE FORMAT: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "accept"}}  // Accept the opponent's last proposal
{{"type": "propose", "proposal": {{"server_room": 100, "meeting_access": 4, "cleaning": "Shared", "branding": "Moderate"}}}}  // Make a new proposal
{{"type": "reject"}}  // Reject and end negotiation

EXAMPLE PROPOSAL:
{{"type": "propose", "proposal": {{"server_room": 150, "meeting_access": 2, "cleaning": "Shared", "branding": "Minimal"}}}}

Your response:"""
        
        return prompt
