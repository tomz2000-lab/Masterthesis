from typing import Dict, List, Any, Optional, Tuple
import random
import re
import json

from .base_game import BaseGame, PlayerAction
from .negotiation_tools import calculate_percentage_difference, calculate_utility


class IntegrativeNegotiationsGame(BaseGame):
    """
    Integrative negotiations game between IT and Marketing teams with price bargaining logic.
    
    ADAPTED FROM ORIGINAL: Uses the same content (4 issues, point values, weights) but 
    follows the price bargaining game patterns for proposals, offers, and game flow:
    - Same max_rounds structure as price bargaining (5 rounds)
    - Same proposal limits (max_rounds - 1)
    - Same role assignment randomization logic
    - Same JSON parsing and action processing patterns
    - Same agreement/no-agreement creation patterns
    - Same debug logging patterns

    Four issues with point values (unchanged from original):
    - Server Room Size: 50 sqm (10), 100 sqm (30), 150 sqm (60)
    - Meeting Room Access: 2 days/week (10), 4 days/week (30), 7 days/week (60)
    - Cleaning Responsibility: IT handles (30), Shared (50), Outsourced (10)
    - Branding Visibility: Minimal (10), Moderate (30), Prominent (60)
    """

    def __init__(self, config: Dict[str, Any]):
        # Initialize base class with game type as game_id  
        super().__init__(game_id="integrative_negotiations", config=config)
        required_fields = [
            "batnas", "rounds", "batna_decay"
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Extract BATNAs from the batnas dictionary
        batnas = config["batnas"]
        self.marketing_batna = batnas["Marketing"]
        self.it_batna = batnas["IT"]
        self.max_rounds = config["rounds"]
        self.batna_decay = config["batna_decay"]

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

        # Preference weights from original document specifications
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

        # Base BATNA values - keeping original approach but with price bargaining decay
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
        
        # Randomize role assignment like price bargaining - Fix for positional bias
        if random.choice([True, False]):
            self.it_team = players[0]
            self.marketing_team = players[1]
            print(f"🎲 [ROLE ASSIGNMENT] {players[0]} = IT, {players[1]} = MARKETING")
        else:
            self.it_team = players[1] 
            self.marketing_team = players[0]
            print(f"🎲 [ROLE ASSIGNMENT] {players[1]} = IT, {players[0]} = MARKETING")

        self.game_data = {
            "game_type": "integrative_negotiations",
            "players": self.players,
            "rounds": self.max_rounds,
            "current_round": 1,
            "role_assignments": {
                "IT": self.it_team,
                "Marketing": self.marketing_team
            },
            "private_info": {
                self.it_team: {
                    "role": "IT",
                    "batnas": self.base_batnas["IT"],
                    "preferences": "Prioritizes server room and cleaning costs"
                },
                self.marketing_team: {
                    "role": "Marketing", 
                    "batnas": self.base_batnas["Marketing"],
                    "preferences": "Prioritizes meeting access and branding costs"
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
        if player == self.marketing_team:
            # Handle both dict and float formats for batna_decay
            if isinstance(self.batna_decay, dict):
                decay_rate = self.batna_decay.get("Marketing", self.batna_decay.get("marketing", 0.0))
            else:
                decay_rate = self.batna_decay
            base_batna = self.marketing_batna
        else:
            # Handle both dict and float formats for batna_decay
            if isinstance(self.batna_decay, dict):
                decay_rate = self.batna_decay.get("IT", self.batna_decay.get("it", 0.0))
            else:
                decay_rate = self.batna_decay
            base_batna = self.it_batna

        return base_batna * ((1 - decay_rate) ** (round_num - 1))

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
            
        max_proposals = self.max_rounds - 1  # Like price bargaining
        player_proposals = game_state.get(f"{player}_proposal_count", 0)

        if action_type == "propose":
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} tried to make proposal but exceeded limit ({player_proposals}/{max_proposals})")
                return False
            
            # Handle both nested and flat proposal formats
            if "proposal" in action_data:
                # Nested format: {"type": "propose", "proposal": {...}}
                proposal = action_data["proposal"]
            else:
                # Flat format from Pydantic: {"type": "propose", "server_room": 150, ...}
                proposal = {}
                for field in ["server_room", "meeting_access", "cleaning", "branding"]:
                    if field in action_data:
                        proposal[field] = action_data[field]
                
                if not proposal:
                    print(f"⚠️ Player {player} made propose action but no proposal fields found in {action_data}")
                    return False
            
            return self.is_valid_proposal(proposal)

        elif action_type in ["accept", "reject"]:
            return True

        elif action_type in ["counter", "counter-offer"]:
            # Check proposal limit for counters too
            if player_proposals >= max_proposals:
                return False
            
            # Handle both nested and flat proposal formats
            if "proposal" in action_data:
                # Nested format
                proposal = action_data["proposal"]
            else:
                # Flat format from Pydantic
                proposal = {}
                for field in ["server_room", "meeting_access", "cleaning", "branding"]:
                    if field in action_data:
                        proposal[field] = action_data[field]
                
                if not proposal:
                    return False
            
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
        max_proposals = self.max_rounds - 1  # Like price bargaining

        print(f"\n{'='*80}")
        print(f"🔍 [PROCESS_ACTIONS] Round {current_round}: Processing actions")
        print(f"🔍 [PROCESS_ACTIONS] Raw actions received: {actions}")
        print(f"{'='*80}\n")

        # Initialize proposal counters if not present
        for player in [self.it_team, self.marketing_team]:
            if f"{player}_proposal_count" not in game_state:
                game_state[f"{player}_proposal_count"] = 0

        # Process JSON responses and extract decision data
        processed_actions = {}
        for player, raw_action in actions.items():
            print(f"🔍 [{player}] Raw action type: {type(raw_action)}")
            print(f"🔍 [{player}] Raw action: {raw_action}")
            
            if isinstance(raw_action, str):
                # If it's a string, try to parse it as JSON response
                parsed = self.parse_json_response(raw_action)
                action_data = parsed["decision"]
                print(f"🔍 [{player}] Parsed from string, extracted decision: {action_data}")
            elif isinstance(raw_action, dict) and "decision" in raw_action:
                # Already structured
                action_data = raw_action["decision"]
                print(f"🔍 [{player}] Dict with 'decision' key, extracted: {action_data}")
            else:
                # Regular action format
                action_data = raw_action
                print(f"🔍 [{player}] Regular dict format: {action_data}")
            
            processed_actions[player] = action_data
            print(f"🔍 [{player}] Processed action: {action_data}\n")

        # Normalize action types - treat counter/counter-offer as proposals, handle empty types
        normalized_actions = {}
        for player, action in processed_actions.items():
            action_type = action.get("type", "")
            print(f"🔍 [{player}] Normalizing action type: '{action_type}'")
            
            # Handle empty or invalid action types
            if not action_type or action_type == "":
                print(f"⚠️ Player {player} provided empty action type, treating as reject")
                normalized_actions[player] = {"type": "reject"}
            elif action_type in ["counter", "counter-offer"]:
                # Convert to propose
                normalized_actions[player] = {"type": "propose", "proposal": action.get("proposal", {})}
                print(f"🔧 [{player}] Converted counter to propose: {normalized_actions[player]}")
            elif action_type == "propose":
                # Handle both nested and flat proposal formats
                if "proposal" in action:
                    # Already in nested format
                    normalized_actions[player] = action
                    print(f"✅ [{player}] Already in nested format: {action}")
                else:
                    # Flat format - extract proposal fields and nest them
                    proposal_fields = {}
                    for field in ["server_room", "meeting_access", "cleaning", "branding"]:
                        if field in action:
                            proposal_fields[field] = action[field]
                    
                    if proposal_fields:
                        print(f"🔧 [FORMAT FIX] Converting flat proposal to nested format for {player}: {proposal_fields}")
                        normalized_actions[player] = {"type": "propose", "proposal": proposal_fields}
                    else:
                        print(f"⚠️ Player {player} made propose action but no proposal fields found")
                        print(f"⚠️ Action keys: {list(action.keys())}")
                        normalized_actions[player] = {"type": "reject"}
            else:
                normalized_actions[player] = action
                print(f"✅ [{player}] Action accepted as-is: {action}")
        
        print(f"\n🔍 [NORMALIZATION COMPLETE] Normalized actions: {normalized_actions}\n")

        # Check for proposals and responses
        proposals = {player: action for player, action in normalized_actions.items()
                    if action.get("type") == "propose"}
        responses = {player: action for player, action in normalized_actions.items()
                    if action.get("type") in ["accept", "reject"]}
        
        print(f"🔍 [CATEGORIZATION] Proposals: {proposals}")
        print(f"🔍 [CATEGORIZATION] Responses: {responses}\n")

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
            
            print(f"🔍 [{player}] Processing proposal (count: {player_proposals}/{max_proposals})")
            print(f"🔍 [{player}] Action: {action}")
            
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} exceeded proposal limit ({player_proposals}/{max_proposals}). Ignoring additional proposals.")
                # Don't process this proposal, but don't end negotiation unless they also rejected
                continue
            
            # Valid proposal - process it
            proposal = action.get("proposal", {})
            print(f"🔍 [{player}] Extracted proposal: {proposal}")
            
            if self.is_valid_proposal(proposal):
                game_state[f"{player}_last_proposal"] = proposal
                game_state[f"{player}_last_proposal_round"] = current_round  # Track when proposal was made
                game_state[f"{player}_proposal_count"] = player_proposals + 1
                print(f"💡 Player {player} made proposal (#{player_proposals + 1}/{max_proposals}): {proposal}")
                print(f"✅ [{player}] Stored in game_state as '{player}_last_proposal'")
                
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

        # Check for convergence: if both players made identical proposals, create agreement
        if len(proposals) == 2:  # Both players made proposals this round
            proposal_dicts = [action.get("proposal", {}) for action in proposals.values()]
            if len(proposal_dicts) == 2 and proposal_dicts[0] == proposal_dicts[1] and proposal_dicts[0]:
                agreed_proposal = proposal_dicts[0]
                print(f"🎉 CONVERGENCE! Both players proposed identical terms - Creating automatic agreement!")
                return self._create_agreement(agreed_proposal, current_round, game_state)

        # Process acceptances (rejections already handled above)
        for player, action in responses.items():
            if action.get("type") == "accept":
                # Find the proposal being accepted (from the other player)
                other_player = self.marketing_team if player == self.it_team else self.it_team
                print(f"🔍 [{player}] Trying to accept proposal from {other_player}")
                print(f"🔍 [{player}] Looking for key: '{other_player}_last_proposal' in game_state")
                print(f"🔍 [{player}] Game state keys: {list(game_state.keys())}")
                
                if f"{other_player}_last_proposal" in game_state:
                    accepted_proposal = game_state[f"{other_player}_last_proposal"]
                    # Get the round when the accepted proposal was made
                    proposal_round = game_state.get(f"{other_player}_last_proposal_round", current_round)
                    print(f"✅ Player {player} accepted proposal: {accepted_proposal} (made in round {proposal_round})")
                    return self._create_agreement(accepted_proposal, proposal_round, game_state)
                else:
                    print(f"⚠️ Player {player} tried to accept but no proposal exists from {other_player}")
                    print(f"⚠️ Available game_state keys: {[k for k in game_state.keys() if 'proposal' in k.lower()]}")

        # Update round
        game_state["current_round"] += 1

        # Check if deadline reached - like price bargaining
        if game_state["current_round"] > self.max_rounds:
            return self._create_no_agreement(game_state)

        return game_state

    def _create_no_agreement(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create no agreement result with BATNA values."""
        print(f"🎲 [ROLE DEBUG] No Agreement - IT={self.it_team}, Marketing={self.marketing_team}")
        print(f"🎲 [ROLE DEBUG] {self.it_team} utility=0, {self.marketing_team} utility=0")
        
        game_state.update({
            "agreement_reached": False,
            "game_ended": True,  # Explicitly mark game as ended
            "role_assignments": {
                "IT": self.it_team,
                "Marketing": self.marketing_team
            },
            "final_utilities": {
                self.it_team: 0,  # No deal utility like price bargaining
                self.marketing_team: 0
            },
            "termination_reason": "deadline_reached",
            "final_round": game_state["current_round"]
        })
        
        return game_state

    def _create_agreement(self, final_proposal: Dict[str, Any], round_num: int, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create agreement result with utilities."""
        it_utility = self.calculate_utility(self.it_team, final_proposal)
        marketing_utility = self.calculate_utility(self.marketing_team, final_proposal)

        it_batna = self.get_current_batna(self.it_team, round_num)
        marketing_batna = self.get_current_batna(self.marketing_team, round_num)

        # DEBUG: Log the calculation like price bargaining
        print(f"🔍 [BATNA DEBUG] Round {round_num}: proposal={final_proposal}")
        print(f"🔍 [BATNA DEBUG] Config BATNAs: IT={self.base_batnas['IT']}, Marketing={self.base_batnas['Marketing']}")
        print(f"🔍 [BATNA DEBUG] Decay rate: {self.batna_decay}")
        print(f"🔍 [BATNA DEBUG] Calculated BATNAs: IT={it_batna:.2f}, Marketing={marketing_batna:.2f}")
        print(f"🔍 [BATNA DEBUG] Utilities: IT={it_utility:.2f}, Marketing={marketing_utility:.2f}")
        print(f"🎲 [ROLE DEBUG] IT={self.it_team}, Marketing={self.marketing_team}")
        print(f"🎲 [ROLE DEBUG] {self.it_team} utility={it_utility:.2f}, {self.marketing_team} utility={marketing_utility:.2f}")

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
            "role_assignments": {
                "IT": self.it_team,
                "Marketing": self.marketing_team
            },
            "final_utilities": {
                self.it_team: it_utility,
                self.marketing_team: marketing_utility
            },
            "batnas_at_agreement": {
                self.it_team: it_batna,
                self.marketing_team: marketing_batna
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
        
        # Check for opponent's proposal - follow price bargaining pattern
        other_player = self.marketing_team if player_id == self.it_team else self.it_team
        other_offer = current_state.get(f"{other_player}_last_proposal", None)
        my_proposal = current_state.get(f"{player_id}_last_proposal", None)

        # Role-specific configuration
        role_priorities = ""
        if role == "IT":
            role_priorities = (
                f"Server Room Size (40% weight): Need adequate space for technical infrastructure",
                f"Cleaning Responsibility (30% weight): Prefer shared arrangements",
                f"Branding Visibility (20% weight): Moderate visibility acceptable", 
                f"Meeting Room Access (10% weight): Basic access sufficient",
                f"Note: Server room size and cleaning are top priorities for IT"
            )

        else:  # Marketing
            role_priorities = (
                f"Meeting Room Access (40% weight): Critical for client presentations",
                f"Branding Visibility (30% weight): Important for company image",
                f"Cleaning Responsibility (20% weight): Prefer outsourced cleaning",
                f"Server Room Size (10% weight): Not a priority for marketing",
                f"Note: Meeting access and branding are top priorities for Marketing"
            )

        # Track proposals made by this player
        player_proposals = current_state.get(f"{player_id}_proposal_count", 0)
        max_proposals = self.max_rounds - 1  # Use rounds from YAML config like price bargaining
        
        # Enhanced acceptance guidance
        acceptance_guidance = ""
        can_propose = player_proposals < max_proposals
        rounds_remaining = max_proposals - player_proposals
        
        if other_offer is not None:
            proposal_utility = self.calculate_utility(player_id, other_offer)
            is_above_batna = proposal_utility > batna
            
            if is_above_batna:
                if rounds_remaining == 0:  # No proposals left - encourage acceptance
                    acceptance_guidance = (
                        f"🎯 FINAL ANALYSIS: The opponent's proposal gives you {proposal_utility:.1f} points, "
                        f"which is ABOVE your BATNA ({batna:.1f}). You have no proposals left - ACCEPT to secure this beneficial deal!\n"
                    )
                elif rounds_remaining == 1:  # Last proposal - be more encouraging
                    acceptance_guidance = (
                        f"🎯 ANALYSIS: The opponent's proposal gives you {proposal_utility:.1f} points, "
                        f"which is ABOVE your BATNA ({batna:.1f}). With only 1 proposal left, consider accepting or making the last counter proposal.\n"
                    )
                else:  # Multiple proposals left - encourage exploration
                    acceptance_guidance = (
                        f"💡 ANALYSIS: The opponent's proposal gives you {proposal_utility:.1f} points, "
                        f"which is ABOVE your BATNA ({batna:.1f}), but you have {rounds_remaining} proposals left. You might negotiate for an even better deal.\n"
                    )
            else:
                gap = abs(proposal_utility - batna)
                if rounds_remaining == 0:  # No proposals left - suggest accepting to avoid no-deal
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: The opponent's proposal gives you {proposal_utility:.1f} points, "
                        f"which is {gap:.1f} points below your BATNA ({batna:.1f}). You have no proposals left. "
                        f"ACCEPT to avoid no-deal or REJECT this proposal.\n"
                    )
                elif rounds_remaining == 1:  # Last proposal - be more encouraging
                    acceptance_guidance = (
                        f"🎯 ANALYSIS: The opponent's proposal gives you {proposal_utility:.1f} points, "
                        f"which is {gap:.1f} points below your BATNA ({batna:.1f}). With only 1 proposal left, consider accepting or making the last counter proposal.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"⚠️ ANALYSIS: The opponent's proposal gives you {proposal_utility:.1f} points, "
                        f"which is {gap:.1f} points below your BATNA ({batna:.1f}). You should negotiate for a better deal.\n"
                    )

        # Build offer history like price bargaining
        offer_history = []
        if my_proposal:
            proposal_str = ", ".join([f"{k}: {v}" for k, v in my_proposal.items()])
            offer_history.append(f"- Your last proposal: {proposal_str}")
        if other_offer:
            offer_str = ", ".join([f"{k}: {v}" for k, v in other_offer.items()])
            offer_history.append(f"- Opponent's last proposal: {offer_str}")
        offer_status = "\n".join(offer_history) if offer_history else "No proposals made yet."

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

        prompt = f"""=== OFFICE SPACE NEGOTIATION ===
{round_display} | Role: {role.upper()}

GOAL: Reach agreement that maximizes your team's utility
Your BATNA (Best Alternative): {batna:.1f} points
{proposal_guidance}

YOUR OPTIONS:
- **Server Room Size:** 50 sqm (10 pts), 100 sqm (30 pts), or 150 sqm (60 pts)
- **Meeting Room Access:** 2 days/week (10 pts), 4 days/week (30 pts), or 7 days/week (60 pts)
- **Cleaning Responsibility:** "IT" (30 pts), "Shared" (50 pts), or "Outsourced" (10 pts)
- **Branding Visibility:** "Minimal" (10 pts), "Moderate" (30 pts), or "Prominent" (60 pts)

YOUR PRIORITIES:
{role_priorities}

CURRENT SITUATION:
{offer_status}
{acceptance_guidance}

RESPONSE FORMAT: Respond with ONLY a single valid JSON object. Choose ONE of these three formats:

1. To accept the opponent's proposal:
{{"type": "accept"}}

2. To make a new proposal:
{{"type": "propose", "server_room": 150, "meeting_access": 2, "cleaning": "Shared", "branding": "Minimal"}}

3. To reject and end negotiation:
{{"type": "reject"}}

IMPORTANT: 
- Use exact values from YOUR OPTIONS above
- Respond with ONLY the JSON, no explanations or additional text
- For propose actions, include all four fields: server_room, meeting_access, cleaning, branding

Your response:"""
        
        return prompt

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
        return {
            "game_type": "integrative_negotiations",
            "agreement_reached": game_state.get("agreement_reached", False),
            "final_utilities": game_state.get("final_utilities", {}),
            "agreement_round": game_state.get("agreement_round"),
            "final_proposal": game_state.get("final_proposal", {}),
            "role_assignments": game_state.get("role_assignments", {}),
            "total_rounds": game_state.get("current_round", 1) - 1
        }
