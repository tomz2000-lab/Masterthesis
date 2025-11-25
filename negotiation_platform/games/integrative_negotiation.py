from typing import Dict, List, Any, Optional, Tuple
import random
import re
import json
from .base_game import BaseGame, PlayerAction
from .negotiation_tools import calculate_percentage_difference, calculate_utility


class IntegrativeNegotiationsGame(BaseGame):
    """
    Integrative negotiations game between IT and Marketing teams with price bargaining logic.

    Four issues with point values (unchanged from original):
    - Server Room Size: 50 sqm (10), 100 sqm (30), 150 sqm (60)
    - Meeting Room Access: 2 days/week (10), 4 days/week (30), 7 days/week (60)
    - Cleaning Responsibility: IT handles (10), Shared (30), Outsourced (60)
    - Branding Visibility: Minimal (10), Moderate (30), Prominent (60)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize integrative negotiations game with configuration parameters.
        
        Sets up multi-issue office space negotiation between IT and Marketing teams.
        Configures team preferences, BATNA values, issue structures, and decay rates
        based on provided configuration dictionary.
        
        Args:
            config (Dict[str, Any]): Game configuration containing:
                - batnas (Dict[str, float]): BATNA values for IT and Marketing teams
                - rounds (int): Maximum negotiation rounds allowed
                - batna_decay (float or Dict): Decay rate(s) for time pressure
                - issues (Dict, optional): Issue configurations and point values
                - weights (Dict, optional): Team preference weights by issue
        
        Raises:
            ValueError: If required configuration fields are missing.
        
        Example:
            >>> config = {
            ...     "batnas": {"IT": 35, "Marketing": 30},
            ...     "rounds": 8,
            ...     "batna_decay": 0.02
            ... }
            >>> game = IntegrativeNegotiationsGame(config)
        
        Note:
            Supports both hardcoded fallback values and flexible configuration
            for research customization and parameter sensitivity analysis.
        """
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

        # Load issue configurations from config instead of hardcoding
        if "issues" in config:
            self.issues = {}
            for issue_name, issue_config in config["issues"].items():
                self.issues[issue_name] = {
                    "options": issue_config["options"],
                    "points": issue_config["points"],
                    "labels": self._generate_labels(issue_name, issue_config["options"])
                }
        else:
            # Fallback to hardcoded values if not in config (for backward compatibility)
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
                    "points": [10, 30, 60],
                    "labels": ["IT handles", "Shared responsibility", "Outsourced"]
                },
                "branding": {
                    "options": ["Minimal", "Moderate", "Prominent"],
                    "points": [10, 30, 60],
                    "labels": ["Minimal visibility", "Moderate visibility", "Prominent visibility"]
                }
            }

        # Load preference weights from config instead of hardcoding
        if "weights" in config:
            self.weights = config["weights"]
        else:
            # Fallback to hardcoded values if not in config (for backward compatibility)
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
        #self.base_batnas = config.get("batnas", {"IT": 35, "Marketing": 30})

    def _generate_labels(self, issue_name: str, options: List[Any]) -> List[str]:
        """
        Generate human-readable descriptive labels for negotiation issue options.
        
        Creates contextual labels for each option within negotiation issues to
        improve readability in prompts and result reporting. Handles both standard
        issues and custom configurations with appropriate fallbacks.
        
        Args:
            issue_name (str): Name of the negotiation issue (server_room,
                meeting_access, cleaning, branding, or custom issue).
            options (List[Any]): List of available options for the issue.
        
        Returns:
            List[str]: Descriptive labels corresponding to each option.
                For server_room: ["50 sqm", "100 sqm", "150 sqm"]
                For meeting_access: ["2 days/week", "4 days/week", "7 days/week"]
                For cleaning: ["IT handles", "Shared responsibility", "Outsourced"]
                For branding: ["Minimal visibility", "Moderate visibility", "Prominent visibility"]
                For unknown issues: String representations of options
        
        Example:
            >>> game._generate_labels("server_room", [50, 100, 150])
            ["50 sqm", "100 sqm", "150 sqm"]
            >>> game._generate_labels("cleaning", ["IT", "Shared", "Outsourced"])
            ["IT handles", "Shared responsibility", "Outsourced"]
        
        Note:
            Supports extensibility for custom issues while maintaining backward
            compatibility with standard office space negotiation scenarios.
        """
        if issue_name == "server_room":
            return [f"{option} sqm" for option in options]
        elif issue_name == "meeting_access":
            return [f"{option} days/week" for option in options]
        elif issue_name == "cleaning":
            label_map = {
                "IT": "IT handles",
                "Shared": "Shared responsibility", 
                "Outsourced": "Outsourced"
            }
            return [label_map.get(str(option), str(option)) for option in options]
        elif issue_name == "branding":
            label_map = {
                "Minimal": "Minimal visibility",
                "Moderate": "Moderate visibility",
                "Prominent": "Prominent visibility"
            }
            return [label_map.get(str(option), str(option)) for option in options]
        else:
            # Generic labels for unknown issues
            return [str(option) for option in options]

    def validate_json_response(self, response: str) -> bool:
        """
        Validate that AI model response is properly formatted JSON with required structure.
        
        Performs structural validation of negotiation responses to ensure they
        contain the minimum required fields for processing. Used as a quick
        validation step before detailed parsing and action processing.
        
        Args:
            response (str): Raw response string from AI model to validate.
        
        Returns:
            bool: True if response is valid JSON dict with "type" field,
                False for malformed JSON or missing required structure.
        
        Validation Criteria:
            - Must be valid JSON syntax
            - Must parse to a dictionary object
            - Must contain "type" field for action identification
        
        Example:
            >>> game.validate_json_response('{"type": "propose", "proposal": {}}')
            True
            >>> game.validate_json_response('invalid json')
            False
            >>> game.validate_json_response('{"proposal": {}}')
            False
        
        Note:
            This is a lightweight validation step. Full semantic validation
            of proposals and action types occurs in subsequent processing steps.
        """
        try:
            data = json.loads(response.strip())
            return isinstance(data, dict) and "type" in data
        except (json.JSONDecodeError, TypeError):
            return False

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and extract decision data from AI model JSON responses with error recovery.
        
        Handles multiple response formats and provides robust parsing with graceful
        error recovery for malformed responses. Extracts decision data and preserves
        raw response for debugging purposes.
        
        Args:
            response (str): Raw JSON response string from AI model containing
                negotiation decision and potentially surrounding text.
        
        Returns:
            Dict[str, Any]: Parsed response containing:
                - decision (Dict): Extracted action data with type and parameters
                - raw_response (str): Original unmodified response for debugging
        
        Response Format Handling:
            - Pure JSON: {"type": "propose", "proposal": {...}}
            - Embedded JSON: Text containing JSON objects
            - Malformed JSON: Fallback to regex extraction and default rejection
        
        Error Recovery:
            - JSON parsing errors: Attempts regex extraction of key fields
            - Missing type field: Defaults to "reject" action
            - Invalid structure: Returns safe rejection response
        
        Example:
            >>> response = '{"type": "propose", "proposal": {"server_room": 150}}'
            >>> parsed = game.parse_json_response(response)
            >>> print(parsed["decision"]["type"])
            "propose"
            >>> print(parsed["decision"]["proposal"])
            {"server_room": 150}
        
        Note:
            Designed for robustness with AI model responses that may include
            explanatory text, formatting inconsistencies, or parsing errors.
        """
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
        """
        Initialize the integrative negotiation game with randomized role assignments.
        
        Sets up a multi-issue office space negotiation between IT and Marketing
        teams with randomized role assignment to minimize positional bias.
        Establishes complete game state including private information, utility
        functions, and BATNA parameters.
        
        Args:
            players (List[str]): List of exactly 2 player identifiers to
                participate in the bilateral negotiation.
        
        Returns:
            Dict[str, Any]: Complete initialized game state containing:
                - game_type (str): "integrative_negotiations"
                - players (List[str]): Player identifiers
                - role_assignments (Dict[str, str]): Mapping of roles to players
                - private_info (Dict[str, Dict]): Individual BATNAs and preferences
                - public_info (Dict[str, Any]): Shared issue descriptions
                - game_config (Dict[str, Any]): Configuration parameters
        
        Raises:
            ValueError: If number of players is not exactly 2.
        
        Example:
            >>> game = IntegrativeNegotiationsGame(config)
            >>> state = game.initialize_game(["alice", "bob"])
            >>> state["role_assignments"]
            {"IT": "alice", "Marketing": "bob"}
        
        Note:
            Role assignment is randomized to prevent systematic positional
            advantages. The first player is not always IT team.
        """
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
                    "batna": self.it_batna,
                    "preferences": "Prioritizes server room and cleaning costs"
                },
                self.marketing_team: {
                    "role": "Marketing", 
                    "batna": self.marketing_batna,
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
            "round_proposals": {},  # Track proposals by round to prevent overwriting
            "game_config": {
                "batna_decay": {
                    "IT": self.batna_decay if isinstance(self.batna_decay, float) else self.batna_decay.get("IT", 0.015),
                    "Marketing": self.batna_decay if isinstance(self.batna_decay, float) else self.batna_decay.get("Marketing", 0.015)
                },
                "issues": self.issues,
                "weights": self.weights
            }
        }
        
        return self.game_data

    def get_current_batna(self, player: str, round_num: int) -> float:
        """
        Calculate time-adjusted BATNA value accounting for negotiation urgency.
        
        Computes the Best Alternative to a Negotiated Agreement with exponential
        decay to model increasing time pressure and opportunity costs as rounds
        progress. Different decay rates can be applied per team role.
        
        Args:
            player (str): Player identifier to calculate BATNA for.
            round_num (int): Current round number (1-indexed) for time adjustment.
        
        Returns:
            float: Time-adjusted BATNA value for the specified player and round.
                Always less than or equal to the initial BATNA value.
        
        Formula:
            BATNA(t) = base_BATNA * (1 - decay_rate)^(round - 1)
        
        Example:
            >>> game.get_current_batna("alice", 1)  # Round 1
            85.0
            >>> game.get_current_batna("alice", 5)  # Round 5
            79.8  # Decreased due to time pressure
        
        Note:
            Supports both uniform decay rates (float) and role-specific
            decay rates (dict) for asymmetric time pressure modeling.
        """
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
        """
        Calculate total weighted utility for a player given a specific proposal.
        
        Computes utility by evaluating each negotiation issue against the
        player's preferences and applying role-specific weights. Uses the
        additive utility model where total utility is the sum of weighted
        issue utilities.
        
        Args:
            player (str): Player identifier to calculate utility for.
            proposal (Dict[str, Any]): Proposal dictionary containing selections
                for each negotiation issue (server_room, meeting_access, etc.).
        
        Returns:
            float: Total weighted utility value for the player. Higher values
                indicate more preferred outcomes.
        
        Utility Calculation:
            For each issue: utility += issue_points * role_weight
            Where issue_points are determined by option selection and
            role_weights reflect strategic importance to the player's team.
        
        Example:
            >>> proposal = {"server_room": 150, "meeting_access": 4,
            ...            "cleaning": "Shared", "branding": "Moderate"}
            >>> game.calculate_utility("alice", proposal)
            87.5
        
        Note:
            Returns 0.0 for invalid proposals or unrecognized players.
            Role assignment determines which weight set is applied.
        """
        # Determine player role based on exact team assignment
        if player == self.it_team:
            player_role = "IT"
        elif player == self.marketing_team:
            player_role = "Marketing"
        else:
            # Fallback: should not happen if game is properly initialized
            print(f"⚠️ Warning: Unknown player {player}, defaulting to IT role")
            player_role = "IT"
            
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
        """
        Validate that a proposal contains valid selections for all negotiation issues.
        
        Performs comprehensive validation to ensure proposals are complete and
        contain only valid options. Checks both completeness (all issues addressed)
        and validity (selections exist in defined option sets).
        
        Args:
            proposal (Dict[str, Any]): Proposal dictionary to validate containing
                issue names as keys and selected options as values.
        
        Returns:
            bool: True if proposal is complete and contains only valid selections,
                False if missing issues or invalid options detected.
        
        Validation Requirements:
            - Must be non-empty dictionary
            - Must contain all required issues (server_room, meeting_access, etc.)
            - All selections must exist in the corresponding issue option sets
        
        Example:
            >>> valid = {"server_room": 150, "meeting_access": 4,
            ...          "cleaning": "Shared", "branding": "Moderate"}
            >>> game.is_valid_proposal(valid)
            True
            >>> invalid = {"server_room": 200}  # Missing issues, invalid size
            >>> game.is_valid_proposal(invalid)
            False
        
        Note:
            This method validates structure and options but not semantic
            reasonableness or strategic value of proposals.
        """
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
        """
        Validate AI model response structure for required negotiation components.
        
        Performs comprehensive validation of parsed responses to ensure they
        contain all necessary fields and valid action types for multi-issue
        negotiation processing.
        
        Args:
            response (Dict[str, Any]): Parsed response dictionary to validate
                containing action type and associated parameters.
        
        Returns:
            bool: True if response contains valid action structure,
                False if missing required fields or invalid action types.
        
        Validation Rules:
            - Must contain "type" and "proposal" keys
            - Action type must be "propose", "accept", or "reject"
            - Proposal validation handled separately by is_valid_proposal()
        
        Example:
            >>> response = {"type": "propose", "proposal": {"server_room": 150}}
            >>> game.validate_response(response)
            True
            >>> invalid = {"type": "invalid_action"}
            >>> game.validate_response(invalid)
            False
        
        Note:
            This method validates response structure but not semantic validity
            of proposals. Content validation occurs in downstream processing.
        """
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

        #print(f"\n{'='*80}")
        #print(f"🔍 [PROCESS_ACTIONS] Round {current_round}: Processing actions")
        #print(f"🔍 [PROCESS_ACTIONS] Raw actions received: {actions}")
        #print(f"{'='*80}\n")

        # Initialize proposal counters if not present
        for player in [self.it_team, self.marketing_team]:
            if f"{player}_proposal_count" not in game_state:
                game_state[f"{player}_proposal_count"] = 0

        # Process JSON responses and extract decision data
        processed_actions = {}
        for player, raw_action in actions.items():
            #print(f"🔍 [{player}] Raw action type: {type(raw_action)}")
            #print(f"🔍 [{player}] Raw action: {raw_action}")
            
            if isinstance(raw_action, str):
                # If it's a string, try to parse it as JSON response
                parsed = self.parse_json_response(raw_action)
                action_data = parsed["decision"]
                #print(f"🔍 [{player}] Parsed from string, extracted decision: {action_data}")
            elif isinstance(raw_action, dict) and "decision" in raw_action:
                # Already structured
                action_data = raw_action["decision"]
                #print(f"🔍 [{player}] Dict with 'decision' key, extracted: {action_data}")
            else:
                # Regular action format
                action_data = raw_action
                #print(f"🔍 [{player}] Regular dict format: {action_data}")
            
            processed_actions[player] = action_data
            #print(f"🔍 [{player}] Processed action: {action_data}\n")

        # Normalize action types - treat counter/counter-offer as proposals, handle empty types
        normalized_actions = {}
        for player, action in processed_actions.items():
            action_type = action.get("type", "")
            #print(f"🔍 [{player}] Normalizing action type: '{action_type}'")
            
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
        
        #print(f"\n🔍 [NORMALIZATION COMPLETE] Normalized actions: {normalized_actions}\n")

        # Check for proposals and responses
        proposals = {player: action for player, action in normalized_actions.items()
                    if action.get("type") == "propose"}
        responses = {player: action for player, action in normalized_actions.items()
                    if action.get("type") in ["accept", "reject"]}
        
        #print(f"🔍 [CATEGORIZATION] Proposals: {proposals}")
        #print(f"🔍 [CATEGORIZATION] Responses: {responses}\n")

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
            
            #print(f"🔍 [{player}] Processing proposal (count: {player_proposals}/{max_proposals})")
            #print(f"🔍 [{player}] Action: {action}")
            
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} exceeded proposal limit ({player_proposals}/{max_proposals}). Ignoring additional proposals.")
                # Don't process this proposal, but don't end negotiation unless they also rejected
                continue
            
            # Valid proposal - process it
            proposal = action.get("proposal", {})
            #print(f"🔍 [{player}] Extracted proposal: {proposal}")
            
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
        """
        Create final game state when negotiation ends without agreement.
        
        Generates the terminal state for failed negotiations where players
        could not reach mutual agreement. Players receive their current
        BATNA values as final utilities, representing their fallback options.
        
        Args:
            game_state (Dict[str, Any]): Current game state to be finalized
                with no-agreement outcomes.
        
        Returns:
            Dict[str, Any]: Updated game state with no-agreement results including:
                - agreement_reached: False
                - reason: "no_agreement"
                - final_utilities: BATNA values for each player
                - winner: None (no winner in failed negotiations)
        
        Utility Assignment:
            Each player receives their time-adjusted BATNA value as utility,
            representing the value of their best alternative option.
        
        Example:
            >>> final_state = game._create_no_agreement(game_state)
            >>> final_state["agreement_reached"]
            False
            >>> final_state["final_utilities"]
            {"alice": 78.5, "bob": 82.1}  # BATNA values
        
        Note:
            BATNA values reflect time decay from negotiation duration.
            No winner is declared in failed negotiations.
        """
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
        """
        Create final game state when negotiation concludes with mutual agreement.
        
        Processes successful negotiations by calculating final utilities,
        determining the winner based on utility surplus over BATNA, and
        generating comprehensive agreement details for analysis.
        
        Args:
            final_proposal (Dict[str, Any]): The agreed-upon proposal containing
                selections for all negotiation issues.
            round_num (int): Round number when agreement was reached for
                time-adjusted BATNA calculations.
            game_state (Dict[str, Any]): Current game state to be finalized
                with agreement outcomes.
        
        Returns:
            Dict[str, Any]: Updated game state with agreement results including:
                - agreement_reached: True
                - agreement_round: Round of agreement
                - final_utilities: Calculated utilities for both players
                - utility_breakdown: Detailed utility analysis by issue
                - winner: Player with highest utility surplus over BATNA
        
        Winner Determination:
            Winner is the player with the largest positive difference between
            negotiated utility and their time-adjusted BATNA value.
        
        Example:
            >>> proposal = {"server_room": 150, "cleaning": "Shared"}
            >>> final_state = game._create_agreement(proposal, 3, game_state)
            >>> final_state["winner"]
            "alice"  # Highest utility surplus
        
        Note:
            Includes detailed debugging output for utility calculations
            and BATNA comparisons to support research analysis.
        """
        it_utility = self.calculate_utility(self.it_team, final_proposal)
        marketing_utility = self.calculate_utility(self.marketing_team, final_proposal)

        it_batna = self.get_current_batna(self.it_team, round_num)
        marketing_batna = self.get_current_batna(self.marketing_team, round_num)

        # DEBUG: Log the calculation like price bargaining
        print(f"🔍 [BATNA DEBUG] Round {round_num}: proposal={final_proposal}")
        print(f"🔍 [BATNA DEBUG] Config BATNAs: IT={self.it_batna}, Marketing={self.marketing_batna}")
        print(f"🔍 [BATNA DEBUG] Decay rate: {self.batna_decay}")
        print(f"🔍 [BATNA DEBUG] Calculated BATNAs: IT={it_batna:.2f}, Marketing={marketing_batna:.2f}")
        
        # DEBUG: Show detailed utility breakdown
        it_breakdown = self._calculate_utility_breakdown(self.it_team, final_proposal)
        marketing_breakdown = self._calculate_utility_breakdown(self.marketing_team, final_proposal)
        
        print(f"🔍 [UTILITY BREAKDOWN] IT Team:")
        for issue, data in it_breakdown.items():
            print(f"    {issue}: {data['selection']} = {data['raw_points']} * {data['weight']} = {data['weighted_utility']:.2f}")
        print(f"🔍 [UTILITY BREAKDOWN] Marketing Team:")
        for issue, data in marketing_breakdown.items():
            print(f"    {issue}: {data['selection']} = {data['raw_points']} * {data['weight']} = {data['weighted_utility']:.2f}")
            
        print(f"🔍 [UTILITY DEBUG] Utilities: IT={it_utility:.2f}, Marketing={marketing_utility:.2f}")
        print(f"🔍 [SURPLUS DEBUG] Surpluses: IT={it_utility - it_batna:.2f}, Marketing={marketing_utility - marketing_batna:.2f}")
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
        """
        Calculate detailed utility breakdown showing contribution of each issue.
        
        Provides granular analysis of utility calculation by decomposing
        total utility into per-issue contributions. Shows raw points,
        weights, and weighted utilities for transparency and debugging.
        
        Args:
            player (str): Player identifier to calculate breakdown for.
            proposal (Dict[str, Any]): Proposal containing issue selections
                to analyze for utility contributions.
        
        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary with issue names
                as keys and breakdown details as values:
                - selection: The chosen option for this issue
                - raw_points: Unweighted points for the selection
                - weight: Role-specific weight for this issue
                - weighted_utility: Final weighted contribution
        
        Breakdown Structure:
            Each issue provides: selection, raw_points, weight, weighted_utility
            enabling detailed analysis of negotiation outcomes.
        
        Example:
            >>> breakdown = game._calculate_utility_breakdown("alice", proposal)
            >>> breakdown["server_room"]
            {"selection": 150, "raw_points": 60, "weight": 0.4, "weighted_utility": 24.0}
        
        Note:
            Used primarily for detailed analysis and debugging of utility
            calculations in research contexts.
        """
        # Use exact same logic as calculate_utility method
        if player == self.it_team:
            player_role = "IT"
        elif player == self.marketing_team:
            player_role = "Marketing"
        else:
            # Fallback: should not happen if game is properly initialized
            print(f"⚠️ Warning: Unknown player {player}, defaulting to IT role")
            player_role = "IT"
            
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
        """
        Determine if the negotiation has reached a terminal state.
        
        Checks multiple termination conditions to determine if the game
        should end. Used by the game engine to control round progression
        and trigger final result calculations.
        
        Args:
            game_state (Dict[str, Any]): Current game state containing
                round information and agreement status.
        
        Returns:
            bool: True if game should terminate, False if negotiation
                should continue with additional rounds.
        
        Termination Conditions:
            - Agreement reached between players
            - Game explicitly marked as ended
            - Maximum rounds exceeded
        
        Example:
            >>> game.is_game_over({"agreement_reached": True})
            True
            >>> game.is_game_over({"current_round": 10})  # max_rounds=8
            True
            >>> game.is_game_over({"current_round": 3})
            False
        
        Note:
            Multiple termination conditions ensure robust game state
            management across different negotiation scenarios.
        """
        return (game_state.get("agreement_reached", False) or
                game_state.get("game_ended", False) or
                game_state.get("current_round", 1) > self.max_rounds)

    def get_winner(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        Determine negotiation winner based on utility surplus over BATNA.
        
        Identifies the player who achieved the greatest benefit from the
        negotiation by comparing their utility gain above their Best
        Alternative to a Negotiated Agreement (BATNA). No winner is
        declared for failed negotiations.
        
        Args:
            game_state (Dict[str, Any]): Final game state containing
                agreement details and utility calculations.
        
        Returns:
            Optional[str]: Player identifier of the winner, or None if:
                - No agreement was reached
                - Both players have equal utility surplus
        
        Winner Criteria:
            Winner = max(utility - time_adjusted_BATNA) for each player
            Only positive surpluses indicate successful negotiation outcomes.
        
        Example:
            >>> # Player utilities: alice=85, bob=78; BATNAs: alice=75, bob=80
            >>> game.get_winner(final_state)
            "alice"  # Surplus: alice=10, bob=-2
            >>> game.get_winner(no_agreement_state)
            None  # No agreement reached
        
        Note:
            Winner determination encourages value-creating negotiations
            rather than purely competitive zero-sum outcomes.
        """
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

        # Only consider players with positive surplus
        positive_surplus_players = {player: surplus for player, surplus in surpluses.items() if surplus > 0}
        
        if not positive_surplus_players:
            # No player has positive surplus - no winner
            return None
        elif len(positive_surplus_players) == 1:
            # Only one player has positive surplus - they win
            return list(positive_surplus_players.keys())[0]
        else:
            # Multiple players with positive surplus - highest surplus wins
            return max(positive_surplus_players, key=positive_surplus_players.get)

    def get_game_summary(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive summary of negotiation results for analysis.
        
        Creates a structured summary containing all key negotiation outcomes,
        player roles, agreement details, and utility calculations. Used for
        research analysis, reporting, and comparative studies.
        
        Args:
            game_state (Dict[str, Any]): Final game state containing complete
                negotiation history and outcomes.
        
        Returns:
            Dict[str, Any]: Comprehensive summary containing:
                - game_type: "Integrative Negotiations"
                - players: Mapping of player IDs to role names
                - agreement_reached: Boolean success indicator
                - agreement_details: Final proposal if successful
                - utilities: Final utility values for each player
                - failure_reason: Explanation if negotiation failed
        
        Summary Structure:
            Successful negotiations include agreement round, final proposal,
            utilities, and detailed breakdowns. Failed negotiations include
            failure reasons and context.
        
        Example:
            >>> summary = game.get_game_summary(final_state)
            >>> summary["agreement_reached"]
            True
            >>> summary["final_agreement"]
            {"server_room": 150, "cleaning": "Shared"}
        
        Note:
            Provides standardized output format for research data collection
            and comparative analysis across different negotiation scenarios.
        """
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
        """
        Process a single player action (required by BaseGame interface).
        
        Implements the abstract BaseGame method for single-action processing.
        In integrative negotiations, multi-player action processing via
        process_actions() is preferred. This method serves as a compatibility
        interface for the base class contract.
        
        Args:
            action: Single player action to process (format varies).
        
        Returns:
            Dict[str, Any]: Processing result indicating successful handling.
        
        Implementation Note:
            This game uses simultaneous bilateral action processing through
            process_actions() rather than sequential single-action processing.
            This method provides base class compatibility.
        
        Example:
            >>> result = game.process_action({"type": "propose"})
            >>> result["processed"]
            True
        
        Note:
            For actual negotiation processing, use process_actions() which
            handles simultaneous bilateral actions appropriately.
        """
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
        """
        Check if the game should end (required by BaseGame interface).
        
        Implements the abstract BaseGame method for termination checking.
        Delegates to the state-based is_game_over() method when game data
        is available, providing base class compatibility.
        
        Returns:
            bool: True if game should terminate, False otherwise.
                Delegates to is_game_over() when game state exists.
        
        Implementation Note:
            This game uses state-based termination checking through
            is_game_over() which analyzes current game state for
            termination conditions.
        
        Example:
            >>> game.check_end_conditions()
            True  # If agreement reached or rounds exceeded
        
        Note:
            Primary termination logic resides in is_game_over() which
            requires game state for proper condition evaluation.
        """
        if hasattr(self, 'game_data'):
            return self.is_game_over(self.game_data)
        return False
    
    def calculate_scores(self) -> Dict[str, float]:
        """
        Calculate final scores for all players (required by BaseGame interface).
        
        Implements the abstract BaseGame method for score calculation.
        Returns final utilities from completed negotiations or zero scores
        for failed negotiations, providing base class compatibility.
        
        Returns:
            Dict[str, float]: Dictionary mapping player IDs to final scores:
                - Successful negotiations: actual utility values
                - Failed negotiations: 0.0 for all players
                - No game data: 0.0 for all players
        
        Score Calculation:
            Scores are the final utility values calculated during agreement
            creation, representing negotiated value for each player.
        
        Example:
            >>> scores = game.calculate_scores()
            >>> scores
            {"alice": 87.5, "bob": 78.3}  # After successful negotiation
        
        Note:
            Scores represent utility values rather than competitive rankings.
            Both players can achieve positive scores in value-creating negotiations.
        """
        if hasattr(self, 'game_data'):
            if self.game_data.get("agreement_reached", False):
                return self.game_data.get("final_utilities", {})
        return {player: 0.0 for player in getattr(self, 'players', [])}
    
    def _get_neutral_role_label(self, player_id: str) -> str:
        """
        Map player identifier to neutral role label to minimize cognitive bias.
        
        Provides neutral terminology ("ROLE A"/"ROLE B") instead of loaded
        domain-specific terms ("IT"/"Marketing") to reduce behavioral biases
        and role-based assumptions in negotiation prompts and analysis.
        
        Args:
            player_id (str): Player identifier to map to neutral label.
        
        Returns:
            str: Neutral role label ("ROLE A" for IT team, "ROLE B" for Marketing team).
        
        Bias Reduction:
            - Eliminates domain-specific role assumptions
            - Reduces stereotype-based behavioral influences
            - Enables more objective negotiation analysis
            - Supports fair comparison across different scenarios
        
        Example:
            >>> # If player_123 is assigned as IT team
            >>> game._get_neutral_role_label("player_123")
            "ROLE A"
            >>> # If player_456 is assigned as Marketing team  
            >>> game._get_neutral_role_label("player_456")
            "ROLE B"
        
        Note:
            Used primarily in prompt generation and result reporting to maintain
            experimental validity and reduce confounding variables in analysis.
        """
        if player_id == self.it_team:
            return "ROLE A"
        else:
            return "ROLE B"
    
    def get_game_prompt(self, player_id: str, game_state: Dict[str, Any] = None) -> str:
        """
        Generate structured negotiation prompt for AI model interaction.
        
        Creates comprehensive, role-specific prompts that provide players with
        all necessary context for strategic decision-making. Includes current
        situation, available options, utility guidance, and proper JSON
        response formatting requirements.
        
        Args:
            player_id (str): Identifier of the player to generate prompt for.
            game_state (Dict[str, Any], optional): Current game state to use
                for prompt generation. If None, uses internal game_data.
        
        Returns:
            str: Complete formatted prompt string containing:
                - Current round and role information
                - Available options and point values
                - Role-specific priorities and preferences
                - Proposal history and opponent actions
                - Response format requirements and examples
        
        Prompt Components:
            - Header with round/role identification
            - BATNA and remaining proposal information
            - Option descriptions with utility values
            - Strategic guidance based on current situation
            - JSON format requirements and examples
        
        Example:
            >>> prompt = game.get_game_prompt("alice", game_state)
            >>> "OFFICE SPACE NEGOTIATION" in prompt
            True
            >>> "RESPONSE FORMAT" in prompt
            True
        
        Note:
            Prompts use neutral role labels to minimize cognitive bias
            while providing complete strategic context for informed decisions.
        """
        current_state = game_state if game_state is not None else getattr(self, 'game_data', {})
        if not current_state:
            return "Game not initialized properly"

        private_info = current_state.get("private_info", {}).get(player_id, {})
        current_round = current_state.get("current_round", 1)
        role = private_info.get("role", "unknown")
        neutral_role = self._get_neutral_role_label(player_id)
        
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
                f"Server Room Size (40% weight): Prefer 150 sqm of room size",
                f"Cleaning Responsibility (30% weight): Prefer shared arrangements",
                f"Branding Visibility (20% weight): Moderate visibility acceptable", 
                f"Meeting Room Access (10% weight): 2 days access sufficient",
                f"Note: Server room size and cleaning are top priorities"
            )

        else:  # Marketing
            role_priorities = (
                f"Meeting Room Access (40% weight): Prefer 7 days access to meeting room",
                f"Branding Visibility (30% weight): Prefer Prominent branding",
                f"Cleaning Responsibility (20% weight): Prefer IT to handle cleaning",
                f"Server Room Size (10% weight): 50 sqm are sufficient",
                f"Note: Meeting access and branding are top priorities"
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
{round_display} | Role: {neutral_role}

GOAL: Reach agreement that maximizes your utility
Your BATNA (Best Alternative): {batna:.1f} points
{proposal_guidance}

YOUR OPTIONS:
- **Server Room Size:** 50 sqm (10 pts), 100 sqm (30 pts), or 150 sqm (60 pts)
- **Meeting Room Access:** 2 days/week (10 pts), 4 days/week (30 pts), or 7 days/week (60 pts)
- **Cleaning Responsibility:** "IT" (10 pts), "Shared" (30 pts), or "Outsourced" (60 pts)
- **Branding Visibility:** "Minimal" (10 pts), "Moderate" (30 pts), or "Prominent" (60 pts)

YOUR PRIORITIES:
{role_priorities}

CURRENT SITUATION:
{offer_status}
{acceptance_guidance}

RESPONSE FORMAT: Respond with ONLY valid JSON. No explanations.
Valid responses:

{{"type": "accept"}}  // Accept the opponent's last offer
{{"type": "propose", "server_room": 150, "meeting_access": 2, "cleaning": "Shared", "branding": "Minimal"}} // // Propose new allocation
{{"type": "reject"}}  // Reject and end negotiation

EXAMPLE OFFERS:
{{"type": "propose", "server_room": 150, "meeting_access": 2, "cleaning": "Shared", "branding": "Minimal"}}

Do NOT repeat any of the rules or instructions in your response. Focus on negotiation.

Your response:"""
        
        return prompt
