"""
Resource Allocation Negotiation Game
====================================

Multi-resource distribution negotiation between development and marketing teams.

This module implements a complex resource allocation scenario where two teams 
(Development and Marketing) negotiate the distribution of limited resources 
including GPUs, developers, and budget allocation for a project.

The game simulates realistic organizational resource conflicts where:
- Teams have different priorities and valuation of resources
- Resources are limited and must be allocated efficiently
- Teams must balance their needs against organizational constraints
- Win-win solutions require creative resource sharing and trade-offs

Key Features:
    - Multi-resource negotiation (GPUs, developers, budget)
    - Team-based roles with different resource priorities
    - Complex utility calculations based on resource combinations
    - BATNA values representing alternative resource sources
    - Structured JSON proposal system for resource requests

Game Components:
    - Development Team: Focuses on GPU resources and technical staff
    - Marketing Team: Prioritizes budget and developer support
    - Resource Pool: Limited resources that must be allocated
    - Utility Functions: Team-specific valuation of resource combinations

Example:
    >>> config = {
    ...     "max_rounds": 5,
    ...     "total_gpus": 10,
    ...     "total_developers": 8,
    ...     "total_budget": 100000
    ... }
    >>> game = ResourceAllocationGame(config)
    >>> game.initialize_game(["dev_team", "marketing_team"])
"""

import random
import re
import json
import sys
from typing import Dict, List, Any, Optional
from .base_game import BaseGame, PlayerAction


class ResourceAllocationGame(BaseGame):
    """
    Complex multi-resource allocation negotiation between Development and Marketing teams.
    
    This class implements a sophisticated resource distribution scenario where two
    organizational teams must negotiate the allocation of limited computational and
    human resources for a shared project. The game simulates realistic organizational
    resource conflicts with complex interdependencies and trade-offs.
    
    The negotiation involves multiple resource types with different valuations for
    each team, requiring creative resource sharing and package deals to achieve
    mutually beneficial outcomes. Teams must balance their specific needs against
    organizational constraints and counterpart requirements.
    
    Key Features:
        - Multi-resource negotiation (GPUs, developers, budget allocation)
        - Team-specific utility functions with different resource valuations
        - Complex resource interdependencies and constraints
        - BATNA values representing alternative resource sources
        - Structured JSON proposal system for resource requests
        - Win-win solution detection based on efficient resource utilization
    
    Resource Types:
        1. GPU Hours: Computational resources for processing tasks
           - Development Team: High priority for model training and testing
           - Marketing Team: Moderate priority for data analysis
        2. Developer Hours: Human resource allocation
           - Development Team: Critical for implementation work
           - Marketing Team: Needed for integration and campaign development
        3. Budget Allocation: Financial resources for project components
           - Development Team: Infrastructure and tool costs
           - Marketing Team: Campaign execution and market research
    
    Game Mechanics:
        1. Teams propose resource allocation packages
        2. Proposals validated against total resource constraints
        3. Utility calculated based on team-specific resource valuations
        4. Teams can negotiate, trade, or share resources creatively
        5. BATNA decay encourages timely resolution
        6. Success measured by total utility maximization
    
    Attributes:
        total_gpus (int): Total GPU hours available for allocation.
        total_developers (int): Total developer hours available.
        total_budget (int): Total budget available for distribution.
        team_utilities (Dict[str, Dict]): Team-specific utility functions
            defining how each team values different resource combinations.
        batna_values (Dict[str, float]): Alternative resource source values.
        resource_weights (Dict[str, Dict[str, float]]): Team-specific importance
            weights for each resource type in utility calculations.
    
    Example:
        >>> config = {
        ...     "max_rounds": 5,
        ...     "total_gpus": 10,
        ...     "total_developers": 8,
        ...     "total_budget": 100000,
        ...     "batna_decay": 0.02
        ... }
        >>> game = ResourceAllocationGame(config)
        >>> game.initialize_game(["dev_team", "marketing_team"])
        >>> 
        >>> # Development team proposes resource allocation
        >>> proposal = {
        ...     "gpu_hours": 7,      # High GPU need for development
        ...     "developer_hours": 5, # Core development team
        ...     "budget": 60000      # Infrastructure costs
        ... }
        >>> action = {"type": "proposal", "allocation": proposal}
        >>> valid = game.is_valid_action("dev_team", action)
    
    Strategic Considerations:
        - Development Team: Prioritize GPU and developer resources
        - Marketing Team: Focus on budget and developer support
        - Both teams: Identify resource trades that create mutual value
        - Optimal: Find allocations where total utility exceeds individual BATNAs
    
    Resource Efficiency:
        Game encourages:
        - Creative resource sharing arrangements
        - Time-based resource allocation (sequential usage)
        - Hybrid solutions combining different resource types
        - Recognition of complementary resource needs between teams
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize resource allocation negotiation game with team configurations.
        
        Sets up multi-resource negotiation between Development and Marketing
        teams with utility functions, BATNA values, resource constraints,
        and time decay mechanisms. Validates required configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - batnas (Dict[str, float]): BATNA values for each team
                - rounds (int): Maximum negotiation rounds allowed
                - batna_decay (float): Per-round BATNA decay rate (0.0-1.0)
                - total_resources (Dict[str, int]): Available resource pools
                - constraints (Dict): Resource allocation constraints
                - utility_functions (Dict): Team-specific utility parameters
                - uncertainty (Dict, optional): Uncertainty parameters
        
        Raises:
            ValueError: If any required configuration field is missing.
        
        Example:
            >>> config = {
            ...     "batnas": {"development": 50, "marketing": 45},
            ...     "rounds": 5,
            ...     "batna_decay": 0.02,
            ...     "total_resources": {"gpu": 100, "cpu": 100},
            ...     "constraints": {"max_gpu_per_team": 80},
            ...     "utility_functions": {
            ...         "development": {"gpu_coefficient": 0.8, "cpu_coefficient": 0.2},
            ...         "marketing": {"gpu_coefficient": 0.3, "cpu_coefficient": 0.7}
            ...     }
            ... }
            >>> game = ResourceAllocationGame(config)
        """
        # Initialize base class with game type as game_id  
        super().__init__(game_id="resource_allocation", config=config)
        required_fields = [
            "batnas", "rounds", "batna_decay", "total_resources", "constraints", "utility_functions"
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Extract BATNAs from the batnas dictionary
        batnas = config["batnas"]
        self.development_batna = batnas["development"]
        self.marketing_batna = batnas["marketing"]
        self.max_rounds = config["rounds"]
        self.batna_decay = config["batna_decay"]
        
        # Resource allocation specific configuration
        self.total_resources = config["total_resources"]
        self.constraints = config["constraints"]
        
        # Load utility function parameters from config
        utility_functions = config["utility_functions"]
        self.utility_functions = {
            "development": {
                "gpu_coeff": utility_functions["development"]["gpu_coefficient"],
                "cpu_coeff": utility_functions["development"]["cpu_coefficient"],
                "uncertainty_min": utility_functions["development"]["uncertainty_min"],
                "uncertainty_max": utility_functions["development"]["uncertainty_max"]
            },
            "marketing": {
                "gpu_coeff": utility_functions["marketing"]["gpu_coefficient"],
                "cpu_coeff": utility_functions["marketing"]["cpu_coefficient"],
                "uncertainty_min": utility_functions["marketing"]["uncertainty_min"],
                "uncertainty_max": utility_functions["marketing"]["uncertainty_max"]
            }
        }
        
        # Uncertainty parameters (optional)
        self.uncertainty = config.get("uncertainty", {})

    def validate_json_response(self, response: str) -> bool:
        """
        Validate that a response string contains properly formatted JSON.
        
        Checks if the provided response can be parsed as valid JSON and
        contains the required "type" field for action identification.
        Used for input validation before processing team responses.
        
        Args:
            response (str): Raw response string from team to validate.
        
        Returns:
            bool: True if response is valid JSON with "type" field,
                 False otherwise.
        
        Example:
            >>> valid_response = '{"type": "propose", "gpu": 60, "cpu": 40}'
            >>> game.validate_json_response(valid_response)
            True
            >>> invalid_response = 'We want 60 GPU hours'
            >>> game.validate_json_response(invalid_response) 
            False
        """
        try:
            data = json.loads(response.strip())
            return isinstance(data, dict) and "type" in data
        except (json.JSONDecodeError, TypeError):
            return False

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and normalize JSON response from teams into standard format.
        
        Extracts decision data from various JSON response formats, handling
        both direct action format and structured response format. Provides
        robust error recovery with fallback parsing for malformed responses.
        
        Args:
            response (str): Raw JSON response string from team.
        
        Returns:
            Dict[str, Any]: Parsed response containing:
                - decision (Dict[str, Any]): Extracted action data with "type" field
                - raw_response (str): Original response for debugging
        
        Example:
            >>> response = '{"type": "propose", "gpu": 60, "cpu": 40}'
            >>> parsed = game.parse_json_response(response)
            >>> print(parsed["decision"]["type"])
            propose
            >>> print(parsed["decision"]["gpu"])
            60
        
        Note:
            Falls back to {"type": "reject"} for unparseable responses
            to ensure graceful handling of malformed input.
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
            
            # Try to extract type and resource allocation manually as fallback
            type_match = re.search(r'"type":\s*"([^"]+)"', response)
            gpu_match = re.search(r'"gpu_hours":\s*(\d+(?:\.\d+)?)', response)
            cpu_match = re.search(r'"cpu_hours":\s*(\d+(?:\.\d+)?)', response)
            if type_match:
                decision_data = {"type": type_match.group(1)}
                if gpu_match and cpu_match:
                    decision_data["gpu_hours"] = float(gpu_match.group(1))
                    decision_data["cpu_hours"] = float(cpu_match.group(1))
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
        Initialize multi-resource allocation negotiation between development and marketing teams.
        
        Sets up the negotiation environment with randomized role assignments to minimize
        order bias, initializes team-specific utility functions, establishes resource
        constraints, and prepares the game state for active negotiation.
        
        Args:
            players (List[str]): List of exactly 2 player identifiers representing
                the negotiating teams. Order is randomized for role assignment.
        
        Returns:
            Dict[str, Any]: Initial game state containing:
                - players: List of player identifiers with assigned roles
                - current_round: Starting round number (1)
                - max_rounds: Maximum negotiation rounds allowed
                - total_gpu_hours: Total GPU resources available (10)
                - total_cpu_hours: Total CPU resources available (10)
                - private_info: Team-specific utility functions and preferences
                - resource_history: Empty history for tracking allocations
                - agreement_reached: False (negotiation not yet concluded)
        
        Initialization Process:
            1. Validate exactly 2 players provided
            2. Randomly assign development and marketing roles
            3. Set up team-specific utility functions and preferences
            4. Initialize resource constraints and tracking
            5. Create private information for each team
            6. Prepare negotiation state tracking
        
        Role Assignment:
            - Development Team: Higher GPU preference, moderate CPU needs
            - Marketing Team: Higher CPU preference, moderate GPU needs
            - Random assignment prevents order bias effects
        
        Resource Setup:
            - Total GPU Hours: 10 (must be allocated between teams)
            - Total CPU Hours: 10 (must be allocated between teams)
            - Team-specific utility functions for each resource type
        
        Example:
            >>> players = ["model_a", "model_b"]
            >>> initial_state = game.initialize_game(players)
            >>> print(initial_state["total_gpu_hours"])
            10
            >>> print("private_info" in initial_state)
            True
        
        Raises:
            ValueError: If number of players is not exactly 2.
        
        Note:
            Role assignments are logged for debugging but kept private from
            players to maintain negotiation authenticity.
        """
        if len(players) != 2:
            raise ValueError("Resource allocation game requires exactly 2 players")

        self.players = players
        
        # Randomly assign development and marketing roles to eliminate role-based bias
        if random.choice([True, False]):
            self.development = players[0]
            self.marketing = players[1]
            print(f"🎲 [ROLE ASSIGNMENT] {players[0]} = DEVELOPMENT, {players[1]} = MARKETING")
        else:
            self.development = players[1] 
            self.marketing = players[0]
            print(f"🎲 [ROLE ASSIGNMENT] {players[1]} = DEVELOPMENT, {players[0]} = MARKETING")
            
        self.state = self.state.__class__.ACTIVE  # Set to active state

        self.game_data = {
            "game_type": "resource_allocation",
            "players": self.players,
            "rounds": self.max_rounds,
            "current_round": 1,
            "role_assignments": {
                "development": self.development,
                "marketing": self.marketing
            },
            "private_info": {
                self.development: {
                    "role": "development",
                    "team": "development",
                    "utility_function": f"{self.utility_functions['development']['gpu_coeff']}x + {self.utility_functions['development']['cpu_coeff']}y + ε",
                    "batna": self.development_batna,
                    "constraints": self.constraints
                },
                self.marketing: {
                    "role": "marketing", 
                    "team": "marketing",
                    "utility_function": f"{self.utility_functions['marketing']['gpu_coeff']}x + {self.utility_functions['marketing']['cpu_coeff']}y + ι", 
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
        """
        Calculate time-adjusted BATNA value for specified team and round.
        
        Applies exponential decay to the team's initial BATNA value based on
        the current round number, simulating decreasing value of alternative
        resource sources over time. Creates time pressure encouraging agreement.
        
        Args:
            player (str): Team identifier ("development" or "marketing").
            round_num (int): Current round number (1-based).
        
        Returns:
            float: Time-adjusted BATNA value for the specified round.
        
        Example:
            >>> # Initial development BATNA: 50, decay rate: 0.02
            >>> round_1_batna = game.get_current_batna("development", 1)
            >>> print(f"Round 1 BATNA: {round_1_batna:.1f}")
            Round 1 BATNA: 49.0
            >>> round_3_batna = game.get_current_batna("development", 3)
            >>> print(f"Round 3 BATNA: {round_3_batna:.1f}")
            Round 3 BATNA: 48.0
        
        Note:
            BATNA decay formula: initial_batna * (1 - decay_rate)^round_num
        """
        if player == self.development:
            decay_rate = self.batna_decay["development"]
            base_batna = self.development_batna
        else:
            decay_rate = self.batna_decay["marketing"]
            base_batna = self.marketing_batna

        return base_batna * ((1 - decay_rate) ** (round_num-1))

    def calculate_utility(self, player: str, gpu_hours: float, cpu_hours: float, round_num: int) -> float:
        """
        Calculate team-specific utility value for proposed resource allocation.
        
        Computes utility scores using configurable team-specific coefficients and
        uncertainty factors. Each team has different preferences for GPU vs CPU
        resources based on their operational needs and strategic priorities.
        
        Args:
            player (str): Team identifier (development or marketing).
            gpu_hours (float): Proposed GPU resource allocation for the team.
            cpu_hours (float): Proposed CPU resource allocation for the team.
            round_num (int): Current negotiation round (used for future extensions).
        
        Returns:
            float: Total utility value including base utility and uncertainty factor.
                Higher values indicate more attractive proposals for the team.
        
        Utility Calculation:
            Base utility = (gpu_coeff × gpu_hours) + (cpu_coeff × cpu_hours)
            Final utility = base_utility + random_uncertainty_factor
        
        Team Coefficients:
            - Development: Higher GPU coefficient, moderate CPU coefficient
            - Marketing: Higher CPU coefficient, moderate GPU coefficient
            - Configurable via utility_functions in game setup
        
        Uncertainty Factor:
            Random value within team-specific bounds to model negotiation
            uncertainty and prevent deterministic outcomes.
        
        Example:
            >>> # Development team evaluating GPU-heavy allocation
            >>> utility = game.calculate_utility("dev_team", 8.0, 2.0, 1)
            >>> print(f"Development utility: {utility:.1f}")
            Development utility: 28.3  # Including uncertainty
        
        Note:
            Uncertainty factors add realism but may cause slight result
            variations between identical runs. Set narrow bounds for
            more predictable behavior.
        """
        role = "development" if player == self.development else "marketing"
        utility_params = self.utility_functions[role]
        
        # Calculate base utility using configurable coefficients
        base_utility = (utility_params["gpu_coeff"] * gpu_hours + 
                       utility_params["cpu_coeff"] * cpu_hours)
        
        # Add configurable uncertainty factor
        uncertainty = random.uniform(utility_params["uncertainty_min"], 
                                   utility_params["uncertainty_max"])
        
        return base_utility + uncertainty

    def _validate_resource_constraints(self, gpu_hours: float, cpu_hours: float) -> bool:
        """
        Validate proposed resource allocation against system constraints.
        
        Checks if the proposed resource allocation satisfies all defined
        constraints including total resource limits, minimum allocations,
        and any custom business rules defined in the configuration.
        
        Args:
            gpu_hours (float): Proposed GPU hours allocation.
            cpu_hours (float): Proposed CPU hours allocation.
        
        Returns:
            bool: True if allocation satisfies all constraints, False otherwise.
        
        Example:
            >>> # Check if allocation is within total resource limits
            >>> valid = game._validate_resource_constraints(60, 40)
            >>> print(f"Allocation valid: {valid}")
            Allocation valid: True
            >>> # Check over-allocation
            >>> valid = game._validate_resource_constraints(120, 40)
            >>> print(f"Over-allocation valid: {valid}")
            Over-allocation valid: False
        
        Note:
            Constraints typically include total resource limits and
            minimum viable allocations for each team.
        """
        # Total resource constraint: x + y <= total_resources
        if gpu_hours + cpu_hours > self.total_resources:
            return False

        # GPU-CPU constraint: 4x + 4y <= gpu_bandwidth
        if 4 * gpu_hours + 4 * cpu_hours > self.constraints["gpu_bandwidth"]:
            return False
            
        # Minimum allocation constraints
        if gpu_hours < self.constraints["min_gpu"]:
            return False
        if cpu_hours < self.constraints["min_cpu"]:
            return False
            
        return True

    def check_constraints_and_update(self, gpu_hours: float, cpu_hours: float) -> None:
        """
        Validate resource allocation against all constraints and update game state.
        
        Performs comprehensive validation of proposed resource allocation against
        multiple constraint types including total resource limits, bandwidth
        constraints, and resource coupling requirements. Updates game state
        with detailed validation results and error messages.
        
        Args:
            gpu_hours (float): Proposed GPU resource allocation to validate.
            cpu_hours (float): Proposed CPU resource allocation to validate.
        
        Side Effects:
            Updates self.game_data['constraint_check'] with detailed validation
            results including constraint status, violation messages, and
            resource utilization analysis.
        
        Constraint Validation:
            1. Total Resource Limit: gpu_hours + cpu_hours ≤ total_resources
            2. GPU Bandwidth: 4×gpu_hours + 4×cpu_hours ≤ gpu_bandwidth
            3. Individual Resource Bounds: Non-negative allocations
            4. Resource Coupling: Interdependency constraints
        
        Game State Updates:
            Creates or updates 'constraint_check' entry containing:
                - constraints_met: Boolean overall validation result
                - messages: List of specific constraint violation descriptions
                - gpu_hours: Validated GPU allocation
                - cpu_hours: Validated CPU allocation
                - total_usage: Combined resource utilization
        
        Example:
            >>> # Valid allocation within all constraints
            >>> game.check_constraints_and_update(4.0, 6.0)
            >>> print(game.game_data['constraint_check']['constraints_met'])
            True
            
            >>> # Invalid allocation exceeding total resources
            >>> game.check_constraints_and_update(8.0, 12.0)
            >>> print(game.game_data['constraint_check']['constraints_met'])
            False
            >>> print(game.game_data['constraint_check']['messages'])
            ['Total resources exceeded: 20.0 > 10.0']
        
        Note:
            This method provides detailed constraint analysis for debugging
            and user feedback, supporting complex multi-constraint validation
            scenarios in resource allocation negotiations.
        """
        constraints_met = True
        messages = []

        # Check total resource constraint
        if gpu_hours + cpu_hours > self.total_resources:
            constraints_met = False
            messages.append(f"Total resources exceeded: {gpu_hours + cpu_hours} > {self.total_resources}")

        # Check GPU-CPU constraint
        if 4 * gpu_hours + 4 * cpu_hours > self.constraints["gpu_bandwidth"]:
            constraints_met = False
            messages.append(f"GPU-Bandwidth limit exceeded: {4 * gpu_hours + 4 * cpu_hours} > {self.constraints['gpu_bandwidth']}")

        # Check minimum allocation constraints
        if gpu_hours < self.constraints["min_gpu"]:
            constraints_met = False
            messages.append(f"GPU hours below minimum: {gpu_hours} < {self.constraints['min_gpu']}")
        if cpu_hours < self.constraints["min_cpu"]:
            constraints_met = False
            messages.append(f"CPU hours below minimum: {cpu_hours} < {self.constraints['min_cpu']}")

        # Update game data
        self.game_data["constraints_met"] = constraints_met

        # Print results
        if constraints_met:
            print("✅ All constraints are satisfied.")
        else:
            print("❌ Constraints violated:")
            for message in messages:
                print(f"   - {message}")

    def is_valid_action(self, player: str, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """
        Validate team action against resource allocation rules and constraints.
        
        Comprehensive validation of negotiation actions including proposals and
        acceptances. Supports both direct action format and structured response
        format with "decision" wrapper. Ensures actions comply with resource
        constraints, proposal limits, and valid action types.
        
        Args:
            player (str): Identifier of the team taking the action.
                Must be registered development or marketing team.
            action (Dict[str, Any]): Action data to validate. Supported formats:
                - Direct: {"type": "propose", "gpu": 6, "cpu": 4}
                - Structured: {"decision": {"type": "propose", "gpu": 6, "cpu": 4}}
            game_state (Dict[str, Any]): Current game state containing round
                information, proposal counts, and resource constraints.
        
        Returns:
            bool: True if action is valid and can be processed, False otherwise.
        
        Validation Rules:
            - Action must have valid "type" field (propose, accept)
            - Proposals must include numeric "gpu" and "cpu" fields
            - Resource allocations must respect total resource constraints
            - GPU + CPU allocations must not exceed available pools
            - Resource values must be non-negative numbers
            - Player must not exceed proposal limits
        
        Resource Constraints:
            - Total GPU hours available: 10
            - Total CPU hours available: 10
            - Individual allocations must be ≤ total resources
            - Combined team allocations must sum to ≤ totals
        
        Example:
            >>> # Valid resource proposal
            >>> action = {"type": "propose", "gpu": 6, "cpu": 4}
            >>> is_valid = game.is_valid_action("dev_team", action, game_state)
            >>> print(is_valid)
            True
            
            >>> # Invalid proposal exceeding resources
            >>> invalid = {"type": "propose", "gpu": 12, "cpu": 8}
            >>> is_valid = game.is_valid_action("mkt_team", invalid, game_state)
            >>> print(is_valid)
            False
        
        Note:
            Invalid actions are logged but do not raise exceptions, allowing
            graceful handling of malformed AI model responses and constraint
            violations.
        """
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
            
        max_proposals = self.max_rounds-1
        player_proposals = game_state.get(f"{player}_proposal_count", 0)

        if action_type in ["offer", "propose"]:  # Accept both "offer" and "propose"
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} tried to make offer but exceeded proposal limit ({player_proposals}/{max_proposals})")
                return False
                
            gpu_hours = action_data.get("gpu_hours", 0)
            cpu_hours = action_data.get("cpu_hours", 0)

            if gpu_hours <= 0 or cpu_hours <= 0:
                return False

            # Validate constraints
            if not self._validate_resource_constraints(gpu_hours, cpu_hours):
                print(f"⚠️ Player {player} offer violates constraints: GPU={gpu_hours}, CPU={cpu_hours}")
                return False

            return True

        elif action_type in ["accept", "reject"]:
            return True

        elif action_type in ["counter", "counteroffer"]:
            # Check proposal limit for counters too
            if player_proposals >= max_proposals:
                return False
                
            # Treat counter/counteroffer as regular offers
            gpu_hours = action_data.get("gpu_hours", 0)
            cpu_hours = action_data.get("cpu_hours", 0)
            return (gpu_hours > 0 and cpu_hours > 0 and
                   self._validate_resource_constraints(gpu_hours, cpu_hours))

        elif action_type in ["offer_accepted", "offer_response"]:
            # Treat these as accept actions
            return True

        elif action_type == "noop":
            # Allow no-op actions as fallback
            return True

        return False

    def process_actions(self, actions: Dict[str, Dict[str, Any]], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process player actions with proposal limits and enhanced validation."""
        current_round = game_state["current_round"]
        max_proposals = self.max_rounds-1  # Use rounds from YAML config
        
        print(f"🔍 Processing actions for round {current_round}: {actions}", file=sys.stderr)

        # Initialize proposal counters if not present
        for player in [self.development, self.marketing]:
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

        # Normalize action types - treat counter/counteroffer as offers, handle empty types
        normalized_actions = {}
        for player, action in processed_actions.items():
            action_type = action.get("type", "")
            
            # Handle empty or invalid action types
            if not action_type or action_type == "":
                print(f"⚠️ Player {player} provided empty action type, treating as reject")
                normalized_actions[player] = {"type": "reject"}
            elif action_type in ["counter", "counteroffer"]:
                # Convert to offer
                normalized_actions[player] = {
                    "type": "offer", 
                    "gpu_hours": action.get("gpu_hours"),
                    "cpu_hours": action.get("cpu_hours")
                }
            elif action_type == "propose":
                # Convert "propose" to "offer" for consistency
                normalized_actions[player] = {
                    "type": "offer", 
                    "gpu_hours": action.get("gpu_hours"),
                    "cpu_hours": action.get("cpu_hours")
                }
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

        # Process rejections - only end if proposal limit reached
        for player, action in responses.items():
            if action.get("type") == "reject":
                player_proposals = game_state.get(f"{player}_proposal_count", 0)
                if player_proposals >= max_proposals:
                    print(f"❌ Player {player} rejected after reaching proposal limit ({player_proposals}/{max_proposals})")
                    return self._create_no_agreement(game_state)
                else:
                    print(f"⚠️ Player {player} rejected but still has proposals remaining ({player_proposals}/{max_proposals}). Continuing negotiation.")

        # Process acceptances FIRST (before new offers) to ensure only previous round offers can be accepted
        for player, action in responses.items():
            if action.get("type") == "accept":
                # Find the offer being accepted
                other_player = self.marketing if player == self.development else self.development
                if f"{other_player}_last_offer" in game_state:
                    offer_data = game_state[f"{other_player}_last_offer"]
                    gpu_hours = offer_data["gpu_hours"]
                    cpu_hours = offer_data["cpu_hours"]
                    # Get the round when the accepted offer was made
                    offer_round = game_state.get(f"{other_player}_last_offer_round", current_round)
                    
                    # Validate that the offer being accepted was made in a previous round
                    if offer_round >= current_round:
                        print(f"⚠️ Player {player} tried to accept offer made in same round {offer_round}. Offers can only be accepted from previous rounds.")
                        continue  # Skip this acceptance, don't end the game

                    print(f"✅ Player {player} accepted offer of GPU={gpu_hours}, CPU={cpu_hours} (made in round {offer_round})")

                    # Use the BATNA from when the offer was made, but record agreement as happening in current round
                    return self._create_agreement(gpu_hours, cpu_hours, current_round, game_state, offer_round_for_batna=offer_round)
                else:
                    print(f"⚠️ Player {player} tried to accept but no offer exists")

        # Process offers with proposal limit validation (AFTER acceptances)
        for player, action in offers.items():
            player_proposals = game_state.get(f"{player}_proposal_count", 0)
            
            # Check proposal limit
            if player_proposals >= max_proposals:
                print(f"⚠️ Player {player} exceeded proposal limit ({player_proposals}/{max_proposals}). Ignoring additional offers.")
                # Don't process this offer, but don't end negotiation unless they also rejected
                continue
            
            # Valid offer - process it
            gpu_hours = action.get("gpu_hours")
            cpu_hours = action.get("cpu_hours")
            game_state[f"{player}_last_offer"] = {"gpu_hours": gpu_hours, "cpu_hours": cpu_hours}
            game_state[f"{player}_last_offer_round"] = current_round  # Track when offer was made
            game_state[f"{player}_proposal_count"] = player_proposals + 1
            print(f"💡 Player {player} made offer GPU={gpu_hours}, CPU={cpu_hours} (proposal {player_proposals + 1}/{max_proposals})")

        # Check for convergence: if both players made identical offers, create agreement
        if len(offers) == 2:  # Both players made offers this round
            offer_data = [action for action in offers.values()]
            if (len(offer_data) == 2 and 
                offer_data[0].get("gpu_hours") == offer_data[1].get("gpu_hours") and
                offer_data[0].get("cpu_hours") == offer_data[1].get("cpu_hours") and
                offer_data[0].get("gpu_hours") is not None):  # All offers are identical and valid
                gpu_hours = offer_data[0].get("gpu_hours")
                cpu_hours = offer_data[0].get("cpu_hours")
                print(f"🎉 CONVERGENCE! Both players offered GPU={gpu_hours}, CPU={cpu_hours} - Creating automatic agreement!")
                return self._create_agreement(gpu_hours, cpu_hours, current_round, game_state)

        # Update round
        game_state["current_round"] += 1
        
        # Debug logging for round progression
        print(f"🔄 Updated to round {game_state['current_round']}/{self.max_rounds}", file=sys.stderr)
        
        # Check proposal status for both players
        dev_proposals = game_state.get(f"{self.development}_proposal_count", 0)
        mkt_proposals = game_state.get(f"{self.marketing}_proposal_count", 0)
        print(f"🔍 Proposal status: {self.development}={dev_proposals}/4, {self.marketing}={mkt_proposals}/4", file=sys.stderr)

        # Check if deadline reached - but allow extra rounds for final responses
        # Players should have a chance to accept/reject final proposals
        max_total_rounds = self.max_rounds
        
        if game_state["current_round"] > max_total_rounds:
            print(f"⏰ Maximum rounds ({max_total_rounds}) reached - Ending negotiation")
            return self._create_no_agreement(game_state)

        return game_state

    def _create_agreement(self, gpu_hours: float, cpu_hours: float, current_round: int, game_state: Dict[str, Any], offer_round_for_batna: int = None) -> Dict[str, Any]:
        """Create agreement result.
        
        Args:
            gpu_hours: GPU hours in the agreement
            cpu_hours: CPU hours in the agreement
            current_round: Round when agreement was reached (for recording)
            game_state: Current game state
            offer_round_for_batna: Round when the accepted offer was made (for BATNA calculation)
        """
        # Use offer_round_for_batna if provided, otherwise use current_round
        batna_round = offer_round_for_batna if offer_round_for_batna is not None else current_round
        
        dev_batna = self.get_current_batna(self.development, batna_round)
        mkt_batna = self.get_current_batna(self.marketing, batna_round)

        # Calculate utility using the configurable utility functions
        dev_params = self.utility_functions["development"]
        mkt_params = self.utility_functions["marketing"]
        dev_utility = dev_params["gpu_coeff"] * gpu_hours + dev_params["cpu_coeff"] * cpu_hours
        mkt_utility = mkt_params["gpu_coeff"] * gpu_hours + mkt_params["cpu_coeff"] * cpu_hours

        # Calculate utility surplus (utility - BATNA)
        dev_surplus = dev_utility - dev_batna
        mkt_surplus = mkt_utility - mkt_batna
        
        # DEBUG: Log the exact calculation values
        print(f"🔍 [RESOURCE DEBUG] Agreement in round {current_round}: GPU={gpu_hours}, CPU={cpu_hours}")
        if offer_round_for_batna is not None and offer_round_for_batna != current_round:
            print(f"🔍 [BATNA DEBUG] Using BATNA from offer round {batna_round} (offer made), agreement in round {current_round}")
        else:
            print(f"🔍 [BATNA DEBUG] Using BATNA from round {batna_round}")
        print(f"🔍 [BATNA DEBUG] Config BATNAs: development={self.development_batna}, marketing={self.marketing_batna}")
        print(f"🔍 [BATNA DEBUG] Decay rates: development={self.batna_decay['development']}, marketing={self.batna_decay['marketing']}")
        print(f"🔍 [BATNA DEBUG] Calculated BATNAs: development={dev_batna:.2f}, marketing={mkt_batna:.2f}")
        print(f"🔍 [UTILITY DEBUG] Utilities: development={dev_utility:.2f}, marketing={mkt_utility:.2f}")
        print(f"🔍 [SURPLUS DEBUG] Surpluses: development={dev_surplus:.2f}, marketing={mkt_surplus:.2f}")
        print(f"🎲 [ROLE DEBUG] Development={self.development}, Marketing={self.marketing}")

        game_state.update({
            "agreement_reached": True,
            "game_ended": True,  # Explicitly mark game as ended
            "agreed_allocation": {
                "gpu_hours": gpu_hours,
                "cpu_hours": cpu_hours
            },
            "agreement_round": current_round,
            "role_assignments": {
                "development": self.development,
                "marketing": self.marketing
            },
            "final_utilities": {
                self.development: dev_utility,
                self.marketing: mkt_utility
            },
            "utility_surpluses": {
                self.development: dev_surplus,
                self.marketing: mkt_surplus
            },
            "batnas_at_agreement": {
                self.development: dev_batna,
                self.marketing: mkt_batna
            }
        })

        return game_state

    def _create_no_agreement(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create final result when no resource allocation agreement is reached.
        
        Generates comprehensive failure outcome where both teams resort to
        their alternative resource sources (BATNAs). No resource allocation
        is made and no surplus is generated as negotiation failed.
        
        Args:
            game_state (Dict[str, Any]): Current game state dictionary.
        
        Returns:
            Dict[str, Any]: No agreement result containing:
                - agreement_reached (bool): False
                - final_allocation (Dict): No resource allocation made
                - final_utilities (Dict[str, float]): BATNA-based utilities
                - batnas_at_agreement (Dict[str, float]): Final BATNA values
                - utility_surplus (Dict[str, float]): Zero surplus for both
                - winner (None): No winner in failed negotiations
        
        Example:
            >>> result = game._create_no_agreement(game_state)
            >>> print(f"Agreement: {result['agreement_reached']}")
            >>> print(f"Final utilities: {result['final_utilities']}")
            Agreement: False
            Final utilities: {'development': 0.0, 'marketing': 0.0}
        """
        print(f"🎲 [ROLE DEBUG] No Agreement - Development={self.development}, Marketing={self.marketing}")
        print(f"🎲 [ROLE DEBUG] {self.development} utility=0, {self.marketing} utility=0")
        
        game_state.update({
            "agreement_reached": False,
            "game_ended": True,  # Explicitly mark game as ended
            "role_assignments": {
                "development": self.development,
                "marketing": self.marketing
            },
            "final_utilities": {
                self.development: 0.0,  # No deal utility
                self.marketing: 0.0
            }
        })
        return game_state

    def is_game_over(self, game_state: Dict[str, Any]) -> bool:
        """
        Determine if the resource allocation negotiation has reached a terminal state.
        
        Checks for various end conditions including resource agreement reached,
        maximum rounds exceeded, or explicit rejections that end negotiation.
        
        Args:
            game_state (Dict[str, Any]): Current game state to evaluate.
        
        Returns:
            bool: True if game should terminate, False if negotiation continues.
        
        Example:
            >>> # Agreement reached
            >>> game_state = {"agreement_reached": True}
            >>> game.is_game_over(game_state)
            True
            >>> # Maximum rounds exceeded
            >>> game_state = {"current_round": 6, "agreement_reached": False}
            >>> game.is_game_over(game_state)  # max_rounds = 5
            True
        """
        current_round = game_state.get("current_round", 1)
        agreement_reached = game_state.get("agreement_reached", False)
        game_ended = game_state.get("game_ended", False)
        
        print(f"🔍 is_game_over check: round={current_round}/{self.max_rounds}, agreement={agreement_reached}, ended={game_ended}", file=sys.stderr)
        
        result = (agreement_reached or game_ended or current_round > self.max_rounds)
        print(f"🔍 is_game_over result: {result}", file=sys.stderr)
        
        return result
    def _get_neutral_role_label(self, player_id: str) -> str:
        """
        Map team identifier to neutral role label to reduce cognitive bias.
        
        Provides neutral terminology ("Team A"/"Team B") instead of loaded
        terms ("development"/"marketing") to minimize role-based behavioral
        biases in prompts and communications.
        
        Args:
            player_id (str): Team identifier to map.
        
        Returns:
            str: Neutral role label ("Team A" or "Team B").
        
        Example:
            >>> # If development is team1, marketing is team2
            >>> game._get_neutral_role_label("team1")
            'Team A'
            >>> game._get_neutral_role_label("team2")
            'Team B'
        """
        if player_id == self.development:
            return "ROLE A"
        else:
            return "ROLE B"

    def get_game_prompt(self, player_id: str) -> str:
        """
        Generate comprehensive resource allocation negotiation prompt for teams.
        
        Creates detailed, contextual prompts for GPU and CPU resource negotiations
        between Development and Marketing teams. Includes current game state,
        resource constraints, utility calculations, and structured action formatting
        requirements. Uses neutral role terminology to minimize cognitive bias.
        
        Args:
            player_id (str): Identifier of the team requesting the prompt.
                Must be either development or marketing team identifier.
        
        Returns:
            str: Comprehensive negotiation prompt containing:
                - Team-specific role context and resource priorities
                - Current resource allocation status and constraints
                - BATNA thresholds and utility calculations
                - Opponent's latest proposal (if available)
                - Available actions and JSON formatting requirements
                - Strategic guidance for resource optimization
                - Proposal limits and round tracking information
        
        Prompt Components:
            - Neutral role terminology (Team A/B vs development/marketing)
            - Resource constraint specifications (GPU/CPU limits)
            - Team-specific utility functions and preferences
            - Current negotiation state and round progression
            - BATNA-based acceptance criteria
            - Structured JSON response format requirements
            - Strategic recommendations for win-win solutions
        
        Resource Context:
            - Total GPU hours: 10 (split between teams)
            - Total CPU hours: 10 (split between teams)
            - Team-specific utility calculations
            - Time-decaying BATNA values
        
        Example:
            >>> prompt = game.get_game_prompt("dev_team")
            >>> print("GPU hours" in prompt)    # Resource context
            True
            >>> print("Team A" in prompt)       # Neutral terminology
            True
            >>> print("JSON" in prompt)         # Format requirements
            True
        
        Note:
            Returns error message if game is not properly initialized with
            team assignments. Prompts adapt to current resource constraints
            and proposal limits.
        """
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
        max_proposals = self.max_rounds - 1
        can_propose = player_proposals < max_proposals

        # Map internal roles to neutral display roles to reduce bias
        internal_role = "development" if player_id == self.development else "marketing"
        neutral_role = self._get_neutral_role_label(player_id)
        
        role = private_info.get("role", "unknown")
        team_name = "Development Team" if role == "development" else "Marketing Team"
        utility_func = private_info.get("utility_function", "unknown")
        preference = "GPU-heavy tasks" if role == "development" else "CPU-intensive operations"
        
        # Offer status
        offer_history = []
        if my_offer:
            gpu = my_offer.get("gpu_hours", 0)
            cpu = my_offer.get("cpu_hours", 0)
            offer_history.append(f"- Your last offer: GPU={gpu}, CPU={cpu}")
        if other_offer:
            gpu = other_offer.get("gpu_hours", 0)
            cpu = other_offer.get("cpu_hours", 0)
            offer_history.append(f"- Opponent's offer: GPU={gpu}, CPU={cpu}")
        offer_status = "\n".join(offer_history) if offer_history else "No offers made yet."

        # Role-specific configuration
        role_priorities = ""
        if role == "development":
            role_priorities = (
                f"Your priority is to maximize GPU hours.\n"
            )

        else:  # Marketing
            role_priorities = (
                f"Your priority is to maximize CPU hours.\n"
            )

        # Acceptance guidance
        acceptance_guidance = ""
        proposal_guidance = f"📊 You have **{max_proposals - player_proposals}** proposals remaining out of {max_proposals} total."
        rounds_remaining = max_proposals - player_proposals
        
        if other_offer is not None:
            # Calculate utility for the proposed offer
            gpu_hours = other_offer.get("gpu_hours", 0)
            cpu_hours = other_offer.get("cpu_hours", 0)
            proposed_utility = self.calculate_utility(player_id, gpu_hours, cpu_hours, current_round)
            is_within_batna = proposed_utility >= batna
            
            if is_within_batna:
                if rounds_remaining == 0:  # No proposals left - encourage acceptance
                    acceptance_guidance = (
                        f"🎯 FINAL ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is better than your BATNA ({batna:.1f}). You have no proposals left - ACCEPT to secure this beneficial deal!\n"
                    )
                elif rounds_remaining == 1:  # Last proposal - be more encouraging
                    acceptance_guidance = (
                        f"🎯 ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is better than your BATNA ({batna:.1f}). With only 1 proposal left, consider accepting or making the last counter offer.\n"
                    )
                else:  # Multiple proposals left - encourage exploration
                    acceptance_guidance = (
                        f"💡 ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is better than your BATNA ({batna:.1f}), but you have {rounds_remaining} proposals left. You might negotiate for an even better deal.\n"
                    )
            else:
                utility_gap = batna - proposed_utility
                if rounds_remaining == 0:  # No proposals left - suggest accepting to avoid no-deal
                    acceptance_guidance = (
                        f"🚨 FINAL DECISION: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is {utility_gap:.1f} below your BATNA ({batna:.1f}). You have no proposals left. ACCEPT to avoid no-deal or REJECT.\n"
                    )
                elif rounds_remaining == 1:  # Last proposal - be more encouraging
                    acceptance_guidance = (
                        f"🎯 ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is {utility_gap:.1f} below your BATNA ({batna:.1f}). With only 1 proposal left, consider accepting or making the last counter offer.\n"
                    )
                else:
                    acceptance_guidance = (
                        f"⚠️ ANALYSIS: The opponent's offer (GPU={gpu_hours}, CPU={cpu_hours}) gives you utility {proposed_utility:.1f}, "
                        f"which is {utility_gap:.1f} below your BATNA ({batna:.1f}). You should negotiate for a better allocation.\n"
                    )

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

        prompt = f"""=== RESOURCE ALLOCATION NEGOTIATION ===
{round_display} | Role: {neutral_role}

YOUR PRIORITIES: {role_priorities}
GOAL: Maximize your utility: {utility_func}
Your BATNA (Best Alternative): {batna:.1f}
{proposal_guidance}

CURRENT SITUATION:
{offer_status}
{acceptance_guidance}

CONSTRAINTS & RULES:
- x is the total GPU hours in system
- y is the total CPU hours in system
- Total resources: x + y ≤ {self.total_resources}
- GPU-Bandwidth limit: 4x + 4y ≤ {self.constraints['gpu_bandwidth']}
- Minimum allocations: x ≥ {self.constraints['min_gpu']}, y ≥ {self.constraints['min_cpu']}

RESPONSE FORMAT: Respond with ONLY valid JSON. No explanations.
Valid responses:
{{"type": "accept"}}  // Accept the opponent's last offer
{{"type": "offer", "gpu_hours": 30, "cpu_hours": 25}}  // Propose new allocation (if proposals remain)
{{"type": "reject"}}  // Reject and end negotiation

EXAMPLE OFFERS:
{{"type": "offer", "gpu_hours": 30, "cpu_hours": 25}}

Do NOT repeat any of the rules or instructions in your response. Focus on negotiation.

Your response:"""

        return prompt

    # Abstract methods required by BaseGame interface
    def process_action(self, action: PlayerAction) -> Dict[str, Any]:
        """
        Process a single team action and update the game state accordingly.
        
        Handles individual team actions by converting to batch format and
        delegating to the process_actions method. Required by BaseGame
        interface for single-action processing compatibility.
        
        Args:
            action (PlayerAction): Team action to process containing player_id,
                action_type, action_data, timestamp, and round_number.
        
        Returns:
            Dict[str, Any]: Updated game state after processing the action.
        
        Example:
            >>> action = PlayerAction(
            ...     player_id="development",
            ...     action_type="propose",
            ...     action_data={"gpu": 60, "cpu": 40},
            ...     timestamp=1609459200.0,
            ...     round_number=2
            ... )
            >>> new_state = game.process_action(action)
        """
        # For compatibility with BaseGame interface, delegate to process_actions
        actions_dict = {action.player_id: {"type": action.action_type, **action.action_data}}
        return self.process_actions(actions_dict, self.game_data)

    def check_end_conditions(self) -> bool:
        """
        Check if the resource allocation negotiation should terminate.
        
        Evaluates termination conditions by delegating to the is_game_over
        method. Required by BaseGame interface for consistent end condition
        checking across all game implementations.
        
        Returns:
            bool: True if game should end, False if negotiation continues.
        
        Example:
            >>> game.check_end_conditions()
            True  # If agreement reached or max rounds exceeded
        """
        return self.is_game_over(self.game_data)

    def calculate_scores(self) -> Dict[str, float]:
        """
        Calculate final utility scores for all participating teams.
        
        Returns final utility values if resource agreement was reached,
        or BATNA values for both teams if negotiation failed. Required
        by BaseGame interface for consistent scoring across implementations.
        
        Returns:
            Dict[str, float]: Mapping of team identifiers to final utility
                scores. Positive values indicate successful resource allocation.
        
        Example:
            >>> # Successful negotiation
            >>> scores = game.calculate_scores()
            >>> print(scores)
            {'development': 56.0, 'marketing': 44.0}
            >>> # Failed negotiation  
            >>> scores = game.calculate_scores()
            >>> print(scores)
            {'development': 50.0, 'marketing': 45.0}  # BATNA values
        """
        if self.game_data.get("agreement_reached", False):
            return self.game_data.get("final_utilities", {})
        else:
            # Return BATNA values if no agreement
            return {
                self.development: self.development_batna,
                self.marketing: self.marketing_batna
            }