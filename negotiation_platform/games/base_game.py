"""
Base Game Interface
===================

Defines the abstract interface and data structures for all negotiation games
within the platform. This module provides the foundation for plug-and-play
game integration with consistent APIs and type safety.

Key Components:
    - BaseGame: Abstract base class for all negotiation game implementations
    - GameState: Enumeration of possible game states throughout lifecycle
    - PlayerAction: Data structure representing individual player actions
    - GameResult: Data structure containing complete game outcomes

Architecture:
    The base game interface follows the Template Method pattern, where concrete
    games implement specific game logic while inheriting common infrastructure.
    This enables consistent behavior across different negotiation scenarios.

Game Lifecycle:
    1. WAITING: Game created but not yet initialized with players
    2. ACTIVE: Game in progress with players taking turns
    3. COMPLETED: Game finished successfully with final results
    4. FAILED: Game terminated due to errors or violations

Design Patterns:
    - Template Method: Common game infrastructure with specific implementations
    - State Machine: Clear state transitions and lifecycle management
    - Data Transfer Objects: Structured data containers for actions and results
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class GameState(Enum):
    """
    Enumeration of possible game states throughout the negotiation lifecycle.
    
    Defines the complete state machine for negotiation games, enabling proper
    state tracking and transition validation throughout the game lifecycle.
    
    States:
        WAITING: Game instance created but not yet initialized with players.
            No actions can be taken in this state.
        ACTIVE: Game is in progress with players actively negotiating.
            Actions are validated and processed in this state.
        COMPLETED: Game finished successfully with valid final results.
            No further actions are accepted.
        FAILED: Game terminated abnormally due to errors or rule violations.
            Game data may be incomplete or invalid.
    
    State Transitions:
        WAITING → ACTIVE: When game is initialized with valid players
        ACTIVE → COMPLETED: When end conditions are met successfully
        ACTIVE → FAILED: When unrecoverable errors occur
        Any state → FAILED: When critical failures are detected
    
    Example:
        >>> game = ConcreteGame("game_1", {})
        >>> print(game.state)
        GameState.WAITING
        >>> game.initialize_game(["player1", "player2"])
        >>> print(game.state)
        GameState.ACTIVE
    """
    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PlayerAction:
    """
    Data structure representing a single player action within a negotiation game.
    
    PlayerAction serves as a standardized container for all player interactions
    during negotiations. It captures both the action content and contextual
    metadata necessary for game processing and analysis.
    
    Attributes:
        player_id (str): Unique identifier of the player who took this action.
            Must match one of the registered players in the game.
        action_type (str): Category or type of action taken. Common types include:
            - "proposal": Player making an offer or suggestion
            - "acceptance": Player accepting a previous proposal
            - "rejection": Player rejecting a previous proposal
            - "counter_proposal": Player modifying a previous proposal
            - "message": General communication or clarification
        action_data (Dict[str, Any]): Action-specific data and parameters.
            Structure varies by action_type and game type. Examples:
            - Proposal: {"price": 45000, "terms": "cash_payment"}
            - Acceptance: {"accepted_proposal_id": "prop_123"}
            - Message: {"text": "I need to consider this offer"}
        timestamp (float): Unix timestamp when the action was taken.
            Used for timing analysis and action ordering.
        round_number (int): The negotiation round when this action occurred.
            Enables round-based analysis and game flow tracking.
    
    Example:
        >>> action = PlayerAction(
        ...     player_id="player_1",
        ...     action_type="proposal",
        ...     action_data={"price": 42000, "warranty": True},
        ...     timestamp=1609459200.0,
        ...     round_number=3
        ... )
        >>> print(action.action_data["price"])
        42000
    
    Note:
        PlayerAction objects are immutable once created (dataclass with frozen=False
        by default, but should be treated as read-only after creation).
    """
    player_id: str
    action_type: str
    action_data: Dict[str, Any]
    timestamp: float
    round_number: int

@dataclass
class GameResult:
    """
    Data structure containing the complete outcome and results of a negotiation game.
    
    GameResult serves as the authoritative record of a completed negotiation,
    containing all final state information needed for analysis, metrics calculation,
    and reporting. It provides a standardized format for game outcomes across
    different negotiation types.
    
    Attributes:
        game_id (str): Unique identifier for this specific game session.
            Used to correlate results with logs and analysis data.
        players (List[str]): Complete list of all players who participated.
            Maintains original player order for consistent analysis.
        winner (Optional[str]): Identifier of the winning player, if applicable.
            None for games without clear winners (mutual agreements, ties, etc.).
            Some games may have multiple winners or no winner concept.
        final_scores (Dict[str, float]): Final scores for each player.
            Maps player_id to their game-specific score. Score interpretation
            varies by game type (utility values, points, satisfaction ratings, etc.).
        total_rounds (int): Total number of negotiation rounds completed.
            Includes all rounds from game start to termination.
        game_data (Dict[str, Any]): Complete final game state and additional data.
            Contains game-specific information such as:
            - final_agreement: Terms of any reached agreement
            - batna_values: Best alternatives for each player
            - resource_allocations: Final resource distributions
            - negotiation_history: Detailed interaction log
        success (bool): Whether the game completed successfully.
            True for normal completion (agreement or deadline reached).
            False for abnormal termination (errors, rule violations, crashes).
    
    Example:
        >>> result = GameResult(
        ...     game_id="session_2023_001",
        ...     players=["model_a", "model_b"],
        ...     winner="model_a",
        ...     final_scores={"model_a": 8.5, "model_b": 6.2},
        ...     total_rounds=4,
        ...     game_data={"agreement_reached": True, "final_price": 43500},
        ...     success=True
        ... )
        >>> print(result.final_scores["model_a"])
        8.5
    
    Usage:
        GameResult objects are created by game implementations upon completion
        and consumed by the metrics system, reporting tools, and analysis scripts.
        They provide the primary interface between game execution and result analysis.
    """
    game_id: str
    players: List[str]
    winner: Optional[str]
    final_scores: Dict[str, float]
    total_rounds: int
    game_data: Dict[str, Any]
    success: bool

class BaseGame(ABC):
    """
    Abstract base class defining the interface and common infrastructure for all negotiation games.
    
    BaseGame provides the template and shared functionality for all negotiation game types
    within the platform. It implements common game management features while defining
    abstract methods that concrete games must implement for game-specific logic.
    
    Design Pattern:
        Implements the Template Method pattern where this base class handles common
        game lifecycle management (state tracking, action history, round counting)
        while delegating game-specific logic to concrete implementations.
    
    Key Responsibilities:
        - Game state management and lifecycle tracking
        - Player registration and management
        - Action history maintenance and validation
        - Round counting and termination detection
        - Common configuration parameter handling
        - Abstract interface definition for game-specific logic
    
    Attributes:
        game_id (str): Unique identifier for this game instance.
        config (Dict[str, Any]): Game configuration parameters.
        state (GameState): Current game state (WAITING, ACTIVE, COMPLETED, FAILED).
        players (List[str]): List of registered player identifiers.
        current_round (int): Current round number (0-based).
        max_rounds (int): Maximum allowed rounds before forced termination.
        actions_history (List[PlayerAction]): Complete chronological action log.
        game_data (Dict[str, Any]): Game-specific state and data storage.
    
    Abstract Methods:
        Concrete games must implement:
        - initialize_game(): Set up initial game state with players
        - is_valid_action(): Validate player actions against game rules
        - process_action(): Update game state based on valid actions
        - check_end_conditions(): Determine if game should terminate
        - calculate_scores(): Compute final player scores
        - get_game_prompt(): Generate player-specific prompts
    
    Game Lifecycle:
        1. Construction: Game created with configuration
        2. Initialization: Players assigned and game state prepared
        3. Active Phase: Players take turns, actions processed
        4. Termination: End conditions met or max rounds reached
        5. Scoring: Final scores calculated and results prepared
    
    Example:
        >>> class CustomGame(BaseGame):
        ...     def initialize_game(self, players):
        ...         self.players = players
        ...         self.state = GameState.ACTIVE
        ...         return True
        ...     # ... implement other abstract methods
        >>> game = CustomGame("game_1", {"max_rounds": 5})
        >>> success = game.initialize_game(["player1", "player2"])
    
    Note:
        This is an abstract class and cannot be instantiated directly. Use concrete
        implementations like CompanyCarGame, ResourceAllocationGame, etc.
    """

    def __init__(self, game_id: str, config: Dict[str, Any]):
        """
        Initialize a new game instance with identifier and configuration.
        
        Sets up the basic game infrastructure including state tracking, player
        management, and configuration storage. The game starts in WAITING state
        until players are assigned via initialize_game().
        
        Args:
            game_id (str): Unique identifier for this game instance. Used for
                logging, result tracking, and debugging. Should be unique across
                all concurrent game sessions.
            config (Dict[str, Any]): Configuration dictionary containing game
                parameters. Common parameters include:
                    - max_rounds (int): Maximum negotiation rounds (default: 10)
                    - time_limit (int): Session time limit in seconds
                    - game_specific_params: Parameters unique to the game type
        
        Initialization Process:
            1. Stores game identifier and configuration
            2. Sets initial state to WAITING
            3. Initializes empty player list
            4. Resets round counter to 0
            5. Sets up empty action history
            6. Prepares game data dictionary for game-specific state
        
        Example:
            >>> config = {"max_rounds": 5, "time_limit": 1800}
            >>> game = ConcreteGame("negotiation_001", config)
            >>> print(game.state)
            GameState.WAITING
            >>> print(game.max_rounds)
            5
        
        Note:
            After construction, call initialize_game() to assign players and
            transition to ACTIVE state before game play can begin.
        """
        self.game_id = game_id
        self.config = config
        self.state = GameState.WAITING
        self.players: List[str] = []
        self.current_round = 0
        self.max_rounds = config.get('max_rounds', 10)
        self.actions_history: List[PlayerAction] = []
        self.game_data: Dict[str, Any] = {}

    @abstractmethod
    def initialize_game(self, players: List[str]) -> bool:
        """Initialize the game with given players"""
        pass

    @abstractmethod
    def is_valid_action(self, player: str, action: Dict[str, Any], game_state: Dict[str, Any]) -> bool:
        """Check if an action is valid in current game state"""
        pass

    @abstractmethod
    def process_action(self, action: PlayerAction) -> Dict[str, Any]:
        """Process a player action and update game state"""
        pass

    @abstractmethod
    def check_end_conditions(self) -> bool:
        """Check if the game should end"""
        pass

    @abstractmethod
    def calculate_scores(self) -> Dict[str, float]:
        """Calculate final scores for all players"""
        pass

    @abstractmethod
    def get_game_prompt(self, player_id: str) -> str:
        """Get the current game prompt for a specific player"""
        pass

    def add_action(self, action: PlayerAction):
        """
        Add a player action to the game's chronological history log.
        
        Maintains a complete record of all player actions taken during the
        negotiation session for analysis, metrics calculation, and debugging.
        
        Args:
            action (PlayerAction): The player action to add to history.
                Must contain player_id, action_type, action_data, timestamp,
                and round_number for complete action tracking.
        
        Example:
            >>> action = PlayerAction(
            ...     player_id="player1",
            ...     action_type="offer",  
            ...     action_data={"price": 42000},
            ...     timestamp=1609459200.0,
            ...     round_number=3
            ... )
            >>> game.add_action(action)
        """
        self.actions_history.append(action)

    def get_game_info(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive information about the current game state.
        
        Provides a structured overview of the game's current status including
        identifiers, state information, player list, round tracking, and
        configuration parameters. Useful for debugging, logging, and analysis.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - game_id (str): Unique game identifier
                - state (str): Current game state (waiting/active/completed/failed)
                - players (List[str]): List of participating player identifiers
                - current_round (int): Current round number (0-based)
                - max_rounds (int): Maximum rounds before forced termination
                - config (Dict[str, Any]): Complete game configuration parameters
        
        Example:
            >>> info = game.get_game_info()
            >>> print(f"Game {info['game_id']} is in {info['state']} state")
            >>> print(f"Round {info['current_round']}/{info['max_rounds']}")
            Game negotiation_001 is in active state
            Round 3/5
        """
        return {
            "game_id": self.game_id,
            "state": self.state.value,
            "players": self.players,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "config": self.config
        }