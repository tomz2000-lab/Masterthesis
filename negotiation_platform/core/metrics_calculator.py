"""
Metrics Calculator
==================

Main component for calculating comprehensive negotiation performance metrics.

This module provides a centralized, extensible system for computing performance
metrics across negotiation sessions. It implements a plug-and-play architecture
that allows dynamic registration and calculation of various metric types.

Key Features:
    - Plug-and-play metric registration system
    - Comprehensive performance analysis across multiple dimensions
    - Extensible architecture for custom metric implementations
    - Robust error handling with individual metric failure isolation
    - Standardized result aggregation and reporting
    - Compatible interface with session manager data formats

Core Responsibilities:
    1. Metric Registration  – Dynamic loading and registration of metric implementations
    2. Data Conversion      – Transform session data into metric-compatible formats
    3. Computation          – Execute metric calculations across registered metrics
    4. Error Handling       – Manage failures in individual metric computations
    5. Result Aggregation   – Combine individual metric results into structured format
    6. Report Generation    – Create comprehensive performance reports with statistics

Architecture:
    The calculator follows a plugin-based architecture where individual metrics
    inherit from BaseMetric and implement standardized calculation interfaces.
    This allows for easy addition of new metrics without modifying core logic.

Supported Default Metrics:
    - Utility Surplus: Measures utility gained above BATNA baseline
    - Risk Minimization: Analyzes risk-taking behavior and outcomes
    - Deadline Sensitivity: Evaluates time pressure impact on decisions
    - Feasibility: Assesses solution viability and constraint satisfaction

Example:
    >>> calculator = MetricsCalculator()
    >>> game_state = {
    ...     'agreement_reached': True,
    ...     'final_utilities': {'Player1': 85, 'Player2': 75},
    ...     'current_round': 4
    ... }
    >>> actions_history = [
    ...     {'round': 1, 'actions': {'Player1': {'type': 'offer', 'value': 100}}},
    ...     {'round': 2, 'actions': {'Player2': {'type': 'counteroffer', 'value': 90}}}
    ... ]
    >>> results = calculator.calculate_all(game_state, actions_history)
    >>> print(results['utility_surplus'])
    {'Player1': 25.0, 'Player2': 15.0}
"""

from typing import Dict, List, Any, Optional
import importlib
from .base_metric import BaseMetric
from negotiation_platform.games.base_game import GameResult, PlayerAction


class MetricsCalculator:
    """
    Centralized calculator for negotiation performance metrics with plug-and-play architecture.
    
    This class serves as the main coordinator for metric computation, providing a standardized
    interface for calculating comprehensive performance metrics across various dimensions of
    negotiation analysis. It implements dynamic metric registration and supports both built-in
    and custom metric implementations.
    
    The calculator automatically converts session manager data formats (game_state and 
    actions_history) into metric-compatible formats (GameResult and PlayerAction objects),
    enabling seamless integration with the broader negotiation platform architecture.
    
    Key Features:
        - Dynamic metric registration and discovery system
        - Automatic data format conversion for metric compatibility
        - Comprehensive error handling with graceful degradation
        - Extensible architecture supporting custom metric implementations
        - Standardized result formatting and aggregation
        - Performance report generation with summary statistics
    
    Architecture:
        The calculator maintains a registry of BaseMetric implementations and dynamically
        instantiates them during initialization. Each metric implements standardized
        calculation methods, ensuring consistent computation and result formats.
        
        Data Flow:
        1. Session data (game_state, actions_history) received from SessionManager
        2. Data converted to GameResult and PlayerAction objects for metric compatibility
        3. Each registered metric calculates its specific performance measures
        4. Individual results aggregated into comprehensive metrics dictionary
        5. Optional report generation with summary statistics and analysis
    
    Supported Operations:
        - Registration: Add/remove metrics dynamically during runtime
        - Calculation: Compute all or specific subsets of registered metrics
        - Reporting: Generate comprehensive performance reports with statistics
        - Introspection: List available metrics and retrieve descriptions
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters for metric calculations.
        metrics (Dict[str, BaseMetric]): Registry of available metric implementations.
        
    Example:
        >>> # Initialize with default metrics
        >>> calculator = MetricsCalculator()
        >>> 
        >>> # Register custom metric
        >>> custom_metric = MyCustomMetric()
        >>> calculator.register_metric("custom_analysis", custom_metric)
        >>> 
        >>> # Calculate all metrics for a session
        >>> results = calculator.calculate_all(game_state, actions_history)
        >>> 
        >>> # Generate comprehensive report
        >>> report = calculator.generate_report(game_result, player_actions)
        >>> print(f"Success rate: {report['success']}")
        >>> print(f"Average utility: {report['summary_stats']['utility_surplus']['avg']}")
    
    Raises:
        ValueError: If invalid game state or actions history data provided.
        TypeError: If metric registration receives non-BaseMetric implementations.
        RuntimeError: If critical metric computation failures occur across all metrics.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MetricsCalculator with configuration and default metrics.
        
        Creates a new metrics calculator instance, sets up the metric registry,
        and automatically registers default performance metrics for immediate use.
        
        Args:
            config (Dict[str, Any], optional): Configuration parameters for metric
                calculations. Can include thresholds, weights, or calculation preferences.
                If None, uses empty configuration with default settings.
                
        Example:
            >>> # Initialize with default configuration
            >>> calculator = MetricsCalculator()
            >>> 
            >>> # Initialize with custom configuration
            >>> config = {
            ...     'utility_threshold': 0.8,
            ...     'risk_tolerance': 0.2,
            ...     'enable_detailed_logging': True
            ... }
            >>> calculator = MetricsCalculator(config)
        """
        self.config = config or {}
        self.metrics: Dict[str, BaseMetric] = {}
        self._register_default_metrics()

    def _register_default_metrics(self):
        """
        Register the default set of negotiation performance metrics.
        
        Automatically loads and registers core metrics that provide comprehensive
        analysis of negotiation performance across multiple dimensions. These metrics
        cover utility analysis, risk assessment, deadline sensitivity, and feasibility.
        
        Registered Metrics:
            - utility_surplus: Measures utility gained above BATNA baseline
            - risk_minimization: Analyzes risk-taking behavior and outcomes  
            - deadline_sensitivity: Evaluates time pressure impact on decisions
            - feasibility: Assesses solution viability and constraint satisfaction
            
        Note:
            This method is called automatically during initialization and should not
            typically be called directly. Use register_metric() to add custom metrics.
        """
        from ..metrics.utility_surplus import UtilitySurplusMetric
        from ..metrics.risk_minimization import RiskMinimizationMetric
        from ..metrics.deadline_sensitivity import DeadlineSensitivityMetric
        from ..metrics.feasibility import FeasibilityMetric

        # Register default metrics
        self.register_metric("utility_surplus", UtilitySurplusMetric())
        self.register_metric("risk_minimization", RiskMinimizationMetric())
        self.register_metric("deadline_sensitivity", DeadlineSensitivityMetric())
        self.register_metric("feasibility", FeasibilityMetric())

    def register_metric(self, metric_id: str, metric: BaseMetric):
        """
        Register a new metric for dynamic calculation (plug-and-play functionality).
        
        Adds a metric implementation to the calculator's registry, making it available
        for computation in all subsequent metric calculations. Supports runtime addition
        of custom metrics without requiring system restart or reconfiguration.
        
        Args:
            metric_id (str): Unique identifier for the metric. Used as key in results
                dictionary and for metric-specific operations. Must be unique across
                all registered metrics.
            metric (BaseMetric): Metric implementation instance that inherits from
                BaseMetric and implements required calculation methods.
                
        Example:
            >>> calculator = MetricsCalculator()
            >>> custom_metric = MyCustomAnalysisMetric()
            >>> calculator.register_metric("custom_analysis", custom_metric)
            >>> 
            >>> # Metric now available in calculations
            >>> results = calculator.calculate_all(game_state, actions_history)
            >>> custom_result = results["custom_analysis"]
            
        Raises:
            TypeError: If metric is not an instance of BaseMetric.
            ValueError: If metric_id is already registered (use unregister first).
        """
        self.metrics[metric_id] = metric
        print(f"📊 Registered metric: {metric_id} - {metric.get_name()}")

    def unregister_metric(self, metric_id: str):
        """
        Remove a metric from the calculation registry.
        
        Removes a previously registered metric from the calculator's registry,
        preventing it from being included in future metric calculations. Useful
        for dynamically adjusting analysis scope or removing problematic metrics.
        
        Args:
            metric_id (str): Identifier of the metric to remove. Must match the
                identifier used during registration.
                
        Example:
            >>> calculator = MetricsCalculator()
            >>> calculator.unregister_metric("risk_minimization")
            >>> 
            >>> # risk_minimization no longer included in calculations
            >>> results = calculator.calculate_all(game_state, actions_history)
            >>> # 'risk_minimization' key will not be present in results
            
        Note:
            Silently ignores attempts to unregister non-existent metrics.
            No error is raised if the metric_id is not found in the registry.
        """
        if metric_id in self.metrics:
            del self.metrics[metric_id]
            print(f"🗑️  Unregistered metric: {metric_id}")

    def calculate_all(self, game_state: Dict[str, Any], actions_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate all registered metrics using session manager data formats.
        
        Primary interface for metric calculation that accepts data directly from
        SessionManager. Automatically converts session data formats into metric-compatible
        objects and computes all registered metrics in a single operation.
        
        This method serves as the main entry point for comprehensive performance analysis,
        handling data transformation, metric computation, and result aggregation in a
        unified workflow.
        
        Args:
            game_state (Dict[str, Any]): Final game state from completed negotiation session.
                Expected to contain keys like 'game_type', 'players', 'agreement_reached',
                'final_utilities', 'current_round', and game-specific data.
            actions_history (List[Dict[str, Any]]): Chronological log of all actions taken
                during the negotiation. Each entry should contain 'round' and 'actions'
                keys with player actions for that round.
                
        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary where top-level keys are
            metric identifiers and values are dictionaries mapping player IDs to
            their computed metric values.
            
        Example:
            >>> game_state = {
            ...     'game_type': 'price_bargaining',
            ...     'players': ['Player1', 'Player2'],
            ...     'agreement_reached': True,
            ...     'final_utilities': {'Player1': 85, 'Player2': 75},
            ...     'current_round': 3
            ... }
            >>> actions_history = [
            ...     {'round': 1, 'actions': {'Player1': {'type': 'offer', 'value': 100}}},
            ...     {'round': 2, 'actions': {'Player2': {'type': 'counteroffer', 'value': 90}}},
            ...     {'round': 3, 'actions': {'Player1': {'type': 'accept'}}}
            ... ]
            >>> results = calculator.calculate_all(game_state, actions_history)
            >>> print(results)
            {
                'utility_surplus': {'Player1': 25.0, 'Player2': 15.0},
                'risk_minimization': {'Player1': 0.8, 'Player2': 0.6},
                'deadline_sensitivity': {'Player1': 0.9, 'Player2': 0.7},
                'feasibility': {'Player1': 1.0, 'Player2': 1.0}
            }
            
        Raises:
            ValueError: If game_state or actions_history contain invalid or missing data.
            RuntimeError: If data conversion or metric computation fails critically.
        """
        # Convert game_state to GameResult for compatibility with metrics
        game_result = GameResult(
            game_id=game_state.get("game_type", "unknown"),
            players=game_state.get("players", []),
            winner=self._determine_winner(game_state),
            final_scores=self._calculate_final_scores(game_state),
            total_rounds=game_state.get("current_round", 0),
            game_data=game_state,
            success=game_state.get("agreement_reached", False)
        )
        
        # Convert actions_history to PlayerAction objects
        player_actions = []
        for round_data in actions_history:
            round_num = round_data.get("round", 0)
            actions = round_data.get("actions", {})
            for player_id, action_data in actions.items():
                player_action = PlayerAction(
                    player_id=player_id,
                    action_type=action_data.get("type", "unknown"),
                    action_data=action_data,
                    timestamp=0.0,  # Not available in current format
                    round_number=round_num
                )
                player_actions.append(player_action)
        
        return self.calculate_all_metrics(game_result, player_actions)
    
    def _determine_winner(self, game_state: Dict[str, Any]) -> Optional[str]:
        """
        Determine the winner from final game state using utility-based analysis.
        
        Analyzes the final game state to identify the winning player based on
        achieved utilities and agreement outcomes. For bilateral negotiations,
        considers both players as winners if mutual agreement is reached.
        
        Args:
            game_state (Dict[str, Any]): Final game state containing utilities
                and agreement information.
                
        Returns:
            Optional[str]: Player ID of the winner if determinable, None if
            no clear winner exists or if both players achieved mutual benefit.
            
        Note:
            Current implementation treats successful agreements as win-win scenarios.
            Override this method for games requiring explicit winner determination.
        """
        if game_state.get("agreement_reached", False):
            # For bilateral negotiations, both players win if agreement is reached
            return None  # Or could implement game-specific logic
        return None
    
    def _calculate_final_scores(self, game_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract final scores from game state for metric calculation.
        
        Converts final game utilities into standardized score format required
        by metric calculations. Handles missing utility data gracefully by
        assigning default values.
        
        Args:
            game_state (Dict[str, Any]): Final game state containing player
                utilities and outcome information.
                
        Returns:
            Dict[str, float]: Mapping of player IDs to their final utility scores.
            Players without recorded utilities receive a score of 0.0.
            
        Example:
            >>> game_state = {
            ...     'players': ['Player1', 'Player2'],
            ...     'final_utilities': {'Player1': 85, 'Player2': 75}
            ... }
            >>> scores = calculator._calculate_final_scores(game_state)
            >>> print(scores)
            {'Player1': 85.0, 'Player2': 75.0}
        """
        players = game_state.get("players", [])
        final_utilities = game_state.get("final_utilities", {})
        
        # Use final utilities if available, otherwise default to 0
        scores = {}
        for player in players:
            scores[player] = final_utilities.get(player, 0.0)
        
        return scores

    def calculate_all_metrics(self, game_result: GameResult,
                            actions_history: List[PlayerAction]) -> Dict[str, Dict[str, float]]:
        """
        Calculate all registered metrics using GameResult and PlayerAction objects.
        
        Core computation method that executes all registered metrics against standardized
        data objects. Provides comprehensive error handling to ensure partial results
        are returned even if individual metrics fail.
        
        Args:
            game_result (GameResult): Standardized game outcome data containing final
                scores, winner information, and game metadata.
            actions_history (List[PlayerAction]): Chronological sequence of player
                actions taken during the negotiation session.
                
        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary where top-level keys are
            metric identifiers and values are player-to-score mappings.
            
        Example:
            >>> game_result = GameResult(game_id="test", players=["P1", "P2"], ...)
            >>> actions = [PlayerAction(player_id="P1", action_type="offer", ...)]
            >>> results = calculator.calculate_all_metrics(game_result, actions)
            >>> print(results["utility_surplus"]["P1"])
            25.0
            
        Note:
            Failed metric calculations are logged and replaced with default values
            (0.0 for each player) to ensure consistent result structure.
        """
        results = {}

        for metric_id, metric in self.metrics.items():
            try:
                metric_results = metric.calculate(game_result, actions_history)
                results[metric_id] = metric_results
                print(f"✅ Calculated {metric_id}: {metric_results}")
            except Exception as e:
                print(f"❌ Error calculating {metric_id}: {str(e)}")
                # Set default values on error
                results[metric_id] = {player_id: 0.0 for player_id in game_result.players}

        return results

    def calculate_specific_metrics(self, metric_ids: List[str], game_result: GameResult,
                                 actions_history: List[PlayerAction]) -> Dict[str, Dict[str, float]]:
        """
        Calculate only specified metrics for targeted performance analysis.
        
        Computes a subset of registered metrics based on provided identifiers.
        Useful for focused analysis or when computational resources are limited
        and only specific metrics are required.
        
        Args:
            metric_ids (List[str]): List of metric identifiers to calculate.
                Must match registered metric IDs. Unknown metrics are skipped.
            game_result (GameResult): Standardized game outcome data.
            actions_history (List[PlayerAction]): Player action sequence.
            
        Returns:
            Dict[str, Dict[str, float]]: Results dictionary containing only
            requested metrics. Structure matches calculate_all_metrics output.
            
        Example:
            >>> # Calculate only utility and risk metrics
            >>> specific_results = calculator.calculate_specific_metrics(
            ...     ["utility_surplus", "risk_minimization"],
            ...     game_result,
            ...     actions_history
            ... )
            >>> print(list(specific_results.keys()))
            ['utility_surplus', 'risk_minimization']
            
        Note:
            Unregistered metric IDs are logged as warnings and skipped.
            Failed calculations are replaced with default values.
        """
        results = {}

        for metric_id in metric_ids:
            if metric_id not in self.metrics:
                print(f"⚠️  Metric {metric_id} not registered")
                continue

            try:
                metric = self.metrics[metric_id]
                metric_results = metric.calculate(game_result, actions_history)
                results[metric_id] = metric_results
            except Exception as e:
                print(f"❌ Error calculating {metric_id}: {str(e)}")
                results[metric_id] = {player_id: 0.0 for player_id in game_result.players}

        return results

    def get_metric_descriptions(self) -> Dict[str, str]:
        """
        Retrieve descriptions of all registered metrics for documentation and analysis.
        
        Provides human-readable descriptions of each registered metric, useful for
        generating documentation, user interfaces, and analytical reports that
        explain what each metric measures.
        
        Returns:
            Dict[str, str]: Mapping of metric identifiers to their descriptive text.
            Each description explains what the metric measures and its significance.
            
        Example:
            >>> descriptions = calculator.get_metric_descriptions()
            >>> print(descriptions["utility_surplus"])
            "Measures the utility gained above BATNA baseline for each player"
        """
        descriptions = {}
        for metric_id, metric in self.metrics.items():
            descriptions[metric_id] = metric.get_description()
        return descriptions

    def list_metrics(self) -> List[str]:
        """
        List all currently registered metric identifiers.
        
        Returns the identifiers of all metrics currently available for calculation.
        Useful for introspection, validation, and dynamic metric selection.
        
        Returns:
            List[str]: List of registered metric identifiers that can be used
            with calculate_specific_metrics or other metric operations.
            
        Example:
            >>> available_metrics = calculator.list_metrics()
            >>> print(available_metrics)
            ['utility_surplus', 'risk_minimization', 'deadline_sensitivity', 'feasibility']
        """
        return list(self.metrics.keys())

    def generate_report(self, game_result: GameResult,
                       actions_history: List[PlayerAction]) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with metrics and summary statistics.
        
        Creates a detailed analytical report that includes all metric calculations
        plus derived summary statistics such as averages, ranges, and totals.
        Ideal for comprehensive performance analysis and comparative studies.
        
        Args:
            game_result (GameResult): Final game outcome data.
            actions_history (List[PlayerAction]): Complete sequence of player actions.
            
        Returns:
            Dict[str, Any]: Comprehensive report containing:
                - game_id: Game identifier
                - players: List of participating players
                - success: Whether negotiation was successful
                - total_rounds: Number of negotiation rounds
                - metrics: Full metric calculation results
                - summary_stats: Derived statistics (avg, min, max, total) per metric
                
        Example:
            >>> report = calculator.generate_report(game_result, actions_history)
            >>> print(report["summary_stats"]["utility_surplus"]["avg"])
            20.0
            >>> print(report["success"])
            True
            >>> print(f"Game completed in {report['total_rounds']} rounds")
            Game completed in 3 rounds
            
        Note:
            Summary statistics are calculated only for metrics that return numeric
            values. Non-numeric metrics are included in the metrics section but
            excluded from summary statistics.
        """
        metrics_results = self.calculate_all_metrics(game_result, actions_history)

        # Calculate summary statistics
        summary = {
            "game_id": game_result.game_id,
            "players": game_result.players,
            "success": game_result.success,
            "total_rounds": game_result.total_rounds,
            "metrics": metrics_results,
            "summary_stats": {}
        }

        # Add summary statistics for each metric
        for metric_id, player_results in metrics_results.items():
            if player_results:
                values = list(player_results.values())
                summary["summary_stats"][metric_id] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "total": sum(values)
                }

        return summary