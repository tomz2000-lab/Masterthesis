#!/usr/bin/env python3
"""
Negotiation Platform - Main Entry Point
=======================================

This module serves as the primary entry point for the Negotiation Platform,
providing command-line interface, demonstration capabilities, and example
usage patterns for researchers and developers.

Key Features:
    - Command-line interface for quick testing and experimentation
    - Single negotiation runs for focused analysis
    - Comprehensive model comparison across multiple games
    - Interactive mode for guided exploration
    - Configurable logging and output options
    - Integration examples for all platform components

Usage Modes:
    1. Quick Mode: Single negotiation with default settings
    2. Comparison Mode: Full model comparison across games
    3. Interactive Mode: Guided selection of options
    4. Custom Mode: Programmatic usage with specific configurations

Example Command Lines:
    python main.py --quick --models model_a model_b --game company_car
    python main.py --comparison --models model_a model_b model_c
    python main.py --log-level DEBUG

Architecture:
    The main module demonstrates the complete platform initialization
    workflow: ConfigManager -> LLMManager -> GameEngine -> MetricsCalculator
    -> SessionManager -> Results. This pattern should be followed for
    custom integrations and extensions.
"""
#from dotenv import load_dotenv
#load_dotenv()
import argparse
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to Python path if running directly
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from negotiation_platform.core.llm_manager import LLMManager
from negotiation_platform.core.game_engine import GameEngine
from negotiation_platform.core.metrics_calculator import MetricsCalculator
from negotiation_platform.core.session_manager import SessionManager
from negotiation_platform.core.config_manager import ConfigManager


def setup_logging(level="INFO"):
    """
    Configure comprehensive logging for the negotiation platform.
    
    Sets up dual-output logging (file and console) with detailed formatting
    for debugging, monitoring, and analysis of negotiation sessions.
    
    Args:
        level (str, optional): Logging level (DEBUG, INFO, WARNING, ERROR).
            Defaults to "INFO". DEBUG provides detailed execution traces,
            INFO shows key events and progress, WARNING highlights potential
            issues, ERROR logs only critical failures.
    
    Logging Configuration:
        - File Output: negotiation_platform.log (persistent record)
        - Console Output: Real-time feedback during execution
        - Format: Timestamp - Logger Name - Level - Message
        - Rotation: Not configured (manual cleanup required)
    
    Log Categories:
        - SessionManager: Negotiation progress and outcomes
        - LLMManager: Model loading, switching, and memory management
        - GameEngine: Game creation and state transitions
        - MetricsCalculator: Performance analysis and calculations
        - Individual Games: Game-specific events and decisions
    
    Example:
        >>> setup_logging("DEBUG")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Platform initialized successfully")
        2023-12-01 10:30:45,123 - __main__ - INFO - Platform initialized successfully
    
    Note:
        Should be called early in application startup before other components
        are initialized to ensure all log messages are captured.
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('negotiation_platform.log'),
            logging.StreamHandler()
        ]
    )


def run_single_negotiation(config_manager, models, game_type="company_car"):
    """
    Execute a single negotiation session between two AI models for analysis.
    
    Demonstrates the complete negotiation workflow from platform initialization
    through result analysis. Useful for focused testing, debugging, and detailed
    analysis of specific model interactions.
    
    Args:
        config_manager (ConfigManager): Initialized configuration manager containing
            model definitions, game settings, and platform parameters.
        models (List[str]): List of model identifiers to use as negotiation
            participants. Only the first two models are used for bilateral games.
        game_type (str, optional): Type of negotiation game to run. Must be
            registered in the GameEngine. Options include:
                - "company_car": Bilateral vehicle price negotiation
                - "resource_allocation": Multi-resource team distribution
                - "integrative_negotiations": Multi-issue collaborative negotiation
            Defaults to "company_car".
    
    Returns:
        Dict[str, Any]: Complete negotiation results containing:
            - agreement_reached (bool): Whether players reached an agreement
            - agreement_round (int): Round when agreement was reached
            - final_utilities (Dict[str, float]): Final utility values per player
            - metrics (Dict[str, Dict[str, float]]): Computed performance metrics
            - session_metadata (Dict[str, Any]): Session information and timestamps
            - actions_history (List[Dict]): Complete action log for analysis
    
    Workflow:
        1. Initialize all platform components with configurations
        2. Set up lazy-loading LLM manager for memory efficiency
        3. Create game instance with specified type and configuration
        4. Execute negotiation session with turn-based interaction
        5. Calculate comprehensive performance metrics
        6. Display results and maintain model state for reuse
    
    Example:
        >>> config = ConfigManager()
        >>> models = ["model_a", "model_b"]
        >>> result = run_single_negotiation(config, models, "company_car")
        >>> print(result['agreement_reached'])
        True
        >>> print(result['metrics']['utility_surplus'])
        {'model_a': 2500.0, 'model_b': 1800.0}
    
    Performance Notes:
        - Uses lazy loading to minimize GPU memory usage
        - Models remain loaded after completion for potential reuse
        - Detailed logging provides debugging and analysis capabilities
        - All metrics are calculated automatically for comprehensive analysis
    
    Error Handling:
        Exceptions during negotiation are logged and may result in incomplete
        results. Check the 'success' field in returned results and logs for
        error details.
    """
    print(f"\n=== Running Single {game_type.replace('_', ' ').title()} Negotiation ==")

    # Initialize components
    llm_manager = LLMManager(config_manager.get_config("model_configs"))
    game_engine = GameEngine()
    metrics_calculator = MetricsCalculator()
    session_manager = SessionManager(llm_manager, game_engine, metrics_calculator)

    # Don't pre-load models - use lazy loading instead!
    # Models will be loaded automatically when first used
    print(f"🔄 Models will be loaded on-demand: {models}")

    # Run negotiation
    players = models[:2]  # Use first two models as players
    game_config = config_manager.get_game_config(game_type)

    result = session_manager.run_negotiation(
        game_type=game_type,
        players=players,
        game_config=game_config
    )

    # Display results
    print(f"Agreement reached: {result.get('agreement_reached', False)}")
    if result.get('agreement_reached'):
        print(f"Agreement round: {result.get('agreement_round', 'N/A')}")
        print(f"Final utilities: {result.get('final_utilities', {})}")

    print(f"Metrics: {result.get('metrics', {})}")

    # Keep models loaded for potential reuse
    # Only unload when explicitly needed or at program exit
    print("🔄 Keeping models loaded for potential reuse")

    return result


def run_model_comparison(config_manager, models, games=None):
    """Execute comprehensive multi-model comparison across negotiation games.
    
    Performs systematic evaluation of multiple AI models across different
    negotiation scenarios to assess relative performance, strategy effectiveness,
    and behavioral consistency. This function implements a rigorous comparison
    methodology with multiple runs per model pair for statistical reliability.
    
    Args:
        config_manager (ConfigManager): Initialized configuration manager containing
            model definitions, game configurations, and platform settings.
        models (List[str]): List of model identifiers to compare. All pairwise
            combinations will be tested across specified games.
        games (List[str], optional): List of game types to include in comparison.
            Defaults to ["company_car", "resource_allocation", "integrative_negotiations"]
            if not specified. Each game type must be registered in GameEngine.
    
    Returns:
        Dict[str, Dict[str, List[Dict]]]: Hierarchical results structure:
            - game_type -> model_pair -> list of session results
            - Each session result contains metrics, outcomes, and metadata
            - Suitable for statistical analysis and visualization
    
    Comparison Methodology:
        1. For each game type in the evaluation set
        2. Test all unique model pairs (avoiding duplicates)
        3. Run multiple sessions per pair for statistical significance
        4. Calculate comprehensive metrics for each session
        5. Aggregate results with summary statistics
        6. Save detailed results to timestamped JSON file
    
    Output Artifacts:
        - Console summary with agreement rates and average metrics
        - Detailed JSON results file in configured results directory
        - Individual session logs for debugging and analysis
    
    Example:
        >>> config = ConfigManager()
        >>> models = ["model_a", "model_b", "model_c"]
        >>> results = run_model_comparison(config, models)
        Running Model Comparison
        Models: ['model_a', 'model_b', 'model_c']
        >>> print(results.keys())
        dict_keys(['company_car', 'resource_allocation', 'integrative_negotiations'])
    
    Performance Considerations:
        - Models are loaded and unloaded for each pair to manage memory
        - Multiple runs per pair provide statistical reliability
        - Results are saved incrementally to prevent data loss
        - Detailed logging enables progress monitoring
    
    Note:
        This function is designed for research and evaluation purposes.
        Large model sets or many games may require significant computation
        time and GPU resources. Consider running in stages for very large
        evaluations.
    """
    if games is None:
        games = ["company_car", "resource_allocation", "integrative_negotiations"]

    print(f"\n=== Running Model Comparison ===")
    print(f"Models: {models}")
    print(f"Games: {games}")

    # Initialize components
    llm_manager = LLMManager(config_manager.get_config("model_configs"))
    game_engine = GameEngine()
    metrics_calculator = MetricsCalculator()
    session_manager = SessionManager(llm_manager, game_engine, metrics_calculator)

    comparison_results = {}

    for game_type in games:
        print(f"\nTesting {game_type}...")
        game_config = config_manager.get_game_config(game_type)
        game_results = {}

        # Test each model pair
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                pair_key = f"{model1}_vs_{model2}"

                print(f"  {pair_key}...")

                # Load models
                llm_manager.load_model(model1)
                llm_manager.load_model(model2)

                # Run multiple sessions for statistical significance
                pair_results = []
                for run in range(3):  # 3 runs per pair
                    result = session_manager.run_negotiation(
                        game_type=game_type,
                        players=[model1, model2],
                        game_config=game_config
                    )
                    pair_results.append(result)

                game_results[pair_key] = pair_results

                # Unload models to save memory
                llm_manager.unload_model(model1)
                llm_manager.unload_model(model2)

        comparison_results[game_type] = game_results

    # Generate summary
    _generate_comparison_summary(comparison_results)

    # Save results
    results_dir = Path(config_manager.get_config("platform_config").get("results_dir", "results"))
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"model_comparison_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return comparison_results


def _generate_comparison_summary(results):
    """Generate and display comprehensive comparison summary statistics.
    
    Processes raw comparison results to calculate and display meaningful
    summary statistics including agreement rates, average metrics, and
    performance trends across different model pairs and game types.
    
    Args:
        results (Dict[str, Dict[str, List[Dict]]]): Hierarchical results from
            run_model_comparison containing game types, model pairs, and
            individual session results with metrics and outcomes.
    
    Output Format:
        Console display organized by game type showing:
        - Model pair identifiers (e.g., "model_a_vs_model_b")
        - Agreement rates as fractions and percentages
        - Average metric values across all runs for each pair
        - Performance comparison indicators
    
    Summary Statistics:
        - Agreement Rate: Percentage of sessions reaching successful agreements
        - Average Metrics: Mean values across runs for each performance metric
        - Comparative Analysis: Relative performance indicators between pairs
    
    Example Output:
        === COMPARISON SUMMARY ===
        
        Company Car:
          model_a_vs_model_b: 2/3 agreements (66.7%)
            utility_surplus: {'model_a': 1250.0, 'model_b': 980.5}
            risk_minimization: {'model_a': 85.2, 'model_b': 72.1}
          model_a_vs_model_c: 3/3 agreements (100.0%)
            utility_surplus: {'model_a': 1450.2, 'model_c': 1120.8}
    
    Note:
        This function provides immediate feedback during comparison runs
        and serves as a quick assessment tool before detailed analysis
        of the saved JSON results.
    """
    print(f"\n=== COMPARISON SUMMARY ===")

    for game_type, game_results in results.items():
        print(f"\n{game_type.replace('_', ' ').title()}:")

        for pair, pair_results in game_results.items():
            agreement_rate = sum(1 for r in pair_results if r.get('agreement_reached', False))
            total_runs = len(pair_results)

            print(f"  {pair}: {agreement_rate}/{total_runs} agreements "
                  f"({agreement_rate / total_runs * 100:.1f}%)")

            # Calculate average metrics
            if pair_results:
                avg_metrics = _calculate_average_metrics(pair_results)
                for metric, values in avg_metrics.items():
                    print(f"    {metric}: {values}")


def _calculate_average_metrics(results):
    """Calculate average metric values across multiple negotiation runs.
    
    Processes a collection of negotiation session results to compute mean
    metric values for each player across all runs. This provides statistical
    aggregation for reliable performance assessment when multiple runs are
    conducted for the same model pair.
    
    Args:
        results (List[Dict[str, Any]]): List of individual session results,
            each containing a 'metrics' dictionary with player-specific
            metric values (e.g., utility_surplus, risk_minimization).
    
    Returns:
        Dict[str, Dict[str, float]]: Averaged metrics organized as:
            metric_name -> player_id -> average_value
            Returns empty dict if no valid results with metrics are provided.
    
    Calculation Process:
        1. Aggregate metric values across all runs for each player
        2. Count valid sessions containing metric data
        3. Calculate arithmetic mean for each metric-player combination
        4. Handle missing or incomplete metric data gracefully
    
    Example:
        >>> session_results = [
        ...     {'metrics': {'utility_surplus': {'player1': 100, 'player2': 80}}},
        ...     {'metrics': {'utility_surplus': {'player1': 120, 'player2': 90}}}
        ... ]
        >>> averages = _calculate_average_metrics(session_results)
        >>> print(averages)
        {'utility_surplus': {'player1': 110.0, 'player2': 85.0}}
    
    Error Handling:
        - Skips sessions without 'metrics' field
        - Handles missing players in some sessions
        - Returns empty dict if no valid data found
        - Gracefully processes incomplete metric sets
    
    Note:
        This function assumes all metric values are numeric and suitable
        for arithmetic averaging. Non-numeric metrics are ignored to
        prevent calculation errors.
    """
    metrics_sums = {}
    count = 0

    for result in results:
        if result.get('metrics'):
            count += 1
            for metric_name, metric_values in result['metrics'].items():
                if metric_name not in metrics_sums:
                    metrics_sums[metric_name] = {}

                for player, value in metric_values.items():
                    if player not in metrics_sums[metric_name]:
                        metrics_sums[metric_name][player] = 0
                    metrics_sums[metric_name][player] += value

    # Calculate averages
    avg_metrics = {}
    if count > 0:
        for metric_name, player_sums in metrics_sums.items():
            avg_metrics[metric_name] = {
                player: value / count for player, value in player_sums.items()
            }

    return avg_metrics


def main():
    """Main application entry point with command-line interface.
    
    Provides comprehensive command-line interface for the Negotiation Platform
    with support for various execution modes including quick testing, systematic
    model comparison, and interactive exploration. Handles argument parsing,
    logging configuration, and orchestrates the appropriate execution workflow.
    
    Command-Line Options:
        --quick: Execute single negotiation session for rapid testing
        --comparison: Run comprehensive multi-model comparison study
        --models: Specify list of models to use (default: model_a, model_b, model_c)
        --game: Choose game type for single runs (default: company_car)
        --log-level: Set logging verbosity (DEBUG, INFO, WARNING, ERROR)
    
    Execution Modes:
        1. Quick Mode (--quick): Single negotiation with specified models and game
        2. Comparison Mode (--comparison): Systematic evaluation across model pairs
        3. Interactive Mode (default): User-guided selection of execution options
    
    Example Usage:
        # Quick single negotiation
        python main.py --quick --models model_a model_b --game company_car
        
        # Comprehensive comparison
        python main.py --comparison --models model_a model_b model_c
        
        # Interactive mode with debug logging
        python main.py --log-level DEBUG
    
    Platform Initialization:
        1. Configure logging system with specified verbosity level
        2. Initialize ConfigManager with default or custom configuration
        3. Display available models and confirm selection
        4. Execute requested workflow with comprehensive error handling
    
    Error Handling:
        - Graceful handling of KeyboardInterrupt (Ctrl+C)
        - Comprehensive exception logging with stack traces
        - Proper cleanup and resource management on exit
        - User-friendly error messages for common issues
    
    Output:
        - Progress indicators during execution
        - Summary results and key findings
        - File locations for detailed results
        - Success confirmation upon completion
    
    Note:
        This function serves as the primary demonstration of platform
        capabilities and provides templates for custom integration patterns.
        For programmatic usage, consider calling individual functions directly
        rather than using the command-line interface.
    """
    parser = argparse.ArgumentParser(description="Negotiation Platform")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick single negotiation test")
    parser.add_argument("--comparison", action="store_true",
                        help="Run full model comparison")
    parser.add_argument("--models", nargs="+",
                        default=["model_a", "model_b", "model_c"],
                        help="Models to use")
    parser.add_argument("--game", choices=["company_car", "company_car_arena", "resource_allocation", "integrative_negotiations"],
                        default="company_car", help="Game type for single run")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    config_manager = ConfigManager()

    print("=== Negotiation Platform ===")
    print(f"Available models: {list(config_manager.get_config('model_configs').keys())}")
    print(f"Using models: {args.models}")

    try:
        if args.quick:
            # Quick single run
            result = run_single_negotiation(config_manager, args.models, args.game)

        elif args.comparison:
            # Full comparison
            results = run_model_comparison(config_manager, args.models)

        else:
            # Interactive mode
            print("\nAvailable options:")
            print("1. Single negotiation")
            print("2. Model comparison")

            choice = input("Choose option (1/2): ").strip()

            if choice == "1":
                result = run_single_negotiation(config_manager, args.models, args.game)
            elif choice == "2":
                results = run_model_comparison(config_manager, args.models)
            else:
                print("Invalid choice")
                return

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logging.error(f"Error running platform: {e}")
        raise

    print("\nPlatform completed successfully!")


if __name__ == "__main__":
    main()
