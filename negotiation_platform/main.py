#!/usr/bin/env python3
"""
Main entry point for the Negotiation Platform.
Demonstrates usage and provides example runs.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from negotiation_platform.core.llm_manager import LLMManager
from core.game_engine import GameEngine
from core.metrics_calculator import MetricsCalculator
from core.session_manager import SessionManager
from core.config_manager import ConfigManager


def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('negotiation_platform.log'),
            logging.StreamHandler()
        ]
    )


def run_single_negotiation(config_manager, models, game_type="company_car"):
    """Run a single negotiation example."""
    print(f"\n=== Running Single {game_type.replace('_', ' ').title()} Negotiation ===")

    # Initialize components
    llm_manager = LLMManager(config_manager.get_config("model_configs"))
    game_engine = GameEngine()
    metrics_calculator = MetricsCalculator()
    session_manager = SessionManager(llm_manager, game_engine, metrics_calculator)

    # Load models
    for model_name in models:
        llm_manager.load_model(model_name)

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

    # Cleanup
    llm_manager.unload_all_models()

    return result


def run_model_comparison(config_manager, models, games=None):
    """Run systematic comparison between models."""
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
    """Generate and print comparison summary."""
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
    """Calculate average metrics across runs."""
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
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Negotiation Platform")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick single negotiation test")
    parser.add_argument("--comparison", action="store_true",
                        help="Run full model comparison")
    parser.add_argument("--models", nargs="+",
                        default=["model_a", "model_b", "model_c"],
                        help="Models to use")
    parser.add_argument("--game", choices=["company_car", "resource_allocation", "integrative_negotiations"],
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
