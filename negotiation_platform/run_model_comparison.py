#!/usr/bin/env python3
"""
Run Model Comparison
===================

Simple script to compare Llama 3B vs 8B models across all three games
with multiple iterations for statistical significance.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from batch_runner import BatchRunner
from negotiation_platform.core.config_manager import ConfigManager


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_comparison.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Run comprehensive model comparison"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model comparison study")
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Create batch runner
    batch_runner = BatchRunner(config_manager)
    
    # Define models to compare (based on your current config)
    models = ["model_a", "model_b"]  # 3B vs 8B Llama models
    
    # Define games to test
    games = [
        "company_car",
        "resource_allocation", 
        "integrative_negotiations"
    ]
    
    # Number of iterations per game/model combination
    iterations = 10  # Adjust as needed
    
    logger.info(f"Testing models: {models}")
    logger.info(f"Testing games: {games}")
    logger.info(f"Iterations per combination: {iterations}")
    
    # Run comprehensive comparison
    results = batch_runner.run_model_comparison(
        models=models,
        games=games,
        iterations=iterations
    )
    
    # Save detailed results
    timestamp = batch_runner._get_timestamp()
    results_file = f"model_comparison_results_{timestamp}.json"
    batch_runner.save_results(results, results_file)
    
    # Create and save summary report
    summary = batch_runner.create_summary_report(results)
    summary_file = f"model_comparison_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    # Print summary to console
    print("\n" + summary)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("Model comparison completed!")


if __name__ == "__main__":
    main()