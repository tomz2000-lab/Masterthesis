#!/usr/bin/env python3
"""
Batch Runner for Multiple Game Iterations
=========================================

Runs multiple iterations of negotiation games and collects aggregated metrics
for model comparison and performance analysis.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from negotiation_platform.core.llm_manager import LLMManager
from negotiation_platform.core.game_engine import GameEngine
from negotiation_platform.core.metrics_calculator import MetricsCalculator
from negotiation_platform.core.session_manager import SessionManager
from negotiation_platform.core.config_manager import ConfigManager


@dataclass
class BatchResult:
    """Container for batch run results"""
    model_pair: str
    game_type: str
    iterations: int
    success_rate: float
    aggregated_metrics: Dict[str, Any]
    individual_results: List[Dict[str, Any]]
    timestamp: str


class BatchRunner:
    """Runs multiple game iterations and aggregates results for model comparison"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.llm_manager = LLMManager(config_manager.get_config("model_configs"))
        self.game_engine = GameEngine()
        self.metrics_calculator = MetricsCalculator()
        self.session_manager = SessionManager(
            self.llm_manager, 
            self.game_engine, 
            self.metrics_calculator
        )
        
    def run_batch(
        self, 
        game_type: str,
        model_a: str,
        model_b: str,
        iterations: int = 10,
        game_config: Optional[Dict[str, Any]] = None
    ) -> BatchResult:
        """
        Run multiple iterations of a game between two models
        
        Args:
            game_type: Type of game to run (e.g., "company_car", "resource_allocation")
            model_a: Name of first model
            model_b: Name of second model
            iterations: Number of iterations to run
            game_config: Optional game configuration override
            
        Returns:
            BatchResult containing aggregated metrics and individual results
        """
        self.logger.info(f"Starting batch run: {game_type} with {model_a} vs {model_b} ({iterations} iterations)")
        
        individual_results = []
        successful_runs = 0
        
        # Use game config from config manager if not provided
        if game_config is None:
            game_config = self.config_manager.get_config("game_configs").get(game_type, {})
        
        for iteration in range(iterations):
            self.logger.info(f"Running iteration {iteration + 1}/{iterations}")
            
            try:
                # Run single negotiation
                result = self.session_manager.run_negotiation(
                    game_type=game_type,
                    players=[model_a, model_b],
                    game_config=game_config,
                    session_id=f"batch_{game_type}_{model_a}_vs_{model_b}_iter_{iteration+1}"
                )
                
                # Extract key information
                individual_result = {
                    "iteration": iteration + 1,
                    "agreement_reached": result.get("agreement_reached", False),
                    "agreement_round": result.get("agreement_round", None),
                    "final_utilities": result.get("final_utilities", {}),
                    "metrics": result.get("metrics", {}),
                    "batnas_at_agreement": result.get("batnas_at_agreement", {}),
                    "model_a": model_a,
                    "model_b": model_b
                }
                
                individual_results.append(individual_result)
                
                if result.get("agreement_reached", False):
                    successful_runs += 1
                    
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration + 1}: {e}")
                # Add failed result
                individual_results.append({
                    "iteration": iteration + 1,
                    "agreement_reached": False,
                    "error": str(e),
                    "model_a": model_a,
                    "model_b": model_b
                })
        
        # Calculate aggregated metrics
        success_rate = successful_runs / iterations
        aggregated_metrics = self._aggregate_metrics(individual_results)
        
        # Create batch result
        batch_result = BatchResult(
            model_pair=f"{model_a}_vs_{model_b}",
            game_type=game_type,
            iterations=iterations,
            success_rate=success_rate,
            aggregated_metrics=aggregated_metrics,
            individual_results=individual_results,
            timestamp=datetime.now().isoformat()
        )
        
        self.logger.info(f"Batch completed: {successful_runs}/{iterations} successful runs ({success_rate:.1%})")
        
        return batch_result
    
    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across multiple runs"""
        
        # Filter successful runs
        successful_results = [r for r in results if r.get("agreement_reached", False)]
        
        if not successful_results:
            return {
                "success_rate": 0.0,
                "mean_metrics": {},
                "std_metrics": {},
                "model_performance": {}
            }
        
        # Extract metrics for aggregation
        utility_surplus_a = []
        utility_surplus_b = []
        risk_minimization_a = []
        risk_minimization_b = []
        deadline_sensitivity_a = []
        deadline_sensitivity_b = []
        feasibility_a = []
        feasibility_b = []
        agreement_rounds = []
        
        for result in successful_results:
            metrics = result.get("metrics", {})
            
            # Utility surplus
            utility_surplus = metrics.get("utility_surplus", {})
            model_a = result.get("model_a")
            model_b = result.get("model_b")
            
            if model_a in utility_surplus:
                utility_surplus_a.append(utility_surplus[model_a])
            if model_b in utility_surplus:
                utility_surplus_b.append(utility_surplus[model_b])
            
            # Risk minimization
            risk_min = metrics.get("risk_minimization", {})
            if model_a in risk_min:
                risk_minimization_a.append(risk_min[model_a])
            if model_b in risk_min:
                risk_minimization_b.append(risk_min[model_b])
            
            # Deadline sensitivity
            deadline_sens = metrics.get("deadline_sensitivity", {})
            if model_a in deadline_sens:
                deadline_sensitivity_a.append(deadline_sens[model_a])
            if model_b in deadline_sens:
                deadline_sensitivity_b.append(deadline_sens[model_b])
            
            # Feasibility
            feasibility = metrics.get("feasibility", {})
            if model_a in feasibility:
                feasibility_a.append(feasibility[model_a])
            if model_b in feasibility:
                feasibility_b.append(feasibility[model_b])
            
            # Agreement round
            if result.get("agreement_round"):
                agreement_rounds.append(result["agreement_round"])
        
        # Calculate means and standard deviations
        def safe_stats(values):
            if not values:
                return {"mean": 0.0, "std": 0.0, "count": 0}
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values)
            }
        
        aggregated = {
            "success_rate": len(successful_results) / len(results),
            "mean_agreement_round": float(np.mean(agreement_rounds)) if agreement_rounds else 0.0,
            "std_agreement_round": float(np.std(agreement_rounds)) if agreement_rounds else 0.0,
            
            # Model A performance
            "model_a_performance": {
                "utility_surplus": safe_stats(utility_surplus_a),
                "risk_minimization": safe_stats(risk_minimization_a),
                "deadline_sensitivity": safe_stats(deadline_sensitivity_a),
                "feasibility": safe_stats(feasibility_a)
            },
            
            # Model B performance
            "model_b_performance": {
                "utility_surplus": safe_stats(utility_surplus_b),
                "risk_minimization": safe_stats(risk_minimization_b),
                "deadline_sensitivity": safe_stats(deadline_sensitivity_b),
                "feasibility": safe_stats(feasibility_b)
            },
            
            # Comparison metrics
            "performance_comparison": {
                "utility_surplus_ratio": (
                    float(np.mean(utility_surplus_a)) / float(np.mean(utility_surplus_b))
                    if utility_surplus_b and np.mean(utility_surplus_b) != 0 else 0.0
                ),
                "mean_utility_difference": (
                    float(np.mean(utility_surplus_a)) - float(np.mean(utility_surplus_b))
                    if utility_surplus_a and utility_surplus_b else 0.0
                )
            }
        }
        
        return aggregated
    
    def run_model_comparison(
        self,
        models: List[str],
        games: List[str],
        iterations: int = 10
    ) -> Dict[str, Dict[str, BatchResult]]:
        """
        Run comprehensive model comparison across multiple games
        
        Args:
            models: List of model names to compare
            games: List of game types to test
            iterations: Number of iterations per game/model combination
            
        Returns:
            Nested dict: {game_type: {model_pair: BatchResult}}
        """
        results = {}
        
        for game_type in games:
            results[game_type] = {}
            
            # Test all model pairs
            for i, model_a in enumerate(models):
                for j, model_b in enumerate(models):
                    if i < j:  # Avoid duplicate pairs and self-comparison
                        self.logger.info(f"Testing {model_a} vs {model_b} on {game_type}")
                        
                        batch_result = self.run_batch(
                            game_type=game_type,
                            model_a=model_a,
                            model_b=model_b,
                            iterations=iterations
                        )
                        
                        results[game_type][f"{model_a}_vs_{model_b}"] = batch_result
        
        return results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save batch results to JSON file"""
        if filename is None:
            filename = f"batch_results_{self._get_timestamp()}.json"
        
        # Convert BatchResult objects to dict for JSON serialization
        json_results = {}
        for game_type, game_results in results.items():
            json_results[game_type] = {}
            for model_pair, batch_result in game_results.items():
                if isinstance(batch_result, BatchResult):
                    json_results[game_type][model_pair] = {
                        "model_pair": batch_result.model_pair,
                        "game_type": batch_result.game_type,
                        "iterations": batch_result.iterations,
                        "success_rate": batch_result.success_rate,
                        "aggregated_metrics": batch_result.aggregated_metrics,
                        "individual_results": batch_result.individual_results,
                        "timestamp": batch_result.timestamp
                    }
                else:
                    json_results[game_type][model_pair] = batch_result
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")
        
    def create_summary_report(self, results: Dict[str, Dict[str, BatchResult]]) -> str:
        """Create a summary report of batch results"""
        report = []
        report.append("=" * 80)
        report.append("MODEL COMPARISON SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for game_type, game_results in results.items():
            report.append(f"\n{game_type.upper().replace('_', ' ')} GAME RESULTS:")
            report.append("-" * 50)
            
            for model_pair, batch_result in game_results.items():
                if isinstance(batch_result, BatchResult):
                    br = batch_result
                else:
                    continue
                    
                report.append(f"\n{model_pair}:")
                report.append(f"  Success Rate: {br.success_rate:.1%} ({br.aggregated_metrics.get('success_rate', 0):.1%})")
                report.append(f"  Iterations: {br.iterations}")
                
                # Model A performance
                model_a_perf = br.aggregated_metrics.get("model_a_performance", {})
                model_b_perf = br.aggregated_metrics.get("model_b_performance", {})
                
                report.append(f"  Model A Utility Surplus: {model_a_perf.get('utility_surplus', {}).get('mean', 0):.2f} ± {model_a_perf.get('utility_surplus', {}).get('std', 0):.2f}")
                report.append(f"  Model B Utility Surplus: {model_b_perf.get('utility_surplus', {}).get('mean', 0):.2f} ± {model_b_perf.get('utility_surplus', {}).get('std', 0):.2f}")
                
                comparison = br.aggregated_metrics.get("performance_comparison", {})
                report.append(f"  Utility Surplus Ratio (A/B): {comparison.get('utility_surplus_ratio', 0):.2f}")
                report.append(f"  Mean Agreement Round: {br.aggregated_metrics.get('mean_agreement_round', 0):.1f}")
        
        return "\n".join(report)


def main():
    """Example usage of BatchRunner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run batch negotiations for model comparison")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per game/model combination")
    parser.add_argument("--models", nargs="+", default=["model_a", "model_b"], help="Models to compare")
    parser.add_argument("--games", nargs="+", default=["company_car", "resource_allocation", "integrative_negotiations"], help="Games to test")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Create batch runner
    batch_runner = BatchRunner(config_manager)
    
    # Run model comparison
    results = batch_runner.run_model_comparison(
        models=args.models,
        games=args.games,
        iterations=args.iterations
    )
    
    # Save results
    batch_runner.save_results(results, args.output)
    
    # Print summary
    summary = batch_runner.create_summary_report(results)
    print(summary)


if __name__ == "__main__":
    main()