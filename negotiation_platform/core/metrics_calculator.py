"""
Metrics Calculator - Main component for calculating all negotiation metrics
"""
from typing import Dict, List, Any, Optional
import importlib
from .base_metric import BaseMetric
from .base_game import GameResult, PlayerAction

class MetricsCalculator:
    """
    Main metrics calculator with plug-and-play metric support
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics: Dict[str, BaseMetric] = {}
        self._register_default_metrics()

    def _register_default_metrics(self):
        """Register the default metrics"""
        from .utility_surplus import UtilitySurplusMetric
        from .risk_minimization import RiskMinimizationMetric
        from .deadline_sensitivity import DeadlineSensitivityMetric
        from .feasibility import FeasibilityMetric

        # Register default metrics
        self.register_metric("utility_surplus", UtilitySurplusMetric())
        self.register_metric("risk_minimization", RiskMinimizationMetric())
        self.register_metric("deadline_sensitivity", DeadlineSensitivityMetric())
        self.register_metric("feasibility", FeasibilityMetric())

    def register_metric(self, metric_id: str, metric: BaseMetric):
        """Register a new metric (plug-and-play functionality)"""
        self.metrics[metric_id] = metric
        print(f"📊 Registered metric: {metric_id} - {metric.get_name()}")

    def unregister_metric(self, metric_id: str):
        """Remove a metric"""
        if metric_id in self.metrics:
            del self.metrics[metric_id]
            print(f"🗑️  Unregistered metric: {metric_id}")

    def calculate_all_metrics(self, game_result: GameResult,
                            actions_history: List[PlayerAction]) -> Dict[str, Dict[str, float]]:
        """Calculate all registered metrics"""
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
        """Calculate only specific metrics"""
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
        """Get descriptions of all registered metrics"""
        descriptions = {}
        for metric_id, metric in self.metrics.items():
            descriptions[metric_id] = metric.get_description()
        return descriptions

    def list_metrics(self) -> List[str]:
        """List all registered metric IDs"""
        return list(self.metrics.keys())

    def generate_report(self, game_result: GameResult,
                       actions_history: List[PlayerAction]) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
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