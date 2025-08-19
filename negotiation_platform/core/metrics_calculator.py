"""
Metrics Calculator - Main component for calculating all negotiation metrics
"""
from typing import Dict, List, Any, Optional
import importlib
from .base_metric import BaseMetric
from negotiation_platform.games.base_game import GameResult, PlayerAction

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
        """Register a new metric (plug-and-play functionality)"""
        self.metrics[metric_id] = metric
        print(f"📊 Registered metric: {metric_id} - {metric.get_name()}")

    def unregister_metric(self, metric_id: str):
        """Remove a metric"""
        if metric_id in self.metrics:
            del self.metrics[metric_id]
            print(f"🗑️  Unregistered metric: {metric_id}")

    def calculate_all(self, game_state: Dict[str, Any], actions_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate all metrics using game_state and actions_history (session manager interface)"""
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
        """Determine winner from game state"""
        if game_state.get("agreement_reached", False):
            # For bilateral negotiations, both players win if agreement is reached
            return None  # Or could implement game-specific logic
        return None
    
    def _calculate_final_scores(self, game_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate final scores from game state"""
        players = game_state.get("players", [])
        final_utilities = game_state.get("final_utilities", {})
        
        # Use final utilities if available, otherwise default to 0
        scores = {}
        for player in players:
            scores[player] = final_utilities.get(player, 0.0)
        
        return scores

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