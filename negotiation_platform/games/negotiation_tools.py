"""
Reusable negotiation tools for all game types.
Import and use these functions/classes in your game logic and prompt generation.
"""

from typing import Dict, Any, List, Optional

def calculate_percentage_difference(a: float, b: float) -> float:
    """Returns the percentage difference between two values."""
    if b == 0:
        return 0.0
    return abs(a - b) / abs(b) * 100

def is_offer_above_batna(offer: float, batna: float) -> bool:
    """Checks if an offer is above (or equal to) the BATNA."""
    return offer >= batna

def is_offer_below_batna(offer: float, batna: float) -> bool:
    """Checks if an offer is below (or equal to) the BATNA."""
    return offer <= batna

def suggest_next_offer(current_offer: float, batna: float, last_rejected: bool, step: float = 0.05) -> float:
    """
    Suggests a next offer moving towards BATNA.
    If last offer was rejected, move closer to BATNA by a step (default 5%).
    """
    if last_rejected:
        direction = 1 if batna > current_offer else -1
        return current_offer + direction * abs(batna - current_offer) * step
    return current_offer

def check_constraints(offer: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    """
    Checks if an offer satisfies all constraints.
    Constraints should be a dict of functions or lambdas that return True/False.
    Example:
        constraints = {
            "min_gpu": lambda offer: offer["gpu_hours"] >= 5,
            "max_total": lambda offer: offer["gpu_hours"] + offer["bandwidth"] <= 100
        }
    """
    return all(check(offer) for check in constraints.values())

def calculate_utility(offer: Dict[str, Any], weights: Dict[str, float], points: Dict[str, Dict[Any, float]]) -> float:
    """
    Generic utility calculation for multi-issue games.
    weights: importance of each issue for the player
    points: mapping from issue to {option: point value}
    offer: dict of issue: chosen option
    """
    utility = 0.0
    for issue, value in offer.items():
        issue_weight = weights.get(issue, 0.0)
        issue_points = points.get(issue, {})
        utility += issue_weight * issue_points.get(value, 0.0)
    return utility
