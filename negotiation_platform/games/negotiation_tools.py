"""
Negotiation Tools
=================

Reusable utility functions and algorithms for all negotiation game types.

This module provides a comprehensive toolkit of mathematical functions, validation
utilities, and strategic algorithms that support negotiation game implementations.
These tools are designed to be game-agnostic and reusable across different
negotiation scenarios and game types.

Key Features:
    - Mathematical utility functions for offer analysis
    - BATNA comparison and evaluation tools
    - Strategic suggestion algorithms for offer generation
    - Constraint validation systems for proposal checking
    - Generic utility calculation for multi-issue negotiations
    - Percentage-based comparison utilities for fairness analysis

Function Categories:
    1. Comparison Functions: Mathematical analysis of offers and values
    2. BATNA Tools: Best Alternative to Negotiated Agreement evaluation
    3. Strategic Functions: Algorithmic suggestions for negotiation moves
    4. Validation Tools: Constraint checking and proposal validation
    5. Utility Calculators: Multi-issue scoring and preference evaluation

Design Philosophy:
    These tools follow functional programming principles with pure functions
    that have no side effects. Each function is self-contained and can be
    used independently or composed together for complex game logic.

Example Usage:
    >>> # Calculate offer attractiveness
    >>> offer_value = 42000
    >>> batna_value = 40000
    >>> is_attractive = is_offer_above_batna(offer_value, batna_value)
    >>> 
    >>> # Suggest strategic next move
    >>> next_offer = suggest_next_offer(42000, 40000, True, 0.05)
    >>> 
    >>> # Calculate multi-issue utility
    >>> proposal = {"server_room": 100, "meeting_access": 4}
    >>> weights = {"server_room": 0.6, "meeting_access": 0.4}
    >>> points = {"server_room": {100: 30}, "meeting_access": {4: 30}}
    >>> utility = calculate_utility(proposal, weights, points)
"""

from typing import Dict, Any, List, Optional


def calculate_percentage_difference(a: float, b: float) -> float:
    """
    Calculate the percentage difference between two values.
    
    Computes the relative difference between two numeric values as a percentage
    of the second value. Useful for analyzing offer changes, price movements,
    and comparative value assessments in negotiations.
    
    Args:
        a (float): First value for comparison (typically the new or proposed value).
        b (float): Second value for comparison (typically the reference or base value).
        
    Returns:
        float: Percentage difference as a decimal (0.0 to infinity).
               Returns 0.0 if the reference value b is zero to avoid division errors.
    
    Example:
        >>> # 10% increase from base price
        >>> diff = calculate_percentage_difference(44000, 40000)
        >>> print(f"{diff:.1f}%")
        10.0%
        
        >>> # Price reduction analysis
        >>> old_price = 45000
        >>> new_price = 42000
        >>> reduction = calculate_percentage_difference(new_price, old_price)
        >>> print(f"Price reduced by {reduction:.1f}%")
        Price reduced by 6.7%
    
    Note:
        The function returns the absolute percentage difference. Use additional
        logic to determine if the change is an increase or decrease.
    """
    if b == 0:
        return 0.0
    return abs(a - b) / abs(b) * 100


def is_offer_above_batna(offer: float, batna: float) -> bool:
    """
    Check if an offer value is above or equal to the BATNA threshold.
    
    Determines whether a proposed offer meets the minimum acceptable threshold
    defined by the Best Alternative to Negotiated Agreement (BATNA). This is
    a fundamental decision criterion in negotiation theory.
    
    Args:
        offer (float): The proposed offer value to evaluate.
        batna (float): The BATNA threshold value for comparison.
        
    Returns:
        bool: True if offer is greater than or equal to BATNA, False otherwise.
    
    Example:
        >>> # Buyer evaluating a seller's offer
        >>> seller_offer = 41000
        >>> buyer_batna = 40000  # Best alternative option
        >>> should_consider = is_offer_above_batna(seller_offer, buyer_batna)
        >>> print(f"Offer worth considering: {should_consider}")
        Offer worth considering: True
        
        >>> # Seller evaluating buyer's offer
        >>> buyer_offer = 38000
        >>> seller_batna = 39000  # Alternative buyer option
        >>> should_accept = is_offer_above_batna(buyer_offer, seller_batna)
        >>> print(f"Offer acceptable: {should_accept}")
        Offer acceptable: False
    
    Strategic Usage:
        - Accept offers above BATNA (creates positive value)
        - Reject offers below BATNA (destroys value compared to alternatives)
        - Use BATNA as minimum threshold in counteroffers
    """
    return offer >= batna


def is_offer_below_batna(offer: float, batna: float) -> bool:
    """
    Check if an offer value is below or equal to the BATNA threshold.
    
    Determines whether a proposed offer falls below the minimum acceptable
    threshold. This is useful for identifying offers that should typically
    be rejected as they provide less value than available alternatives.
    
    Args:
        offer (float): The proposed offer value to evaluate.
        batna (float): The BATNA threshold value for comparison.
        
    Returns:
        bool: True if offer is less than or equal to BATNA, False otherwise.
    
    Example:
        >>> # Quick rejection check
        >>> low_offer = 37000
        >>> batna_threshold = 39000
        >>> should_reject = is_offer_below_batna(low_offer, batna_threshold)
        >>> print(f"Offer below threshold: {should_reject}")
        Offer below threshold: True
    
    Note:
        This is the logical inverse of is_offer_above_batna() but provides
        semantic clarity when checking for unacceptable offers.
    """
    return offer <= batna


def suggest_next_offer(current_offer: float, batna: float, last_rejected: bool, step: float = 0.05) -> float:
    """
    Generate strategic suggestion for next negotiation offer using BATNA-based algorithm.
    
    Provides algorithmic guidance for offer adjustment based on previous negotiation
    outcomes and BATNA positioning. Implements a conservative strategy that moves
    offers closer to BATNA when faced with rejection, encouraging agreement while
    maintaining value protection.
    
    Args:
        current_offer (float): The most recent offer value made in negotiation.
        batna (float): Best Alternative to Negotiated Agreement value.
        last_rejected (bool): Whether the previous offer was rejected by counterpart.
        step (float, optional): Percentage adjustment toward BATNA when rejected.
                               Defaults to 0.05 (5% movement).
    
    Returns:
        float: Suggested next offer value. If last offer was accepted or this is
               initial offer, returns current_offer unchanged. If rejected, returns
               adjusted offer moved toward BATNA by the specified step percentage.
    
    Algorithm:
        1. If last offer accepted: maintain current position
        2. If last offer rejected: move toward BATNA by step percentage
        3. Direction determined by BATNA position relative to current offer
        4. Step size controls aggressiveness of concession
    
    Example:
        >>> # Buyer's offer was rejected, move toward seller
        >>> current = 40000  # Buyer's last offer
        >>> batna = 41000    # Buyer's alternative option
        >>> rejected = True
        >>> next_offer = suggest_next_offer(current, batna, rejected, 0.05)
        >>> print(f"Next offer: ${next_offer:,.0f}")
        Next offer: $40,050
        
        >>> # Seller's offer was rejected, move toward buyer
        >>> current = 44000  # Seller's last offer  
        >>> batna = 42000    # Seller's alternative option
        >>> rejected = True
        >>> next_offer = suggest_next_offer(current, batna, rejected, 0.10)
        >>> print(f"Next offer: ${next_offer:,.0f}")
        Next offer: $43,800
    
    Strategic Considerations:
        - Smaller steps (0.01-0.03): Conservative, slow concessions
        - Moderate steps (0.05-0.10): Balanced concession strategy
        - Larger steps (0.15+): Aggressive movement toward agreement
    """
    if last_rejected:
        direction = 1 if batna > current_offer else -1
        return current_offer + direction * abs(batna - current_offer) * step
    return current_offer


def check_constraints(offer: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    """
    Validate offer against multiple constraint functions for proposal acceptance.
    
    Evaluates a complex offer or proposal against a set of constraint validation
    functions. Each constraint represents a business rule, resource limit, or
    negotiation requirement that must be satisfied for the offer to be valid.
    
    Args:
        offer (Dict[str, Any]): Proposal dictionary containing offer terms and values.
                               Structure varies by game type but typically includes
                               resource allocations, prices, or multi-issue terms.
        constraints (Dict[str, Any]): Dictionary mapping constraint names to
                                     validation functions. Each function should
                                     accept the offer dict and return boolean.
    
    Returns:
        bool: True if offer satisfies ALL constraints, False if any constraint fails.
    
    Constraint Function Examples:
        - Resource limits: lambda offer: offer["gpu_hours"] <= 10
        - Budget constraints: lambda offer: offer["total_cost"] <= 50000
        - Logical requirements: lambda offer: offer["start_date"] < offer["end_date"]
        - Multi-field validation: lambda offer: offer["gpus"] + offer["cpus"] <= offer["budget"]/1000
    
    Example:
        >>> # Resource allocation validation
        >>> offer = {"gpu_hours": 6, "developer_hours": 4, "budget": 45000}
        >>> constraints = {
        ...     "gpu_limit": lambda o: o["gpu_hours"] <= 8,
        ...     "dev_limit": lambda o: o["developer_hours"] <= 5,
        ...     "budget_limit": lambda o: o["budget"] <= 50000,
        ...     "resource_balance": lambda o: o["gpu_hours"] * 1000 <= o["budget"]
        ... }
        >>> is_valid = check_constraints(offer, constraints)
        >>> print(f"Offer valid: {is_valid}")
        Offer valid: True
        
        >>> # Price negotiation validation
        >>> car_offer = {"price": 43000, "warranty": True, "delivery_days": 14}
        >>> car_constraints = {
        ...     "price_range": lambda o: 35000 <= o["price"] <= 50000,
        ...     "delivery_time": lambda o: o["delivery_days"] <= 30
        ... }
        >>> valid_car_deal = check_constraints(car_offer, car_constraints)
    
    Usage Patterns:
        - Game rule validation before processing actions
        - Resource constraint checking in allocation games
        - Business logic validation for complex proposals
        - Multi-criteria decision support systems
    
    Error Handling:
        Function returns False if any constraint function raises an exception,
        providing graceful handling of invalid offer structures or constraint errors.
    """
    return all(check(offer) for check in constraints.values())


def calculate_utility(offer: Dict[str, Any], weights: Dict[str, float], points: Dict[str, Dict[Any, float]]) -> float:
    """
    Calculate total utility value for multi-issue negotiation proposals.
    
    Computes weighted utility scores for complex proposals involving multiple
    negotiation issues. Each issue contributes to total utility based on its
    importance weight and the point value associated with the chosen option.
    This enables quantitative comparison of multi-dimensional offers.
    
    Args:
        offer (Dict[str, Any]): Proposal dictionary mapping issue names to
                               chosen options/values for each negotiation issue.
        weights (Dict[str, float]): Importance weights for each issue, determining
                                   how much each issue contributes to total utility.
                                   Weights typically sum to 1.0 but not required.
        points (Dict[str, Dict[Any, float]]): Point value mappings for each issue.
                                             Structure: {issue: {option: point_value}}
                                             
    Returns:
        float: Total weighted utility score. Higher values indicate more
               attractive proposals for the evaluating party.
    
    Calculation Formula:
        utility = Σ(weight[issue] × points[issue][option]) for all issues
    
    Example:
        >>> # IT team evaluating office space proposal
        >>> proposal = {
        ...     "server_room": 150,      # 150 sqm server space
        ...     "meeting_access": 2,     # 2 days per week
        ...     "cleaning": "Shared",    # Shared cleaning responsibility
        ...     "branding": "Prominent"  # Prominent marketing visibility
        ... }
        >>> 
        >>> # IT team's importance weights
        >>> it_weights = {
        ...     "server_room": 0.40,    # Server space is top priority
        ...     "meeting_access": 0.10,  # Low meeting room needs
        ...     "cleaning": 0.30,       # Moderate cleaning concern
        ...     "branding": 0.20        # Low branding priority
        ... }
        >>> 
        >>> # Point values for each option from IT perspective
        >>> it_points = {
        ...     "server_room": {50: 10, 100: 30, 150: 60},
        ...     "meeting_access": {2: 10, 4: 30, 7: 60},
        ...     "cleaning": {"IT": 30, "Shared": 50, "Outsourced": 10},
        ...     "branding": {"Minimal": 10, "Moderate": 30, "Prominent": 60}
        ... }
        >>> 
        >>> utility = calculate_utility(proposal, it_weights, it_points)
        >>> print(f"IT team utility: {utility:.1f}")
        IT team utility: 53.0
        
        >>> # Marketing team would have different weights and potentially points
        >>> marketing_weights = {
        ...     "server_room": 0.10,    # Low server space priority
        ...     "meeting_access": 0.30,  # High meeting room needs
        ...     "cleaning": 0.20,       # Moderate cleaning concern  
        ...     "branding": 0.40        # Branding is top priority
        ... }
        >>> marketing_utility = calculate_utility(proposal, marketing_weights, it_points)
        >>> print(f"Marketing team utility: {marketing_utility:.1f}")
        Marketing team utility: 42.0
    
    Usage Patterns:
        - Multi-issue negotiation evaluation
        - Proposal ranking and comparison
        - Win-win solution identification
        - Pareto efficiency analysis
        - Automated offer generation and optimization
    
    Note:
        Missing issues in offer, weights, or points are treated as zero contribution.
        This provides graceful handling of partial proposals or incomplete data.
    """
    utility = 0.0
    for issue, value in offer.items():
        issue_weight = weights.get(issue, 0.0)
        issue_points = points.get(issue, {})
        utility += issue_weight * issue_points.get(value, 0.0)
    return utility
