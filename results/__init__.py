"""
Results and Analysis Package
============================

Statistical analysis and research tools for negotiation data processing and performance evaluation.

This package provides comprehensive tools for analyzing negotiation outcomes, detecting biases,
and comparing model performance across different scenarios.

Key Modules:
    - metrics_statistics: Agent-based metrics analysis and statistical processing
    - win_statistics: Advanced bias analysis and model comparison with statistical testing

Example:
    >>> from results.win_statistics import parse_negotiation_log_corrected
    >>> df, game_type = parse_negotiation_log_corrected("negotiation.out")
    >>> print(f"Analyzed {len(df)} games of type {game_type}")
"""

__version__ = "1.0.0"
__author__ = "Tom Ziegler"

# Import main functions for easy access
try:
    from .win_statistics import (
        parse_negotiation_log_corrected,
        analyze_role_bias_corrected,
        analyze_first_mover_bias_corrected,
        bias_adjusted_model_comparison,
        main
    )
    from .metrics_statistics import (
        parse_negotiation_log_agent_metrics
    )
except ImportError:
    # Handle case where dependencies might not be available
    pass

__all__ = ["metrics_statistics", "win_statistics"]