"""
Results Analysis Package
========================

This package contains statistical analysis tools for negotiation game results,
including bias detection, comparative analysis, and statistical testing.

Modules:
    compare_games_statistics_FIXED: Main statistical analysis functions for
        parsing negotiation logs and detecting various types of bias.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main functions for easy access
try:
    from .compare_games_statistics_FIXED import (
        parse_negotiation_log_corrected,
        analyze_role_bias,
        analyze_first_move_advantage,
        analyze_model_bias,
        perform_comprehensive_analysis
    )
except ImportError:
    # Handle case where dependencies might not be available
    pass