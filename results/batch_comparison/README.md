# Batch Comparison Output Files

This directory contains experimental output files from batch runs comparing different LLM models in negotiation scenarios.

## Directory Structure

- `car_game/` - Company car negotiation experiments
- `integrative_game/` - Integrative negotiation experiments  
- `resource_game/` - Resource allocation negotiation experiments

## File Naming Convention

Files are named with the pattern: `{game_type}_{timestamp}.out`

- `company_car_*.out` - Buyer-seller car negotiation logs
- `integrative_negotiation_*.out` - IT-Marketing team office space negotiation logs
- `resource_allocation_*.out` - Development-Marketing team resource allocation logs

## Content Description

Each `.out` file contains:

1. **Game Configuration**: Initial settings, BATNAs, utility functions
2. **Multiple Iterations**: Each file contains multiple negotiation rounds between different LLM models
3. **Role Assignments**: Random role assignment to eliminate bias (🎲 markers)
4. **Negotiation Progress**: Round-by-round proposals, acceptances, rejections
5. **Performance Metrics**: 
   - Risk Minimization scores
   - Deadline Sensitivity scores
   - Feasibility scores
   - Utility Surplus calculations
6. **Agreement Analysis**: Final outcomes and agreement statistics

## Usage

These files can be analyzed using the statistical comparison scripts in the `results/` directory:

```bash
python compare_games_statistics.py batch_comparison/integrative_game/integrative_negotiation_2019176.out
```

## Model Comparisons

The experiments compare different LLM models (typically model_a, model_b, model_c) across:

- **Bias Control**: Role randomization and first-mover randomization
- **Statistical Analysis**: Regression-based bias correction
- **Performance Metrics**: Multiple behavioral and outcome measures

## File Sizes

These files are large (typically 50KB-500KB each) due to detailed logging of:
- Complete negotiation transcripts
- Detailed model reasoning
- Comprehensive metric calculations
- Multiple game iterations per file

## Documentation Purpose

These files serve as:
1. **Experimental Evidence**: Raw data supporting research conclusions
2. **Reproducibility**: Complete logs for result verification
3. **Analysis Examples**: Sample data for testing statistical methods
4. **Model Behavior Documentation**: Detailed traces of LLM negotiation strategies