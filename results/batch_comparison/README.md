# Batch Comparison Output Files

This directory contains experimental output files from batch runs comparing different LLM models in negotiation scenarios.

## Directory Structure

- `car_game/car_errpr` - Company car negotiation experiments
- `integrative_game/integrative_error` - Integrative negotiation experiments  
- `resource_game/resource_error` - Resource allocation negotiation experiments

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
   - Deadline Feasability scores
   - Feasibility scores
   - Utility Surplus calculations
6. **Agreement Analysis**: Final outcomes and agreement statistics
7. **Further metrics**: Average agreement round and the agreement rate are further metircs, measured through the statistics code

