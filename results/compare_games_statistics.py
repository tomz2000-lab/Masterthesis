import re
import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

def parse_negotiation_log_agent_metrics(file_path):
    """
    Parses a negotiation log file to extract agent-based metrics, ensuring
    one row per model per game.

    Args:
        file_path (str): Path to the negotiation log file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with parsed agent-based metrics.
            - str: The detected game type (e.g., 'integrative_negotiation').

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the log file format is invalid or cannot be parsed.

    Example:
        >>> df, game_type = parse_negotiation_log_agent_metrics("log_file.out")
        >>> print(game_type)
        'integrative_negotiation'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        log_text = f.read()
    # Detect game type
    if re.search(r'IT\s+Team.*Marketing\s+Team', log_text, re.IGNORECASE):
        game_type = 'integrative_negotiation'
    elif re.search(r'BUYER.*SELLER|company\s+car', log_text, re.IGNORECASE):
        game_type = 'company_car'
    elif re.search(r'Development\s+Team.*Marketing\s+Team|GPU|resource\s+allocation', log_text, re.IGNORECASE):
        game_type = 'resource_allocation'
    else:
        game_type = 'unknown'
    iteration_blocks = re.split(r'===\s*Iteration\s+(\d+)/\d+\s*===', log_text)
    data = []
    for i in range(1, len(iteration_blocks), 2):
        if i + 1 >= len(iteration_blocks):
            break
        iteration_num = int(iteration_blocks[i])
        block = iteration_blocks[i + 1]
        if not block.strip():
            continue
        # Parse role assignments
        model_role_mapping = {}
        role_assignment_match = re.search(r'🎲\s*\[ROLE ASSIGNMENT\]\s*(.*)', block, re.IGNORECASE)
        if role_assignment_match:
            assignment_text = role_assignment_match.group(1)
            individual_assignments = re.findall(r'(model_[abc])\s*=\s*(\w+)', assignment_text, re.IGNORECASE)
            for model, role in individual_assignments:
                model_role_mapping[model] = role.upper()
        if len(model_role_mapping) < 2:
            continue
        # Parse first mover
        first_proposal_integrative = re.search(r'💡\s+Player\s+(model_[abc])\s+made\s+proposal\s+\(#1/4\)', block, re.IGNORECASE)
        first_proposal_company_car = re.search(r'💡\s+Player\s+(model_[abc])\s+made\s+offer.*\(proposal\s+1/4\)', block, re.IGNORECASE)
        first_mover = (
            first_proposal_integrative.group(1) if first_proposal_integrative
            else first_proposal_company_car.group(1) if first_proposal_company_car
            else 'unknown'
        )
        # Parse metrics for each model
        for model in model_role_mapping.keys():
            metrics = {}
            # Risk Minimization
            risk_match = re.search(rf'✅\s+Calculated\s+risk_minimization:.*{model}[\'"]:\s*([\d.]+)', block, re.IGNORECASE)
            if risk_match:
                metrics['Risk_Minimization'] = float(risk_match.group(1))
            # Deadline Sensitivity  
            deadline_match = re.search(rf'✅\s+Calculated\s+deadline_sensitivity:.*{model}[\'"]:\s*([\d.]+)', block, re.IGNORECASE)
            if deadline_match:
                metrics['Deadline_Sensitivity'] = float(deadline_match.group(1))
            # Feasibility (binary)
            feasibility_match = re.search(rf'✅\s+Calculated\s+feasibility:.*{model}[\'"]:\s*([\d.]+)', block, re.IGNORECASE)
            if feasibility_match:
                metrics['Feasibility'] = float(feasibility_match.group(1))
            # Utility Surplus
            utility_surplus_match = re.search(rf'✅\s+Calculated\s+utility_surplus:.*{model}[\'"]:\s*([-\d.]+)', block, re.IGNORECASE)
            if utility_surplus_match:
                metrics['Utility_Surplus'] = float(utility_surplus_match.group(1))
            data.append({
                'Iteration': iteration_num,
                'Model': model,
                'Role': model_role_mapping[model],
                'Is_First_Mover': int(first_mover == model),
                'First_Mover_Model': first_mover,
                'Game_type': game_type,
                **metrics
            })
    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df)} model performances across {df['Iteration'].nunique()} games")
    return df, game_type

def bias_corrected_metric_analysis(df, metric_name, metric_column):
    """
    Analyzes and reports bias-corrected model comparison for a given metric.

    Args:
        df (pd.DataFrame): DataFrame containing parsed agent-based metrics.
        metric_name (str): Name of the metric being analyzed (e.g., 'Risk Minimization').
        metric_column (str): Column name in the DataFrame corresponding to the metric.

    Returns:
        dict or None: A dictionary with bias-corrected analysis results if
        analysis is successful, otherwise None. The dictionary includes:
            - 'model_a' (str): Name of the first model.
            - 'model_b' (str): Name of the second model.
            - 'adjusted_difference' (float): Bias-adjusted difference between models.
            - 'predicted_mean_a' (float): Predicted mean for model_a.
            - 'predicted_mean_b' (float): Predicted mean for model_b.
            - 'p_value' (float): P-value of the test.
            - 'significant' (bool): Whether the result is statistically significant.

    Example:
        >>> results = bias_corrected_metric_analysis(df, 'Risk Minimization', 'Risk_Minimization')
        >>> print(results['adjusted_difference'])
        0.15
    """
    print(f"\n## 🎯 {metric_name.upper()} - BIAS-ADJUSTED MODEL COMPARISON")
    print("=" * 70)
    df_clean = df.dropna(subset=[metric_column])
    if len(df_clean) == 0:
        print(f"No {metric_name} data available")
        return None
    models = sorted(df_clean['Model'].unique())
    if len(models) != 2:
        print(f"Expected 2 models, found {len(models)}: {models}")
        return None
    model_a, model_b = models
    df_clean['is_model_a'] = (df_clean['Model'] == model_a).astype(int)
    # Choose regression type: OLS for continuous, Logit for binary
    is_binary = set(df_clean[metric_column].dropna().unique()) <= {0, 1}
    if is_binary:
        model = smf.logit(f"{metric_column} ~ is_model_a + C(Role) + C(Is_First_Mover)", data=df_clean).fit(disp=False)
        coeff = model.params.get('is_model_a', 0)
        pval = model.pvalues.get('is_model_a', 1)
        pred_a = model.predict({ 'is_model_a': 1, 'Role': df_clean['Role'].iloc[0], 'Is_First_Mover': 1 }).mean()
        pred_b = model.predict({ 'is_model_a': 0, 'Role': df_clean['Role'].iloc[0], 'Is_First_Mover': 0 }).mean()
    else:
        model = smf.ols(f"{metric_column} ~ is_model_a + C(Role) + C(Is_First_Mover)", data=df_clean).fit()
        coeff = model.params.get('is_model_a', 0)
        pval = model.pvalues.get('is_model_a', 1)
        pred_a = model.predict({ 'is_model_a': 1, 'Role': df_clean['Role'].iloc[0], 'Is_First_Mover': 1 }).mean()
        pred_b = model.predict({ 'is_model_a': 0, 'Role': df_clean['Role'].iloc[0], 'Is_First_Mover': 0 }).mean()
    print(model.summary())
    print(f"Bias-adjusted difference (model_a minus model_b): {coeff:.3f} (p = {pval:.4f})")
    print(f"Predicted means: {model_a}: {pred_a:.3f}, {model_b}: {pred_b:.3f}")
    return {
        'model_a': model_a,
        'model_b': model_b,
        'adjusted_difference': coeff,
        'predicted_mean_a': pred_a,
        'predicted_mean_b': pred_b,
        'p_value': pval,
        'significant': pval < 0.05
    }

def main():
    """
    Main function to perform bias-corrected analysis on agent-based metrics
    from a negotiation log file.

    Usage:
        python compare_agent_metrics_bias_corrected.py <log_file.out>

    Args:
        None (command-line arguments are used).

    Returns:
        None: Outputs results to the console and exports corrected data to a CSV file.

    Example:
        $ python compare_agent_metrics_bias_corrected.py integrative_negotiation_1975553.out
    """
    if len(sys.argv) < 2:
        print("Usage: python compare_agent_metrics_bias_corrected.py <log_file.out>")
        sys.exit(1)
    file_path = sys.argv[1]
    print(f"\n{'='*80}")
    print(f"📂 AGENT METRICS BIAS-CORRECTED ANALYSIS: {file_path}")
    print(f"{'='*80}")
    df, game_type = parse_negotiation_log_agent_metrics(file_path)
    if df.empty:
        print("❌ ERROR: No data could be parsed from the log file.")
        return
    print(f"\n📊 Dataset Summary:")
    print(f"- Game type: {game_type.upper()}")
    print(f"- Total agent records: {len(df)} ({df['Iteration'].nunique()} games)")
    print(f"- Models: {df['Model'].unique()}")
    print(f"- Roles: {df['Role'].unique()}")
    metrics_config = [
        ('Risk Minimization', 'Risk_Minimization'),
        ('Deadline Sensitivity', 'Deadline_Sensitivity'),
        ('Feasibility', 'Feasibility'),
        ('Utility Surplus', 'Utility_Surplus')
    ]
    all_results = {}
    for metric_name, metric_column in metrics_config:
        if metric_column in df.columns and not df[metric_column].isna().all():
            result = bias_corrected_metric_analysis(df, metric_name, metric_column)
            all_results[metric_name] = result
        else:
            print(f"\n⚠️  {metric_name} data not available in parsed results")
    # Export
    output_csv = file_path.replace('.out', '_agent_metrics_bias_corrected.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Agent metrics data exported to '{output_csv}'")
    print(f"\nSUMMARY:")
    for metric_name, result in all_results.items():
        if result and result['significant']:
            print(f"- {metric_name}: {result['model_a']} ({result['predicted_mean_a']:.3f}) vs {result['model_b']} ({result['predicted_mean_b']:.3f}), diff = {result['adjusted_difference']:.3f} (p = {result['p_value']:.4f})")
        elif result:
            print(f"- {metric_name}: No significant difference (p = {result['p_value']:.4f})")
        else:
            print(f"- {metric_name}: Not available")
    
if __name__ == "__main__":
    main()
