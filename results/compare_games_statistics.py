import re
import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency


"""python compare_games_statistics.py integrative_negotiation_1922705.out
    python compare_games_statistics.py company_car_1901974.out
    python compare_games_statistics.py resource_allocation_19.out
"""


def detect_game_type(log_text):
    """Detect which game type this is based on log content."""
    if re.search(r'IT\s+Team.*Marketing\s+Team', log_text, re.IGNORECASE):
        return 'integrative_negotiation'
    elif re.search(r'BUYER.*SELLER|company\s+car', log_text, re.IGNORECASE):
        return 'company_car'
    elif re.search(r'Development\s+Team.*Marketing\s+Team|GPU|resource\s+allocation', log_text, re.IGNORECASE):
        return 'resource_allocation'
    else:
        return 'unknown'


def parse_negotiation_log(file_path):
    """Parse negotiation log extracting iteration, roles, first mover, and winner."""
    with open(file_path, 'r', encoding='utf-8') as f:
        log_text = f.read()

    game_type = detect_game_type(log_text)
    print(f"📊 Detected game type: {game_type.upper()}")

    if game_type == 'integrative_negotiation':
        role_a_pattern = r'(model_[abc]).*?=\s*IT'
        role_b_pattern = r'(model_[abc]).*?=\s*MARKETING'
        role_a_name = 'IT_role'
        role_b_name = 'Marketing_role'
    elif game_type == 'company_car':
        role_a_pattern = r'(model_[abc]).*?=\s*BUYER'
        role_b_pattern = r'(model_[abc]).*?=\s*SELLER'
        role_a_name = 'Buyer_role'
        role_b_name = 'Seller_role'
    elif game_type == 'resource_allocation':
        role_a_pattern = r'(model_[abc]).*?=\s*DEVELOPMENT'
        role_b_pattern = r'(model_[abc]).*?=\s*MARKETING'
        role_a_name = 'Development_role'
        role_b_name = 'Marketing_role'
    else:
        print("⚠️  WARNING: Unknown game type. Attempting generic parsing...")
        role_a_pattern = r'(model_[abc])\s*=\s*\w+'
        role_b_pattern = r'(model_[abc])\s*=\s*\w+'
        role_a_name = 'Role_A'
        role_b_name = 'Role_B'

    iteration_blocks = re.split(r'===\s*Iteration\s+(\d+)/\d+\s*===', log_text)
    data = []

    for i in range(1, len(iteration_blocks), 2):
        if i + 1 >= len(iteration_blocks):
            break

        iteration_num = int(iteration_blocks[i])
        block = iteration_blocks[i + 1]

        if not block.strip():
            continue

        # Find which model plays which role - handle multiple formats
        model_role_mapping = {}
        
        # Format 1: 🎲 [ROLE ASSIGNMENT] model_b = DEVELOPMENT, model_a = MARKETING
        role_assignment_match = re.search(r'🎲\s*\[ROLE ASSIGNMENT\]\s*(.*)', block, re.IGNORECASE)
        if role_assignment_match:
            assignment_text = role_assignment_match.group(1)
            individual_assignments = re.findall(r'(model_[abc])\s*=\s*(\w+)', assignment_text, re.IGNORECASE)
            for model, role in individual_assignments:
                model_role_mapping[model] = role.upper()
        else:
            # Format 2: Look for pattern like "model_a = BUYER, model_b = SELLER" 
            full_assignment = re.search(r'(model_[abc])\s*=\s*(\w+),?\s*(model_[abc])?\s*=?\s*(\w+)?', block, re.IGNORECASE)
            if full_assignment:
                model1, role1, model2, role2 = full_assignment.groups()
                model_role_mapping[model1] = role1.upper()
                if model2 and role2:
                    model_role_mapping[model2] = role2.upper()
            else:
                # Format 3: Try to find individual assignments
                individual_assignments = re.findall(r'(model_[abc])\s*=\s*(\w+)', block, re.IGNORECASE)
                for model, role in individual_assignments:
                    model_role_mapping[model] = role.upper()

        if len(model_role_mapping) < 2:
            print(f"⚠️  Warning: Could not find both model role assignments for iteration {iteration_num}")
            print(f"    Available mappings: {model_role_mapping}")
            continue

        # Store all model-role pairs (no need to determine which is "model_a" vs "model_b")
        model_roles = list(model_role_mapping.items())

        first_proposal_match = re.search(r'Player\s+(model_[abc])\s+made\s+proposal\s+\(#1', block, re.IGNORECASE)
        if first_proposal_match:
            first_mover = first_proposal_match.group(1)
        else:
            first_mover_match = re.search(r'Starting\s+player:\s*(model_[abc])', block, re.IGNORECASE)
            if first_mover_match:
                first_mover = first_mover_match.group(1)
            else:
                first_offer = re.search(r'\[DEBUG\]\s+\[model_([abc])\].*?offer', block, re.IGNORECASE)
                if first_offer:
                    first_mover = f'model_{first_offer.group(1)}'
                else:
                    print(f"⚠️  Warning: Could not determine first mover for iteration {iteration_num}")
                    first_mover = 'unknown'

        winner_match = re.search(r'\[LLM\s+WINNER\].*?\(player\s+(model_[abc])\s+won\)', block, re.IGNORECASE)
        if winner_match:
            winner = winner_match.group(1)
        else:
            utility_debug = re.search(
                r'model_([abc])\s+utility\s*=\s*([-\d.]+),\s*model_([abc])\s+utility\s*=\s*([-\d.]+)',
                block,
                re.IGNORECASE
            )
            if utility_debug:
                model1, util1, model2, util2 = utility_debug.groups()
                util1, util2 = float(util1), float(util2)
                if util1 > util2:
                    winner = f'model_{model1}'
                elif util2 > util1:
                    winner = f'model_{model2}'
                else:
                    winner = 'tie'
            else:
                print(f"⚠️  Warning: Could not determine winner for iteration {iteration_num}")
                continue

        # Store both models with their respective roles
        for model_id, role in model_roles:
            data.append({
                'Iteration': iteration_num,
                'Model_ID': model_id,
                'Role': role,
                'First_mover': first_mover,
                'Winner': winner,
                'Game_type': game_type
            })

    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df)} model-role combinations from {len(df)//2} iterations")
    return df, 'Role', game_type


def analyze_bias(df, factor_col, factor_name):
    """Analyze bias with chi-square test."""
    df_filtered = df[df['Winner'] != 'tie'].copy()

    if len(df_filtered) == 0:
        return None, None, "No data available (all ties)"

    # Create binary outcome for each model-role combination
    df_filtered['model_won'] = (df_filtered['Model_ID'] == df_filtered['Winner']).astype(int)
    
    if factor_col == 'Role':
        # For role bias, we want to see win rates by role
        contingency = pd.crosstab(df_filtered[factor_col], df_filtered['model_won'], margins=True)
        contingency.columns = ['Lost', 'Won', 'Total']
        contingency_pct = pd.crosstab(df_filtered[factor_col], df_filtered['model_won'], normalize='index') * 100
        contingency_pct.columns = ['Lost %', 'Won %']
        contingency_test = pd.crosstab(df_filtered[factor_col], df_filtered['model_won'])
    else:
        # For first mover bias, analyze differently
        contingency = pd.crosstab(df_filtered[factor_col], df_filtered['Winner'], margins=True)
        contingency_pct = pd.crosstab(df_filtered[factor_col], df_filtered['Winner'], normalize='index') * 100
        contingency_test = pd.crosstab(df_filtered[factor_col], df_filtered['Winner'])

    if contingency_test.shape[0] < 2 or contingency_test.shape[1] < 2:
        return contingency, contingency_pct, "Insufficient data for chi-square test"

    chi2, p_value, dof, expected = chi2_contingency(contingency_test)
    significance = "**SIGNIFICANT**" if p_value < 0.05 else "Not significant"
    interpretation = f"χ²({dof}) = {chi2:.2f}, p = {p_value:.4f} → {significance}"

    if p_value < 0.05:
        if factor_col == 'Role':
            # Compare win rates between roles
            role_win_rates = df_filtered.groupby('Role')['model_won'].mean()
            best_role = role_win_rates.idxmax()
            worst_role = role_win_rates.idxmin()
            interpretation += f" (Role bias detected: {best_role} has advantage over {worst_role})"
        elif factor_col == 'First_mover':
            # Analyze first mover advantage
            first_mover_wins = contingency_test.sum(axis=1)
            if 'model_a' in first_mover_wins.index and 'model_b' in first_mover_wins.index:
                if first_mover_wins['model_a'] > first_mover_wins['model_b']:
                    interpretation += " (First-mover advantage detected)"
                else:
                    interpretation += " (Second-mover advantage detected)"
    else:
        if factor_col == 'Role':
            interpretation += f" (No role bias detected)"
        else:
            interpretation += f" (No first-mover bias detected)"

    return contingency, contingency_pct, interpretation


def print_markdown_table(contingency, contingency_pct, title):
    """Print contingency table in Markdown format with counts and percentages."""
    print(f"\n## {title}")
    print("\n### Counts")
    print(contingency.to_markdown())
    print("\n### Percentages (by row)")
    print(contingency_pct.to_markdown(floatfmt=".1f"))


def logistic_regression_adjustment(df, role_col_name):
    """
    Perform logistic regression of model winning ~ role variable to adjust for role bias.
    Outputs the fitted model, overall adjusted win probability, and predicted probabilities by role.
    """
    # Filter out ties and create one row per model per iteration
    df_filtered = df[df['Winner'] != 'tie'].copy()
    
    # Create binary outcome: 1 if this model won, 0 otherwise
    df_filtered['model_won'] = (df_filtered['Model_ID'] == df_filtered['Winner']).astype(int)

    # Logistic regression formula with categorical role covariate
    formula = f'model_won ~ C({role_col_name})'

    # Fit logistic regression model
    model = smf.logit(formula=formula, data=df_filtered).fit(disp=False)

    # Get predicted probabilities by role
    roles = df_filtered[role_col_name].unique()
    pred_probs = {}
    for role in roles:
        pred_data = pd.DataFrame({role_col_name: [role]})
        pred_prob = model.predict(pred_data)[0]
        pred_probs[role] = pred_prob

    # Calculate overall adjusted win probability weighted by role frequencies
    role_freq = df_filtered[role_col_name].value_counts(normalize=True).to_dict()
    adjusted_prob = sum(pred_probs[r] * role_freq.get(r, 0) for r in roles)

    return model, adjusted_prob, pred_probs


def parse_model_names(log_text):
    """
    Extract actual model names like 'meta-llama/Llama-3.2-3B-Instruct' for model_a, model_b etc.
    """
    model_name_pattern = r'📝 Registered model: (model_[abc]) \(new instance for ([^\)]+)\)'
    matches = re.findall(model_name_pattern, log_text)
    return {model: name for model, name in matches}


def calculate_adjusted_model_performance(df, model_names):
    """
    Calculate model performance adjusted for role bias using logistic regression.
    
    This function performs a logistic regression where:
    - Dependent variable: whether the model won (1) or lost (0)
    - Independent variables: model identity + role
    
    The model coefficients represent the log-odds of winning, adjusted for role bias.
    """
    # Create binary outcome for each model (use copy to avoid SettingWithCopyWarning)
    df = df.copy()
    df['model_won'] = (df['Model_ID'] == df['Winner']).astype(int)
    
    # Fit logistic regression with both model and role effects
    formula = 'model_won ~ C(Model_ID) + C(Role)'
    
    try:
        model = smf.logit(formula=formula, data=df).fit(disp=False)
        
        print("### Logistic Regression: Model Performance Controlling for Role")
        print(model.summary())
        
        # Extract model coefficients (adjusted win probabilities)
        print("\n### Model Performance (Role-Adjusted):")
        
        # Get baseline probability (intercept + first model)
        intercept = model.params['Intercept']
        
        # Calculate adjusted win probabilities for each model
        print("NOTE: These probabilities represent each model's chance of winning")
        print("against the 'average opponent' when controlling for role bias.")
        print("They don't sum to 100% because they're not head-to-head probabilities.\n")
        
        for model_id in df['Model_ID'].unique():
            if f'C(Model_ID)[T.{model_id}]' in model.params:
                coef = model.params[f'C(Model_ID)[T.{model_id}]']
                log_odds = intercept + coef
            else:
                # This is the reference model (coefficient = 0)
                log_odds = intercept
            
            # Convert log-odds to probability
            prob = 1 / (1 + np.exp(-log_odds))
            
            model_name = model_names.get(model_id, model_id)
            raw_win_rate = (df[df['Model_ID'] == model_id]['model_won']).mean()
            
            print(f"- {model_name}:")
            print(f"  Raw win rate: {raw_win_rate:.3f} ({raw_win_rate*100:.1f}%)")
            print(f"  Role-adjusted win probability: {prob:.3f} ({prob*100:.1f}%)")
            print(f"  (Probability of winning vs. average opponent, controlling for role)")
            
            # Calculate the difference
            adjustment = prob - raw_win_rate
            if abs(adjustment) > 0.01:  # Only show if adjustment is meaningful
                direction = "higher" if adjustment > 0 else "lower"
                print(f"  Adjustment: {adjustment:+.3f} ({direction} due to role bias)")
        
        # Calculate head-to-head probability
        models = list(df['Model_ID'].unique())
        if len(models) == 2:
            model_a_id, model_b_id = models
            if f'C(Model_ID)[T.{model_b_id}]' in model.params:
                model_b_coef = model.params[f'C(Model_ID)[T.{model_b_id}]']
                # Head-to-head log-odds difference
                log_odds_diff = model_b_coef
                # Convert to probability that model_b beats model_a
                prob_b_wins = 1 / (1 + np.exp(-log_odds_diff))
                prob_a_wins = 1 - prob_b_wins
                
                print(f"\n### Head-to-Head Probability (Role-Adjusted):")
                print(f"- {model_names.get(model_a_id, model_a_id)} beats {model_names.get(model_b_id, model_b_id)}: {prob_a_wins:.1%}")
                print(f"- {model_names.get(model_b_id, model_b_id)} beats {model_names.get(model_a_id, model_a_id)}: {prob_b_wins:.1%}")
                print("(These DO sum to 100% - direct head-to-head comparison)")
        
    except Exception as e:
        print(f"Could not calculate adjusted model performance: {e}")
        print("This might happen if there's insufficient variation in the data.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_games.py <log_file.out>")
        print("\nExample:")
        print("  python compare_games.py integrative_negotiation_1922705.out")
        print("  python compare_games.py company_car_1901974.out")
        print("  python compare_games.py resource_allocation_1922855.out")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"\n{'='*60}")
    print(f"📂 Analyzing: {file_path}")
    print(f"{'='*60}\n")

    df, role_col_name, game_type = parse_negotiation_log(file_path)

    if df.empty:
        print("❌ ERROR: No data could be parsed from the log file.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        log_text = f.read()

    model_names = parse_model_names(log_text)

    print(f"\n# Negotiation Bias Analysis - {game_type.upper().replace('_', ' ')}")
    print(f"\nTotal iterations parsed: {len(df)//2}")
    print(f"Ties: {len(df[df['Winner'] == 'tie'])//2}")
    print(f"Iterations with clear winner: {len(df[df['Winner'] != 'tie'])//2}")

    print(f"\n## Role Distribution")
    role_dist = df[role_col_name].value_counts()
    print(role_dist.to_markdown())
    
    print(f"\n## Model-Role Combinations")
    model_role_dist = df.groupby(['Model_ID', 'Role']).size().reset_index(name='Count')
    print(model_role_dist.to_markdown(index=False))

    print("\n" + "="*60)
    contingency_role, pct_role, interp_role = analyze_bias(df, role_col_name, role_col_name)
    if contingency_role is not None:
        print_markdown_table(contingency_role, pct_role, f"Role Bias ({role_col_name.replace('_role', '').title()})")
        print(f"\n**Statistical Test:** {interp_role}")
    else:
        print(f"\n**Role Bias Analysis:** {interp_role}")

    print("\n" + "="*60)
    contingency_first, pct_first, interp_first = analyze_bias(df, 'First_mover', 'First-Mover Bias')
    if contingency_first is not None:
        print_markdown_table(contingency_first, pct_first, "First-Mover Bias")
        print(f"\n**Statistical Test:** {interp_first}")
    else:
        print(f"\n**First-Mover Bias Analysis:** {interp_first}")

    print("\n" + "="*60)
    print("\n## Summary")
    print(f"- **Role Bias:** {interp_role}")
    print(f"- **First-Mover Bias:** {interp_first}")

    # Logistic regression adjustment for role bias
    model, adjusted_prob, pred_probs = logistic_regression_adjustment(df, role_col_name)

    print("\n## Logistic Regression Model Summary (model win ~ role)")
    print(model.summary())

    print(f"\n## Overall Win Probability Adjusted for Role Bias: {adjusted_prob:.3f}")
    print("## Predicted Win Probabilities by Role:")
    for role, prob in pred_probs.items():
        print(f"- {role}: {prob:.3f}")

    # Print raw wins for comparison
    df_no_ties = df[df['Winner'] != 'tie']
    raw_wins = df_no_ties['Winner'].value_counts()
    print("\n## Raw Win Counts (excluding ties):")
    for model_key in raw_wins.index:
        model_label = model_names.get(model_key, model_key)
        count = raw_wins[model_key]
        print(f"- {model_label}: {count}")
    
    # Show win rates by role
    print("\n## Win Rates by Role:")
    role_wins = df_no_ties.groupby('Role')['Model_ID'].apply(lambda x: (x == df_no_ties.loc[x.index, 'Winner']).mean()).sort_values(ascending=False)
    for role, win_rate in role_wins.items():
        print(f"- {role}: {win_rate:.3f} ({win_rate*100:.1f}%)")
    
    # Calculate model performance adjusted for role bias
    print("\n## Model Performance Adjusted for Role Bias:")
    calculate_adjusted_model_performance(df_no_ties, model_names)

    # Export detailed data
    output_csv = file_path.replace('.out', '_bias_analysis.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Detailed data exported to '{output_csv}'")


if __name__ == "__main__":
    main()
