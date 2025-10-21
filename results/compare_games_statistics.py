import re
import sys
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency


"""python compare_games.py integrative_negotiation_1922705.out
    python compare_games.py company_car_1901974.out
    python compare_games.py resource_allocation_19.out
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

        role_a_match = re.search(role_a_pattern, block, re.IGNORECASE)
        role_b_match = re.search(role_b_pattern, block, re.IGNORECASE)

        if not role_a_match:
            continue

        role_a = role_a_match.group(1)

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

        data.append({
            'Iteration': iteration_num,
            role_a_name: role_a,
            'First_mover': first_mover,
            'Winner': winner,
            'Game_type': game_type
        })

    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df)} iterations")
    return df, role_a_name, game_type


def analyze_bias(df, factor_col, factor_name):
    """Analyze bias with chi-square test."""
    df_filtered = df[df['Winner'] != 'tie'].copy()

    if len(df_filtered) == 0:
        return None, None, "No data available (all ties)"

    contingency = pd.crosstab(df_filtered[factor_col], df_filtered['Winner'], margins=True)
    contingency_pct = pd.crosstab(df_filtered[factor_col], df_filtered['Winner'], normalize='index') * 100
    contingency_test = pd.crosstab(df_filtered[factor_col], df_filtered['Winner'])

    if contingency_test.shape[0] < 2 or contingency_test.shape[1] < 2:
        return contingency, contingency_pct, "Insufficient data for chi-square test"

    chi2, p_value, dof, expected = chi2_contingency(contingency_test)
    significance = "**SIGNIFICANT**" if p_value < 0.05 else "Not significant"
    interpretation = f"χ²({dof}) = {chi2:.2f}, p = {p_value:.4f} → {significance}"

    if p_value < 0.05:
        if 'role' in factor_col.lower():
            role_a_wins = sum(contingency_test.loc[model, model] for model in contingency_test.index if model in contingency_test.columns)
            role_b_wins = sum(contingency_test.loc[model, other] for model in contingency_test.index for other in contingency_test.columns if model != other)
            if role_a_wins > role_b_wins:
                interpretation += f" (Role bias detected: models tend to win when playing as {factor_name.replace('_role', '')})"
            else:
                interpretation += f" (Role bias detected: models tend to lose when playing as {factor_name.replace('_role', '')})"
        elif factor_col == 'First_mover':
            first_wins = sum(contingency_test.loc[model, model] for model in contingency_test.index if model in contingency_test.columns)
            second_wins = sum(contingency_test.loc[model, other] for model in contingency_test.index for other in contingency_test.columns if model != other)
            if first_wins > second_wins:
                interpretation += " (First-mover advantage detected)"
            else:
                interpretation += " (Second-mover advantage detected)"
    else:
        if 'role' in factor_col.lower():
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
    Perform logistic regression of model_a winning ~ role variable to adjust for role bias.
    Outputs the fitted model, overall adjusted win probability for model_a, and predicted probabilities by role.
    """
    # Filter out ties
    df = df[df['Winner'] != 'tie'].copy()

    # Binary outcome: 1 if model_a won, 0 otherwise (model_b assumed)
    df['model_a_win'] = (df['Winner'] == 'model_a').astype(int)

    # Logistic regression formula with categorical role covariate
    formula = f'model_a_win ~ C({role_col_name})'

    # Fit logistic regression model
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Calculate predicted probabilities by role
    roles = df[role_col_name].unique()
    pred_probs = {}
    for role in roles:
        pred_data = pd.DataFrame({role_col_name: [role]})
        pred_prob = model.predict(pred_data)[0]
        pred_probs[role] = pred_prob

    # Calculate overall adjusted win probability weighted by role frequencies
    role_freq = df[role_col_name].value_counts(normalize=True).to_dict()
    adjusted_prob = sum(pred_probs[r] * role_freq.get(r, 0) for r in roles)

    return model, adjusted_prob, pred_probs


def parse_model_names(log_text):
    """
    Extract actual model names like 'meta-llama/Llama-3.2-3B-Instruct' for model_a, model_b etc.
    """
    model_name_pattern = r'📝 Registered model: (model_[abc]) \(new instance for ([^\)]+)\)'
    matches = re.findall(model_name_pattern, log_text)
    return {model: name for model, name in matches}


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
    print(f"\nTotal iterations parsed: {len(df)}")
    print(f"Ties: {len(df[df['Winner'] == 'tie'])}")
    print(f"Iterations with clear winner: {len(df[df['Winner'] != 'tie'])}")

    print(f"\n## Role Distribution")
    role_dist = df[role_col_name].value_counts()
    print(role_dist.to_markdown())

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

    print("\n## Logistic Regression Model Summary (model_a win ~ role)")
    print(model.summary())

    print(f"\n## Adjusted Win Probability for model_a Controlling for Role Bias: {adjusted_prob:.3f}")
    print("## Predicted Win Probabilities for model_a by Role:")
    for role, prob in pred_probs.items():
        print(f"- {role}: {prob:.3f}")

    # Print raw wins for comparison
    raw_wins = df[df['Winner'] != 'tie']['Winner'].value_counts()
    print("\n## Raw Win Counts (excluding ties):")
    for model_key in ['model_a', 'model_b']:
        model_label = model_names.get(model_key, model_key)
        count = raw_wins.get(model_key, 0)
        print(f"- {model_label}: {count}")

    # Export detailed data
    output_csv = file_path.replace('.out', '_bias_analysis.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Detailed data exported to '{output_csv}'")


if __name__ == "__main__":
    main()
