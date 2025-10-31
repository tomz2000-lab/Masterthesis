import re
import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
import math


def parse_negotiation_log_corrected(file_path):
    """Parse negotiation log with CORRECT statistical structure - one row per game.
    python compare_games_statistics_FIXED.py integrative_negotiation_1975553.out
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
            print(f"⚠️ Warning: Could not find role assignments for iteration {iteration_num}")
            continue

        # Parse first mover - handle both integrative and company car patterns
        # Integrative: "💡 Player model_x made proposal (#1/4):"
        # Company car: "💡 Player model_x made offer €XX,XXX (proposal 1/4)"
        first_proposal_integrative = re.search(r'💡\s+Player\s+(model_[abc])\s+made\s+proposal\s+\(#1/4\)', block, re.IGNORECASE)
        first_proposal_company_car = re.search(r'💡\s+Player\s+(model_[abc])\s+made\s+offer.*\(proposal\s+1/4\)', block, re.IGNORECASE)
        
        if first_proposal_integrative:
            first_mover = first_proposal_integrative.group(1)
        elif first_proposal_company_car:
            first_mover = first_proposal_company_car.group(1)
        else:
            first_mover = 'unknown'

        # Parse winner
        winner_match = re.search(r'\[LLM\s+WINNER\].*?\(player\s+(model_[abc])\s+won\)', block, re.IGNORECASE)
        if winner_match:
            winner = winner_match.group(1)
        else:
            utility_debug = re.search(
                r'model_([abc])\s+utility\s*=\s*([-\d.]+),\s*model_([abc])\s+utility\s*=\s*([-\d.]+)',
                block, re.IGNORECASE
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
                print(f"⚠️ Warning: Could not determine winner for iteration {iteration_num}")
                continue

        # Create ONE ROW PER GAME (correct statistical approach)
        if winner != 'tie':
            winning_role = model_role_mapping.get(winner, 'unknown')
            first_mover_role = model_role_mapping.get(first_mover, 'unknown')
            
            data.append({
                'Iteration': iteration_num,
                'Winning_Role': winning_role,  # Which role won (IT/Marketing, etc.)
                'First_Mover_Role': first_mover_role,  # Which role went first
                'Winner_Model': winner,  # Which model won (for model comparison)
                'First_Mover_Model': first_mover,  # Which model went first
                'Game_type': game_type,
                'Model_Assignments': model_role_mapping  # For reference
            })

    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df)} games with clear winners")
    return df, game_type


def analyze_role_bias_corrected(df, game_type):
    """CORRECTED role bias analysis - tests if certain roles have advantages."""
    print("\n## 📈 ROLE BIAS ANALYSIS (CORRECTED)")
    print("="*60)
    print("Question: Do certain roles (IT vs Marketing, Buyer vs Seller) have inherent advantages?")
    
    if len(df) == 0:
        return "No data available"

    # Count wins by role - this is the CORRECT approach
    role_wins = df['Winning_Role'].value_counts()
    total_games = len(df)
    
    print(f"\n### Win Counts by Role:")
    for role, wins in role_wins.items():
        win_rate = wins / total_games
        print(f"- {role}: {wins}/{total_games} games ({win_rate:.1%})")
    
    # Chi-square goodness of fit test
    # H0: Roles are equally likely to win (50/50 for 2 roles)
    if len(role_wins) == 2:
        expected_per_role = total_games / 2
        observed = role_wins.values
        expected = [expected_per_role, expected_per_role]
        
        chi2, p_value = chi2_contingency([observed, expected])[:2]
        
        # Effect size (Cohen's h for proportion difference)
        p1, p2 = observed[0]/total_games, observed[1]/total_games
        cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        
        significance = "**SIGNIFICANT ROLE BIAS**" if p_value < 0.05 else "No significant role bias"
        
        print(f"\n### Statistical Test:")
        print(f"- Chi-square test: χ²(1) = {chi2:.2f}, p = {p_value:.4f}")
        print(f"- Result: {significance}")
        print(f"- Effect size (Cohen's h): {abs(cohens_h):.3f}")
        
        if abs(cohens_h) < 0.2:
            effect_interp = "negligible"
        elif abs(cohens_h) < 0.5:
            effect_interp = "small"
        elif abs(cohens_h) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        print(f"- Effect interpretation: {effect_interp} difference")
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'cohens_h': cohens_h,
            'effect_size': effect_interp,
            'significant': p_value < 0.05
        }
    else:
        print("Cannot perform chi-square test: Need exactly 2 roles")
        return None


def analyze_first_mover_bias_corrected(df):
    """CORRECTED first-mover bias analysis."""
    print("\n## 🚀 FIRST-MOVER BIAS ANALYSIS (CORRECTED)")
    print("="*60)
    print("Question: Does going first provide an advantage?")
    
    # Create binary outcome: did the first mover win?
    df['first_mover_won'] = (df['Winner_Model'] == df['First_Mover_Model']).astype(int)
    
    first_mover_wins = df['first_mover_won'].sum()
    total_games = len(df)
    first_mover_win_rate = first_mover_wins / total_games
    
    print(f"\n### First-Mover Performance:")
    print(f"- First mover won: {first_mover_wins}/{total_games} games ({first_mover_win_rate:.1%})")
    print(f"- Second mover won: {total_games-first_mover_wins}/{total_games} games ({1-first_mover_win_rate:.1%})")
    
    # Chi-square goodness of fit test
    # H0: First mover wins 50% of the time
    observed = [first_mover_wins, total_games - first_mover_wins]
    expected = [total_games/2, total_games/2]
    
    chi2, p_value = chi2_contingency([observed, expected])[:2]
    
    # Effect size (Cohen's h)
    p_first = first_mover_win_rate
    p_expected = 0.5
    cohens_h = 2 * (np.arcsin(np.sqrt(p_first)) - np.arcsin(np.sqrt(p_expected)))
    
    significance = "**SIGNIFICANT FIRST-MOVER ADVANTAGE**" if p_value < 0.05 else "No significant first-mover advantage"
    
    print(f"\n### Statistical Test:")
    print(f"- Chi-square test: χ²(1) = {chi2:.2f}, p = {p_value:.4f}")
    print(f"- Result: {significance}")
    print(f"- Effect size (Cohen's h): {abs(cohens_h):.3f}")
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'cohens_h': cohens_h,
        'first_mover_win_rate': first_mover_win_rate,
        'significant': p_value < 0.05
    }


def bias_adjusted_model_comparison(df, role_bias_significant=False, first_mover_bias_significant=False):
    """Compare models while controlling for role and first-mover biases."""
    print("\n## 🎯 BIAS-ADJUSTED MODEL COMPARISON")
    print("="*80)
    
    if role_bias_significant or first_mover_bias_significant:
        print("Question: Which model performs better when controlling for detected biases?")
        print("Note: Bias correction is NECESSARY due to significant bias detection.")
    else:
        print("Question: Which model performs better? (No significant bias detected)")
        print("Note: Bias correction applied for completeness, but raw results should be similar.")
    
    # Get unique models
    all_models = set()
    for assignments in df['Model_Assignments']:
        all_models.update(assignments.keys())
    
    if len(all_models) != 2:
        print(f"Expected 2 models, found {len(all_models)}: {all_models}")
        return None
    
    model_a, model_b = sorted(all_models)
    
    # Create model outcome: which model won (regardless of role)
    df['model_a_won'] = (df['Winner_Model'] == model_a).astype(int)
    
    # Logistic regression controlling for biases
    # This is the CORRECT approach: outcome = model performance, controls = biases
    try:
        formula = 'model_a_won ~ C(Winning_Role) + C(First_Mover_Model)'
        model = smf.logit(formula=formula, data=df).fit(disp=False)
        
        print(f"\n### Logistic Regression Model:")
        print(f"Formula: {model_a}_won ~ role_advantage + first_mover_advantage")
        print(model.summary())
        
        # Extract bias-adjusted win probability for model_a
        intercept = model.params['Intercept']
        adjusted_log_odds = intercept  # Baseline probability with biases controlled
        adjusted_prob_a = 1 / (1 + np.exp(-adjusted_log_odds))
        adjusted_prob_b = 1 - adjusted_prob_a
        
        print(f"\n### 🏆 BIAS-ADJUSTED MODEL COMPARISON:")
        print(f"- {model_a}: {adjusted_prob_a:.3f} ({adjusted_prob_a*100:.1f}%) win probability")
        print(f"- {model_b}: {adjusted_prob_b:.3f} ({adjusted_prob_b*100:.1f}%) win probability")
        print("(These probabilities are adjusted for role bias and first-mover advantage)")
        
        # Raw comparison for reference
        raw_wins_a = df['model_a_won'].sum()
        raw_prob_a = raw_wins_a / len(df)
        
        print(f"\n### 📊 RAW vs ADJUSTED COMPARISON:")
        print(f"- {model_a} raw win rate: {raw_prob_a:.3f} ({raw_prob_a*100:.1f}%)")
        print(f"- {model_a} bias-adjusted: {adjusted_prob_a:.3f} ({adjusted_prob_a*100:.1f}%)")
        adjustment = adjusted_prob_a - raw_prob_a
        direction = "higher" if adjustment > 0 else "lower"
        print(f"- Adjustment: {adjustment:+.3f} ({direction} due to bias correction)")
        
        return {
            'model_a': model_a,
            'model_b': model_b,
            'adjusted_prob_a': adjusted_prob_a,
            'raw_prob_a': raw_prob_a,
            'adjustment': adjustment
        }
        
    except Exception as e:
        print(f"Could not fit logistic regression: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_games_statistics_FIXED.py <log_file.out>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\n{'='*80}")
    print(f"📂 CORRECTED BIAS ANALYSIS: {file_path}")
    print(f"{'='*80}")

    # Parse with correct structure
    df, game_type = parse_negotiation_log_corrected(file_path)
    
    if df.empty:
        print("❌ ERROR: No data could be parsed from the log file.")
        return

    print(f"\n📊 Dataset Summary:")
    print(f"- Game type: {game_type.upper()}")
    print(f"- Total games analyzed: {len(df)}")
    print(f"- Roles: {df['Winning_Role'].unique()}")

    # Analyze biases with correct methods
    role_results = analyze_role_bias_corrected(df, game_type)
    first_mover_results = analyze_first_mover_bias_corrected(df)
    
    # Check if any bias was significant
    role_bias_significant = role_results and role_results.get('significant', False)
    first_mover_bias_significant = first_mover_results and first_mover_results.get('significant', False)
    
    # Model comparison with bias adjustment
    model_results = bias_adjusted_model_comparison(df, role_bias_significant, first_mover_bias_significant)
    
    print(f"\n{'='*80}")
    print("## 📋 SUMMARY OF CORRECTED ANALYSIS")
    print(f"{'='*80}")
    
    print("### 🧪 BIAS DETECTION RESULTS:")
    if role_results and role_results.get('significant'):
        print(f"✅ **Role bias detected**: {role_results['effect_size']} effect (p = {role_results['p_value']:.4f})")
    else:
        print("✅ **No significant role bias detected**")
    
    if first_mover_results and first_mover_results.get('significant'):
        print(f"✅ **First-mover advantage detected**: {first_mover_results['first_mover_win_rate']:.1%} win rate (p = {first_mover_results['p_value']:.4f})")
    else:
        print("✅ **No significant first-mover advantage detected**")
    
    print("\n### 🏆 FINAL MODEL PERFORMANCE RANKING (BIAS-ADJUSTED):")
    if model_results:
        model_a = model_results['model_a']
        model_b = model_results['model_b']
        prob_a = model_results['adjusted_prob_a']
        prob_b = 1 - prob_a
        
        # Determine winner
        if prob_a > 0.5:
            winner = model_a
            winner_prob = prob_a
            loser = model_b
            loser_prob = prob_b
        else:
            winner = model_b
            winner_prob = prob_b
            loser = model_a
            loser_prob = prob_a
        
        print(f"🥇 **WINNER: {winner}**")
        print(f"   - Bias-adjusted win probability: {winner_prob:.3f} ({winner_prob*100:.1f}%)")
        print(f"🥈 **Second place: {loser}**")
        print(f"   - Bias-adjusted win probability: {loser_prob:.3f} ({loser_prob*100:.1f}%)")
        
        # Show margin of victory
        margin = abs(winner_prob - loser_prob)
        if margin > 0.2:
            dominance = "Strong advantage"
        elif margin > 0.1:
            dominance = "Moderate advantage"
        elif margin > 0.05:
            dominance = "Slight advantage"
        else:
            dominance = "Essentially tied"
            
        print(f"   - Margin: {margin:.3f} ({dominance})")
        
        # Show if bias correction mattered - CORRECTED LOGIC
        raw_prob_a = model_results['raw_prob_a']
        adjustment = model_results['adjustment']
        
        # Check if EITHER bias was statistically significant
        role_bias_significant = role_results and role_results.get('significant', False)
        first_mover_bias_significant = first_mover_results and first_mover_results.get('significant', False)
        any_bias_detected = role_bias_significant or first_mover_bias_significant
        
        if any_bias_detected:
            print(f"\n⚠️  **Bias Correction Applied**: Statistically significant bias was detected and corrected.")
            print(f"   - Adjustment magnitude: {adjustment:+.3f} (difference in win probability)")
            if abs(adjustment) > 0.05:
                print(f"   - Impact: Large correction - raw win rates would have been misleading")
            elif abs(adjustment) > 0.02:
                print(f"   - Impact: Moderate correction applied")
            else:
                print(f"   - Impact: Small correction (bias was significant but effect was limited)")
        else:
            print(f"\n✅ **No Bias Correction Needed**: No statistically significant bias detected.")
            print(f"   - Raw win rates are reliable (adjustment: {adjustment:+.3f})")
            if abs(adjustment) > 0.05:
                print(f"   - Note: Large numerical difference due to random variation, not systematic bias")
        
        print(f"\n📊 **Statistical Details**:")
        print(f"   - Analysis method: Multiple logistic regression")
        print(f"   - Controls: Role bias + First-mover advantage")
        print(f"   - Sample size: {len(df)} independent games")
        print(f"   - Each game counted once (no double-counting)")
        
        # Final interpretation based on bias detection
        print(f"\n🎯 **INTERPRETATION**:")
        if any_bias_detected:
            print(f"   The bias-adjusted results are MORE RELIABLE than raw win rates")
            print(f"   because significant bias was detected and corrected.")
        else:
            print(f"   The bias-adjusted and raw results should be similar")
            print(f"   because no significant bias was detected.")
            print(f"   Either result can be trusted for model comparison.")
        
    else:
        print("❌ **Could not determine model performance** (insufficient data or regression failed)")
    
    # Export data
    output_csv = file_path.replace('.out', '_corrected_analysis.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Corrected data exported to '{output_csv}'")


if __name__ == "__main__":
    main()