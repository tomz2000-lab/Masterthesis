#!/usr/bin/env python3
"""
Statistical Analysis of Role-Controlled Model Performance
Tests for statistical significance of model differences while controlling for role bias.
"""

import re
import sys
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import pandas as pd

def extract_detailed_game_data(filename):
    """Extract comprehensive game data for statistical analysis."""
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract role assignments
    role_assignments = []
    role_matches = re.findall(r'🎲 \[ROLE ASSIGNMENT\] (.+)', content)
    for match in role_matches:
        if 'model_a = BUYER, model_b = SELLER' in match:
            role_assignments.append({'model_a': 'buyer', 'model_b': 'seller'})
        elif 'model_b = BUYER, model_a = SELLER' in match:
            role_assignments.append({'model_a': 'seller', 'model_b': 'buyer'})
    
    # Extract LLM winners
    llm_winners = []
    winner_matches = re.findall(r'🏆 \[LLM WINNER\] .+ \(player (model_[ab]) won\)', content)
    llm_winners.extend(winner_matches)
    
    # Extract utilities for utility-based analysis
    utility_matches = re.findall(r'🔍 \[BATNA DEBUG\] Utilities: buyer=(\d+\.\d+), seller=(\d+\.\d+)', content)
    
    # Extract agreement prices
    price_matches = re.findall(r'🔍 \[BATNA DEBUG\] Round \d+: price=(\d+)', content)
    
    # Extract agreement status
    agreement_matches = re.findall(r'Agreement reached: (True|False)', content)
    
    # Extract metrics
    risk_minimization_matches = re.findall(r'✅ Calculated risk_minimization: \{\'model_a\': ([\d\.]+), \'model_b\': ([\d\.]+)\}', content)
    deadline_sensitivity_matches = re.findall(r'✅ Calculated deadline_sensitivity: \{\'model_a\': ([\d\.]+), \'model_b\': ([\d\.]+)\}', content)
    feasibility_matches = re.findall(r'✅ Calculated feasibility: \{\'model_a\': ([\d\.]+), \'model_b\': ([\d\.]+)\}', content)
    
    # Combine all data
    games_data = []
    min_length = min(len(role_assignments), len(llm_winners), len(utility_matches), 
                     len(price_matches), len(risk_minimization_matches), 
                     len(deadline_sensitivity_matches), len(feasibility_matches))
    
    print(f"Data lengths: roles={len(role_assignments)}, winners={len(llm_winners)}, utilities={len(utility_matches)}")
    print(f"Metrics: risk={len(risk_minimization_matches)}, deadline={len(deadline_sensitivity_matches)}, feasibility={len(feasibility_matches)}")
    print(f"Using minimum length: {min_length}")
    
    for i in range(min_length):
        assignment = role_assignments[i]
        llm_winner = llm_winners[i]
        buyer_utility, seller_utility = utility_matches[i]
        agreed_price = int(price_matches[i])
        
        # Extract metrics for each model
        risk_a, risk_b = risk_minimization_matches[i]
        deadline_a, deadline_b = deadline_sensitivity_matches[i]
        feasibility_a, feasibility_b = feasibility_matches[i]
        
        # Determine utility winner
        buyer_utility = float(buyer_utility)
        seller_utility = float(seller_utility)
        utility_winner_role = 'seller' if seller_utility > buyer_utility else 'buyer'
        
        # Map to model names
        if utility_winner_role == 'seller':
            utility_winner_model = 'model_a' if assignment['model_a'] == 'seller' else 'model_b'
        else:
            utility_winner_model = 'model_a' if assignment['model_a'] == 'buyer' else 'model_b'
        
        games_data.append({
            'game_id': i + 1,
            'model_a_role': assignment['model_a'],
            'model_b_role': assignment['model_b'],
            'llm_winner': llm_winner,
            'llm_winner_role': assignment[llm_winner],
            'utility_winner_model': utility_winner_model,
            'utility_winner_role': utility_winner_role,
            'buyer_utility': buyer_utility,
            'seller_utility': seller_utility,
            'agreed_price': agreed_price,
            'model_a_utility': buyer_utility if assignment['model_a'] == 'buyer' else seller_utility,
            'model_b_utility': seller_utility if assignment['model_a'] == 'buyer' else buyer_utility,
            # Metrics
            'model_a_risk': float(risk_a),
            'model_b_risk': float(risk_b),
            'model_a_deadline': float(deadline_a),
            'model_b_deadline': float(deadline_b),
            'model_a_feasibility': float(feasibility_a),
            'model_b_feasibility': float(feasibility_b)
        })
    
    return games_data

def role_controlled_analysis(games_data):
    """Perform role-controlled statistical analysis."""
    
    print("=== ROLE-CONTROLLED STATISTICAL ANALYSIS ===\n")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(games_data)
    
    # 1. Basic Descriptive Statistics
    print("1. DESCRIPTIVE STATISTICS")
    print("-" * 40)
    
    # LLM Win rates by model and role
    model_a_seller = df[df['model_a_role'] == 'seller']
    model_a_buyer = df[df['model_a_role'] == 'buyer']
    model_b_seller = df[df['model_b_role'] == 'seller']
    model_b_buyer = df[df['model_b_role'] == 'buyer']
    
    ma_seller_wins = len(model_a_seller[model_a_seller['llm_winner'] == 'model_a'])
    ma_buyer_wins = len(model_a_buyer[model_a_buyer['llm_winner'] == 'model_a'])
    mb_seller_wins = len(model_b_seller[model_b_seller['llm_winner'] == 'model_b'])
    mb_buyer_wins = len(model_b_buyer[model_b_buyer['llm_winner'] == 'model_b'])
    
    print(f"Model A: {ma_seller_wins}/{len(model_a_seller)} as seller ({ma_seller_wins/len(model_a_seller):.1%}), "
          f"{ma_buyer_wins}/{len(model_a_buyer)} as buyer ({ma_buyer_wins/len(model_a_buyer):.1%})")
    print(f"Model B: {mb_seller_wins}/{len(model_b_seller)} as seller ({mb_seller_wins/len(model_b_seller):.1%}), "
          f"{mb_buyer_wins}/{len(model_b_buyer)} as buyer ({mb_buyer_wins/len(model_b_buyer):.1%})")
    
    # 2. Chi-Square Test for Role Independence
    print(f"\n2. CHI-SQUARE TEST FOR ROLE INDEPENDENCE")
    print("-" * 50)
    
    # Test if model performance is independent of role
    # H0: Model performance is independent of role
    # H1: Model performance depends on role
    
    role_performance_table = np.array([
        [ma_seller_wins, len(model_a_seller) - ma_seller_wins],  # Model A as seller: wins, losses
        [ma_buyer_wins, len(model_a_buyer) - ma_buyer_wins],     # Model A as buyer: wins, losses
        [mb_seller_wins, len(model_b_seller) - mb_seller_wins],  # Model B as seller: wins, losses
        [mb_buyer_wins, len(model_b_buyer) - mb_buyer_wins]      # Model B as buyer: wins, losses
    ])
    
    chi2, p_role_independence, dof, expected = chi2_contingency(role_performance_table)
    
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_role_independence:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"Result: {'Significant' if p_role_independence < 0.05 else 'Not significant'} role effect (α = 0.05)")
    
    # 3. Role-Controlled Model Comparison
    print(f"\n3. ROLE-CONTROLLED MODEL COMPARISON")
    print("-" * 45)
    
    # Compare models within each role separately
    # Seller role comparison
    seller_contingency = np.array([
        [ma_seller_wins, len(model_a_seller) - ma_seller_wins],
        [mb_seller_wins, len(model_b_seller) - mb_seller_wins]
    ])
    
    chi2_seller, p_seller, _, _ = chi2_contingency(seller_contingency)
    
    # Buyer role comparison
    buyer_contingency = np.array([
        [ma_buyer_wins, len(model_a_buyer) - ma_buyer_wins],
        [mb_buyer_wins, len(model_b_buyer) - mb_buyer_wins]
    ])
    
    chi2_buyer, p_buyer, _, _ = chi2_contingency(buyer_contingency)
    
    print(f"Seller Role Comparison:")
    print(f"  Chi-square: {chi2_seller:.4f}, p-value: {p_seller:.4f}")
    print(f"  Result: Model B {'significantly' if p_seller < 0.05 else 'not significantly'} better as seller")
    
    print(f"Buyer Role Comparison:")
    print(f"  Chi-square: {chi2_buyer:.4f}, p-value: {p_buyer:.4f}")
    print(f"  Result: Model B {'significantly' if p_buyer < 0.05 else 'not significantly'} better as buyer")
    
    # 4. Overall Model Performance (Role-Adjusted)
    print(f"\n4. OVERALL MODEL PERFORMANCE (ROLE-ADJUSTED)")
    print("-" * 55)
    
    model_a_total_wins = ma_seller_wins + ma_buyer_wins
    model_a_total_games = len(model_a_seller) + len(model_a_buyer)
    model_b_total_wins = mb_seller_wins + mb_buyer_wins
    model_b_total_games = len(model_b_seller) + len(model_b_buyer)
    
    overall_contingency = np.array([
        [model_a_total_wins, model_a_total_games - model_a_total_wins],
        [model_b_total_wins, model_b_total_games - model_b_total_wins]
    ])
    
    chi2_overall, p_overall, _, _ = chi2_contingency(overall_contingency)
    
    print(f"Overall Win Rates:")
    print(f"  Model A: {model_a_total_wins}/{model_a_total_games} ({model_a_total_wins/model_a_total_games:.1%})")
    print(f"  Model B: {model_b_total_wins}/{model_b_total_games} ({model_b_total_wins/model_b_total_games:.1%})")
    print(f"Chi-square: {chi2_overall:.4f}, p-value: {p_overall:.4f}")
    print(f"Result: Model B {'significantly' if p_overall < 0.05 else 'not significantly'} better overall")
    
    # 5. Utility Analysis
    print(f"\n5. UTILITY ANALYSIS")
    print("-" * 25)
    
    model_a_utilities = df['model_a_utility'].values
    model_b_utilities = df['model_b_utility'].values
    
    # T-test for utility differences
    t_stat, p_utility = ttest_ind(model_a_utilities, model_b_utilities)
    
    print(f"Model A average utility: {np.mean(model_a_utilities):.1f} ± {np.std(model_a_utilities):.1f}")
    print(f"Model B average utility: {np.mean(model_b_utilities):.1f} ± {np.std(model_b_utilities):.1f}")
    print(f"T-test: t = {t_stat:.4f}, p-value = {p_utility:.4f}")
    print(f"Result: {'Significant' if p_utility < 0.05 else 'Not significant'} utility difference")
    
    # 6. Role Bias Analysis
    print(f"\n6. ROLE BIAS ANALYSIS")
    print("-" * 30)
    
    # Calculate role advantage for each model
    ma_seller_rate = ma_seller_wins / len(model_a_seller) if len(model_a_seller) > 0 else 0
    ma_buyer_rate = ma_buyer_wins / len(model_a_buyer) if len(model_a_buyer) > 0 else 0
    mb_seller_rate = mb_seller_wins / len(model_b_seller) if len(model_b_seller) > 0 else 0
    mb_buyer_rate = mb_buyer_wins / len(model_b_buyer) if len(model_b_buyer) > 0 else 0
    
    ma_role_bias = ma_buyer_rate - ma_seller_rate
    mb_role_bias = mb_buyer_rate - mb_seller_rate
    
    print(f"Model A role bias: {ma_role_bias:+.1%} (buyer advantage)")
    print(f"Model B role bias: {mb_role_bias:+.1%} (buyer advantage)")
    print(f"Bias consistency: {'Yes' if abs(ma_role_bias - mb_role_bias) < 0.1 else 'No'} (difference: {abs(ma_role_bias - mb_role_bias):.1%})")
    
    # 7. Summary and Interpretation
    print(f"\n7. STATISTICAL SUMMARY")
    print("-" * 30)
    print(f"Sample size: {len(games_data)} games")
    print(f"Role distribution: Model A played {len(model_a_seller)} seller, {len(model_a_buyer)} buyer roles")
    print(f"Power analysis: {'Adequate' if len(games_data) >= 30 else 'Limited'} sample size for statistical tests")
    
    print(f"\nKey Findings:")
    if p_role_independence < 0.05:
        print(f"✓ Significant role effect detected (p = {p_role_independence:.4f})")
    else:
        print(f"✗ No significant role effect (p = {p_role_independence:.4f})")
        
    if p_overall < 0.05:
        print(f"✓ Model B significantly outperforms Model A (p = {p_overall:.4f})")
    else:
        print(f"✗ No significant difference between models (p = {p_overall:.4f})")
        
    if abs(ma_role_bias - mb_role_bias) < 0.1:
        print(f"✓ Consistent role bias across models (both favor buyer role)")
    else:
        print(f"✗ Inconsistent role bias between models")
    
    return {
        'role_independence_p': p_role_independence,
        'overall_model_diff_p': p_overall,
        'seller_comparison_p': p_seller,
        'buyer_comparison_p': p_buyer,
        'utility_diff_p': p_utility,
        'model_a_role_bias': ma_role_bias,
        'model_b_role_bias': mb_role_bias
    }

def analyze_metrics_performance(games_data):
    """Analyze performance across all metrics with role-controlled comparison."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS ANALYSIS")
    print("="*80)
    
    metrics = ['utility', 'risk', 'deadline', 'feasibility']
    metric_names = {
        'utility': 'Utility Scores',
        'risk': 'Risk Minimization',
        'deadline': 'Deadline Sensitivity', 
        'feasibility': 'Feasibility Scores'
    }
    
    results = {}
    
    for metric in metrics:
        print(f"\n{metric_names[metric].upper()} ANALYSIS")
        print("-" * 60)
        
        # Extract metric values for each model
        if metric == 'utility':
            model_a_values = [game['model_a_utility'] for game in games_data]
            model_b_values = [game['model_b_utility'] for game in games_data]
        else:
            model_a_values = [game[f'model_a_{metric}'] for game in games_data]
            model_b_values = [game[f'model_b_{metric}'] for game in games_data]
        
        # Calculate descriptive statistics
        model_a_mean = np.mean(model_a_values)
        model_b_mean = np.mean(model_b_values)
        model_a_std = np.std(model_a_values, ddof=1)
        model_b_std = np.std(model_b_values, ddof=1)
        
        print(f"Model A (Llama-3.1-8B) {metric_names[metric]}:")
        print(f"  Mean: {model_a_mean:.4f} (±{model_a_std:.4f})")
        print(f"  Range: [{min(model_a_values):.4f}, {max(model_a_values):.4f}]")
        
        print(f"Model B (Mistral-7B) {metric_names[metric]}:")
        print(f"  Mean: {model_b_mean:.4f} (±{model_b_std:.4f})")
        print(f"  Range: [{min(model_b_values):.4f}, {max(model_b_values):.4f}]")
        
        # Perform t-test
        t_stat, p_value = stats.ttest_rel(model_b_values, model_a_values)
        effect_size = (model_b_mean - model_a_mean) / np.sqrt((model_a_std**2 + model_b_std**2) / 2)
        
        print(f"\nStatistical Comparison:")
        print(f"  Difference (B - A): {model_b_mean - model_a_mean:+.4f}")
        if model_a_mean != 0:
            print(f"  Relative improvement: {((model_b_mean - model_a_mean) / model_a_mean * 100):+.2f}%")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        
        # Store results
        results[f'{metric}_t_stat'] = t_stat
        results[f'{metric}_p'] = p_value
        results[f'{metric}_effect_size'] = effect_size
        
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = ""
        
        if p_value < 0.05:
            better_model = "Model B" if model_b_mean > model_a_mean else "Model A"
            print(f"  Result: {better_model} significantly outperforms {significance}")
        else:
            print(f"  Result: No significant difference")
        
        # Role-based analysis
        print(f"\nRole-Based {metric_names[metric]} Analysis:")
        
        # When Model A is buyer vs seller
        model_a_as_buyer = [game[f'model_a_{metric}'] for game in games_data if game['model_a_role'] == 'buyer']
        model_a_as_seller = [game[f'model_a_{metric}'] for game in games_data if game['model_a_role'] == 'seller']
        
        # When Model B is buyer vs seller  
        model_b_as_buyer = [game[f'model_b_{metric}'] for game in games_data if game['model_b_role'] == 'buyer']
        model_b_as_seller = [game[f'model_b_{metric}'] for game in games_data if game['model_b_role'] == 'seller']
        
        if len(model_a_as_buyer) > 0:
            print(f"  Model A as Buyer: {np.mean(model_a_as_buyer):.4f} (n={len(model_a_as_buyer)})")
        if len(model_a_as_seller) > 0:
            print(f"  Model A as Seller: {np.mean(model_a_as_seller):.4f} (n={len(model_a_as_seller)})")
        if len(model_b_as_buyer) > 0:
            print(f"  Model B as Buyer: {np.mean(model_b_as_buyer):.4f} (n={len(model_b_as_buyer)})")
        if len(model_b_as_seller) > 0:
            print(f"  Model B as Seller: {np.mean(model_b_as_seller):.4f} (n={len(model_b_as_seller)})")
        
        # Test for role independence for each model
        if len(model_a_as_buyer) > 0 and len(model_a_as_seller) > 0:
            t_stat_a, p_val_a = stats.ttest_ind(model_a_as_buyer, model_a_as_seller)
            print(f"  Model A role independence: t={t_stat_a:.3f}, p={p_val_a:.4f}")
        
        if len(model_b_as_buyer) > 0 and len(model_b_as_seller) > 0:
            t_stat_b, p_val_b = stats.ttest_ind(model_b_as_buyer, model_b_as_seller)
            print(f"  Model B role independence: t={t_stat_b:.3f}, p={p_val_b:.4f}")
    
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python statistical_role_analysis.py <output_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        print("COMPREHENSIVE STATISTICAL ROLE ANALYSIS")
        print("="*80)
        
        # Extract detailed game data
        games_data = extract_detailed_game_data(filename)
        print(f"Extracted data for {len(games_data)} games")
        
        # Perform existing role-controlled analysis
        results = role_controlled_analysis(games_data)
        
        # NEW: Comprehensive metrics analysis
        metrics_results = analyze_metrics_performance(games_data)
        
        # Combine results
        results.update(metrics_results)
        
        print(f"\n=== COMPREHENSIVE CONCLUSIONS ===")
        print(f"Statistical analysis complete. Key p-values:")
        for key, value in results.items():
            if key.endswith('_p'):
                print(f"  {key}: {value:.6f}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()