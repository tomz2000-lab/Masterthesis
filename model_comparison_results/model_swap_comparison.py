#!/usr/bin/env python3
"""
Compare original vs swapped model results to test for positional bias
and demonstrate model performance changes.
"""

import re
from pathlib import Path
import json

def extract_game_results(file_path, game_type):
    """Extract all game results from output file"""
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all final utilities directly
    utilities_matches = re.findall(r"Final utilities: \{'model_a': ([\d.]+), 'model_b': ([\d.]+)\}", content)
    
    for i, (model_a_points_str, model_b_points_str) in enumerate(utilities_matches, 1):
        model_a_points = float(model_a_points_str)
        model_b_points = float(model_b_points_str)
        
        # Determine winner
        if model_a_points > model_b_points:
            winner = 'Model A'
            margin = model_a_points - model_b_points
        elif model_b_points > model_a_points:
            winner = 'Model B'
            margin = model_b_points - model_a_points
        else:
            winner = 'Tie'
            margin = 0
        
        results.append({
            'game': i,
            'model_a_points': model_a_points,
            'model_b_points': model_b_points,
            'winner': winner,
            'margin': margin
        })
    
    return results

def analyze_model_comparison():
    """Compare original vs swapped results across all three games"""
    
    print("=" * 80)
    print("MODEL POSITION SWAP ANALYSIS")
    print("Testing if model performance depends on model choice vs position bias")
    print("=" * 80)
    
    games = [
        ("company_car", "Company Car"),
        ("integrative_negotiation", "Integrative Negotiations"),
        ("resource_allocation", "Resource Allocation")
    ]
    
    overall_results = {
        "original": {"3B_wins": 0, "8B_wins": 0, "ties": 0},
        "swapped": {"3B_wins": 0, "8B_wins": 0, "ties": 0}
    }
    
    for game_key, game_name in games:
        print(f"\n{'='*20} {game_name.upper()} {'='*20}")
        
        # Original results
        original_file = Path(f"batch_run_1/{game_key}_output.txt")
        swapped_file = Path(f"batch_run_swapped/{game_key}_swapped.out")
        
        if not original_file.exists():
            print(f"❌ Original file not found: {original_file}")
            continue
            
        if not swapped_file.exists():
            print(f"❌ Swapped file not found: {swapped_file}")
            continue
        
        # Extract results
        original_results = extract_game_results(original_file, game_key)
        swapped_results = extract_game_results(swapped_file, game_key)
        
        if not original_results or not swapped_results:
            print(f"❌ No results found for {game_name}")
            continue
        
        print(f"\nORIGINAL CONFIGURATION:")
        print(f"Model A: Llama-3.2-3B-Instruct")
        print(f"Model B: Llama-3.1-8B-Instruct")
        
        original_a_wins = sum(1 for r in original_results if r['winner'] == 'Model A')
        original_b_wins = sum(1 for r in original_results if r['winner'] == 'Model B')
        original_ties = sum(1 for r in original_results if r['winner'] == 'Tie')
        
        print(f"Results: Model A (3B) wins: {original_a_wins}, Model B (8B) wins: {original_b_wins}, Ties: {original_ties}")
        
        # Show sample scores
        if original_results:
            sample = original_results[0]
            print(f"Sample score: Model A: {sample['model_a_points']}, Model B: {sample['model_b_points']}")
        
        print(f"\nSWAPPED CONFIGURATION:")
        print(f"Model A: Llama-3.1-8B-Instruct")  
        print(f"Model B: Llama-3.2-3B-Instruct")
        
        swapped_a_wins = sum(1 for r in swapped_results if r['winner'] == 'Model A')
        swapped_b_wins = sum(1 for r in swapped_results if r['winner'] == 'Model B')
        swapped_ties = sum(1 for r in swapped_results if r['winner'] == 'Tie')
        
        print(f"Results: Model A (8B) wins: {swapped_a_wins}, Model B (3B) wins: {swapped_b_wins}, Ties: {swapped_ties}")
        
        # Show sample scores
        if swapped_results:
            sample = swapped_results[0]
            print(f"Sample score: Model A: {sample['model_a_points']}, Model B: {sample['model_b_points']}")
        
        # Analysis
        print(f"\n🔍 ANALYSIS:")
        
        # Track which model (3B or 8B) won in each configuration
        original_3b_wins = original_a_wins  # 3B was Model A in original
        original_8b_wins = original_b_wins  # 8B was Model B in original
        
        swapped_3b_wins = swapped_b_wins    # 3B is Model B in swapped
        swapped_8b_wins = swapped_a_wins    # 8B is Model A in swapped
        
        print(f"3B model performance: Original: {original_3b_wins} wins, Swapped: {swapped_3b_wins} wins")
        print(f"8B model performance: Original: {original_8b_wins} wins, Swapped: {swapped_8b_wins} wins")
        
        # Determine which model consistently performs better
        if original_3b_wins > original_8b_wins and swapped_3b_wins > swapped_8b_wins:
            print(f"✅ 3B MODEL DOMINATES - wins regardless of position")
            dominant_model = "3B"
        elif original_8b_wins > original_3b_wins and swapped_8b_wins > swapped_3b_wins:
            print(f"✅ 8B MODEL DOMINATES - wins regardless of position") 
            dominant_model = "8B"
        elif original_3b_wins > original_8b_wins and swapped_8b_wins > swapped_3b_wins:
            print(f"⚠️  RESULTS FLIPPED - indicates positional bias")
            dominant_model = "BIAS"
        elif original_8b_wins > original_3b_wins and swapped_3b_wins > swapped_8b_wins:
            print(f"⚠️  RESULTS FLIPPED - indicates positional bias")
            dominant_model = "BIAS"
        else:
            print(f"🤔 MIXED RESULTS - unclear dominance")
            dominant_model = "MIXED"
        
        # Update overall tracking
        overall_results["original"]["3B_wins"] += original_3b_wins
        overall_results["original"]["8B_wins"] += original_8b_wins
        overall_results["original"]["ties"] += original_ties
        
        overall_results["swapped"]["3B_wins"] += swapped_3b_wins
        overall_results["swapped"]["8B_wins"] += swapped_8b_wins
        overall_results["swapped"]["ties"] += swapped_ties
        
        print(f"🏆 {game_name}: {dominant_model} model advantage")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY ACROSS ALL GAMES")
    print(f"{'='*80}")
    
    orig_3b_total = overall_results["original"]["3B_wins"]
    orig_8b_total = overall_results["original"]["8B_wins"]
    swap_3b_total = overall_results["swapped"]["3B_wins"]
    swap_8b_total = overall_results["swapped"]["8B_wins"]
    
    print(f"\nORIGINAL CONFIGURATION TOTALS:")
    print(f"3B model: {orig_3b_total} wins")
    print(f"8B model: {orig_8b_total} wins")
    print(f"3B win rate: {orig_3b_total/(orig_3b_total+orig_8b_total)*100:.1f}%")
    
    print(f"\nSWAPPED CONFIGURATION TOTALS:")
    print(f"3B model: {swap_3b_total} wins")
    print(f"8B model: {swap_8b_total} wins") 
    print(f"3B win rate: {swap_3b_total/(swap_3b_total+swap_8b_total)*100:.1f}%")
    
    print(f"\n🎯 FINAL CONCLUSION:")
    if (orig_3b_total > orig_8b_total) == (swap_3b_total > swap_8b_total):
        if orig_3b_total > orig_8b_total:
            print("✅ 3B MODEL IS CONSISTENTLY SUPERIOR - No positional bias detected")
        else:
            print("✅ 8B MODEL IS CONSISTENTLY SUPERIOR - No positional bias detected")
        print("🔬 Model performance depends on MODEL CHOICE, not position")
    else:
        print("⚠️  SIGNIFICANT POSITIONAL BIAS DETECTED")
        print("🔬 Results may be influenced by player position, not just model choice")
    
    # Calculate consistency
    total_games = orig_3b_total + orig_8b_total
    consistency = abs(orig_3b_total - swap_3b_total) / total_games * 100
    print(f"📊 Result consistency: {100-consistency:.1f}% (lower = more bias)")

if __name__ == "__main__":
    analyze_model_comparison()