#!/usr/bin/env python3
"""
Model Output Analyzer
=====================

Analyzes the debug output from negotiation runs to show what each model is doing.
"""
import re
import sys
import os
from collections import defaultdict

def analyze_negotiation_output(filename):
    """Analyze negotiation output file to extract model-specific behaviors."""
    
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found")
        return
    
    with open(filename, 'r') as f:
        content = f.read()
    
    print(f"🔍 ANALYZING: {filename}")
    print("=" * 80)
    
    # Extract model configurations
    model_configs = re.findall(r"'(model_[abc])': {'type': 'huggingface', 'model_name': '([^']+)'", content)
    
    print("📋 MODEL CONFIGURATION:")
    print("-" * 40)
    for model_id, model_name in model_configs:
        print(f"  {model_id}: {model_name}")
    print()
    
    # Extract successful responses
    successful_responses = re.findall(r"Successfully parsed JSON from start: (\{[^}]+\})", content)
    failed_responses = len(re.findall(r"Could not parse action from response:", content))
    
    print("📊 RESPONSE ANALYSIS:")
    print("-" * 40)
    print(f"✅ Successful JSON responses: {len(successful_responses)}")
    print(f"❌ Failed responses: {failed_responses}")
    print()
    
    if successful_responses:
        print("💰 SUCCESSFUL OFFERS:")
        print("-" * 40)
        for i, response in enumerate(successful_responses, 1):
            print(f"  Round {i}: {response}")
        print()
    
    # Extract round-by-round analysis
    rounds = re.findall(r"Round (\d+)/(\d+)", content)
    if rounds:
        print(f"🎯 NEGOTIATION PROGRESS:")
        print("-" * 40)
        print(f"  Total rounds: {rounds[0][1] if rounds else 'Unknown'}")
        print(f"  Rounds played: {len(rounds)}")
        print()
    
    # Check for agreement
    agreement = re.search(r"Agreement reached: (True|False)", content)
    if agreement:
        print("🤝 FINAL OUTCOME:")
        print("-" * 40)
        print(f"  Agreement reached: {agreement.group(1)}")
        
        # Extract metrics if available
        metrics_match = re.search(r"Metrics: (\{[^}]+\})", content)
        if metrics_match:
            print(f"  Metrics: {metrics_match.group(1)}")
        print()
    
    # Identify model behavior patterns
    print("🧠 MODEL BEHAVIOR ANALYSIS:")
    print("-" * 40)
    
    # Count patterns for each model
    llama_patterns = len(re.findall(r"Extracted JSON from start", content))
    empty_patterns = len(re.findall(r"Cleaned response: ''", content))
    
    print(f"  Model A {model_name}: Generated {llama_patterns} perfect JSON responses")
    print(f"  Model B {model_name}: Generated {empty_patterns} empty responses")
    print()
    
    # Strategic analysis
    if successful_responses:
        prices = []
        for response in successful_responses:
            price_match = re.search(r'"price": (\d+)', response)
            if price_match:
                prices.append(int(price_match.group(1)))
        
        if prices:
            print("📈 STRATEGIC ANALYSIS:")
            print("-" * 40)
            print(f"  Price range: €{min(prices):,} - €{max(prices):,}")
            print(f"  Average price: €{sum(prices)//len(prices):,}")
            print(f"  Consistency: {'High' if len(set(prices)) <= 2 else 'Variable'}")
            
            # Economic zone analysis
            starting_price = 45000
            buyer_budget = 40000
            buyer_batna = 44000
            seller_batna = 39000
            
            avg_price = sum(prices) // len(prices)
            if seller_batna <= avg_price <= buyer_batna:
                print(f"  ✅ Prices in optimal zone (€{seller_batna:,} - €{buyer_batna:,})")
            else:
                print(f"  ⚠️  Prices outside optimal zone")

def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default to most recent file
        import glob
        files = glob.glob("negotiation_*.out")
        if files:
            filename = max(files, key=os.path.getctime)
            print(f"🔍 Using most recent file: {filename}")
        else:
            print("❌ No negotiation output files found")
            print("Usage: python analyze_model_outputs.py [filename]")
            return
    
    analyze_negotiation_output(filename)

if __name__ == "__main__":
    main()
