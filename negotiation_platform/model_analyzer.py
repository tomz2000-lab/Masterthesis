#!/usr/bin/env python3
"""
Model Performance Analyzer
==========================

Analyze and visualize results from batch model comparison runs.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any


class ModelAnalyzer:
    """Analyze batch comparison results"""
    
    def __init__(self, results_file: str):
        """Load results from JSON file"""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
    
    def create_performance_dataframe(self) -> pd.DataFrame:
        """Create a structured DataFrame from batch results"""
        rows = []
        
        for game_type, game_results in self.results.items():
            for model_pair, batch_result in game_results.items():
                if not isinstance(batch_result, dict):
                    continue
                    
                aggregated = batch_result.get("aggregated_metrics", {})
                
                # Extract model performance data
                model_a_perf = aggregated.get("model_a_performance", {})
                model_b_perf = aggregated.get("model_b_performance", {})
                
                # Model A row
                rows.append({
                    "game_type": game_type,
                    "model_pair": model_pair,
                    "model": "model_a",
                    "model_name": "Llama-3.2-3B",
                    "success_rate": batch_result.get("success_rate", 0),
                    "iterations": batch_result.get("iterations", 0),
                    "utility_surplus_mean": model_a_perf.get("utility_surplus", {}).get("mean", 0),
                    "utility_surplus_std": model_a_perf.get("utility_surplus", {}).get("std", 0),
                    "risk_minimization_mean": model_a_perf.get("risk_minimization", {}).get("mean", 0),
                    "deadline_sensitivity_mean": model_a_perf.get("deadline_sensitivity", {}).get("mean", 0),
                    "feasibility_mean": model_a_perf.get("feasibility", {}).get("mean", 0),
                    "mean_agreement_round": aggregated.get("mean_agreement_round", 0)
                })
                
                # Model B row
                rows.append({
                    "game_type": game_type,
                    "model_pair": model_pair,
                    "model": "model_b",
                    "model_name": "Llama-3.1-8B",
                    "success_rate": batch_result.get("success_rate", 0),
                    "iterations": batch_result.get("iterations", 0),
                    "utility_surplus_mean": model_b_perf.get("utility_surplus", {}).get("mean", 0),
                    "utility_surplus_std": model_b_perf.get("utility_surplus", {}).get("std", 0),
                    "risk_minimization_mean": model_b_perf.get("risk_minimization", {}).get("mean", 0),
                    "deadline_sensitivity_mean": model_b_perf.get("deadline_sensitivity", {}).get("mean", 0),
                    "feasibility_mean": model_b_perf.get("feasibility", {}).get("mean", 0),
                    "mean_agreement_round": aggregated.get("mean_agreement_round", 0)
                })
        
        return pd.DataFrame(rows)
    
    def create_comparison_plots(self, df: pd.DataFrame, output_dir: str = "plots"):
        """Create comparison plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        
        # 1. Utility Surplus Comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        games = df['game_type'].unique()
        for i, game in enumerate(games):
            game_df = df[df['game_type'] == game]
            
            models = game_df['model_name'].unique()
            surplus_means = []
            surplus_stds = []
            
            for model in models:
                model_data = game_df[game_df['model_name'] == model]
                surplus_means.append(model_data['utility_surplus_mean'].iloc[0])
                surplus_stds.append(model_data['utility_surplus_std'].iloc[0])
            
            x_pos = np.arange(len(models))
            axes[i].bar(x_pos, surplus_means, yerr=surplus_stds, capsize=5, alpha=0.7)
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel('Utility Surplus')
            axes[i].set_title(f'{game.replace("_", " ").title()}')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels([m.split('-')[1] for m in models], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/utility_surplus_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Success Rate Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        game_success = {}
        for game in games:
            game_df = df[df['game_type'] == game]
            game_success[game] = game_df['success_rate'].iloc[0]  # Same for both models
        
        bars = ax.bar(range(len(game_success)), list(game_success.values()), alpha=0.7)
        ax.set_xlabel('Game Type')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Game Type')
        ax.set_xticks(range(len(game_success)))
        ax.set_xticklabels([g.replace('_', ' ').title() for g in game_success.keys()], rotation=45)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, game_success.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/success_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Model Performance Heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a matrix for heatmap
        metrics = ['utility_surplus_mean', 'risk_minimization_mean', 'deadline_sensitivity_mean', 'feasibility_mean']
        heatmap_data = []
        row_labels = []
        
        for game in games:
            for model_name in df['model_name'].unique():
                subset = df[(df['game_type'] == game) & (df['model_name'] == model_name)]
                if not subset.empty:
                    row = [subset[metric].iloc[0] for metric in metrics]
                    heatmap_data.append(row)
                    row_labels.append(f"{game.replace('_', ' ').title()}\n{model_name.split('-')[1]}")
        
        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add value annotations
        for i in range(len(row_labels)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.2f}', 
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title("Model Performance Heatmap")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {output_dir}/")
    
    def generate_statistical_report(self, df: pd.DataFrame) -> str:
        """Generate statistical analysis report"""
        report = []
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 50)
        
        for game_type in df['game_type'].unique():
            game_df = df[df['game_type'] == game_type]
            report.append(f"\n{game_type.upper().replace('_', ' ')} GAME:")
            report.append("-" * 30)
            
            # Compare models
            model_3b = game_df[game_df['model_name'].str.contains('3B')]
            model_8b = game_df[game_df['model_name'].str.contains('8B')]
            
            if not model_3b.empty and not model_8b.empty:
                # Utility surplus comparison
                surplus_3b = model_3b['utility_surplus_mean'].iloc[0]
                surplus_8b = model_8b['utility_surplus_mean'].iloc[0]
                
                report.append(f"Utility Surplus:")
                report.append(f"  3B Model: {surplus_3b:.2f} ± {model_3b['utility_surplus_std'].iloc[0]:.2f}")
                report.append(f"  8B Model: {surplus_8b:.2f} ± {model_8b['utility_surplus_std'].iloc[0]:.2f}")
                report.append(f"  Difference: {surplus_8b - surplus_3b:.2f}")
                report.append(f"  Ratio (8B/3B): {surplus_8b/surplus_3b if surplus_3b != 0 else 'N/A':.2f}")
                
                # Other metrics
                report.append(f"Feasibility:")
                report.append(f"  3B Model: {model_3b['feasibility_mean'].iloc[0]:.3f}")
                report.append(f"  8B Model: {model_8b['feasibility_mean'].iloc[0]:.3f}")
                
                report.append(f"Agreement Round:")
                report.append(f"  Average: {model_3b['mean_agreement_round'].iloc[0]:.1f}")
        
        return "\n".join(report)


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model comparison results")
    parser.add_argument("results_file", help="JSON file with batch results")
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory for plots and reports")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ModelAnalyzer(args.results_file)
    
    # Create DataFrame
    df = analyzer.create_performance_dataframe()
    
    # Generate plots
    analyzer.create_comparison_plots(df, args.output_dir)
    
    # Generate statistical report
    statistical_report = analyzer.generate_statistical_report(df)
    
    # Save report
    Path(args.output_dir).mkdir(exist_ok=True)
    with open(f"{args.output_dir}/statistical_analysis.txt", 'w') as f:
        f.write(statistical_report)
    
    # Print report
    print(statistical_report)
    
    # Save DataFrame
    df.to_csv(f"{args.output_dir}/model_comparison_data.csv", index=False)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()