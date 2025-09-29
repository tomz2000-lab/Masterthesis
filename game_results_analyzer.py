#!/usr/bin/env python3
"""
Game Results Analyzer and Visualizer
Creates graphs and comparisons from logged game results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import re
from typing import Dict, List, Any

class GameResultsAnalyzer:
    def __init__(self, log_file: str = "game_results_log.csv"):
        self.log_file = Path(log_file)
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the logged game results."""
        try:
            self.df = pd.read_csv(self.log_file)
            print(f"📊 Loaded {len(self.df)} game results from {self.log_file}")
        except FileNotFoundError:
            print(f"❌ Log file not found: {self.log_file}")
            self.df = pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def clean_data(self):
        """Clean and prepare data for analysis."""
        if self.df.empty:
            return
        
        # Convert timestamps
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        
        # Extract model architectures and sizes
        for prefix in ['model_a', 'model_b']:
            name_col = f'{prefix}_name'
            if name_col in self.df.columns:
                # Extract architecture
                self.df[f'{prefix}_arch'] = self.df[name_col].str.extract(r'(llama|mistral)', flags=re.IGNORECASE)
                
                # Extract version for Mistral
                self.df[f'{prefix}_version'] = self.df[name_col].str.extract(r'v(\d+\.\d+)')
        
        print("✅ Data cleaned and prepared")
    
    def create_agreement_rate_chart(self):
        """Create chart showing agreement rates by model size/architecture."""
        if self.df.empty or 'agreement_reached' not in self.df.columns:
            print("⚠️ No agreement data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agreement Rates by Model Characteristics', fontsize=16)
        
        # By model size
        if 'model_a_size' in self.df.columns and 'model_b_size' in self.df.columns:
            size_data = []
            for prefix in ['model_a', 'model_b']:
                size_col = f'{prefix}_size'
                if size_col in self.df.columns:
                    temp_df = self.df.copy()
                    temp_df['model_size'] = temp_df[size_col]
                    temp_df['model_role'] = prefix
                    size_data.append(temp_df[['model_size', 'agreement_reached', 'model_role']])
            
            if size_data:
                combined_size = pd.concat(size_data, ignore_index=True)
                size_agreement = combined_size.groupby('model_size')['agreement_reached'].mean()
                
                axes[0, 0].bar(size_agreement.index, size_agreement.values * 100)
                axes[0, 0].set_title('Agreement Rate by Model Size')
                axes[0, 0].set_ylabel('Agreement Rate (%)')
                axes[0, 0].set_xlabel('Model Size')
        
        # By game type
        if 'game_type' in self.df.columns:
            game_agreement = self.df.groupby('game_type')['agreement_reached'].mean()
            axes[0, 1].bar(game_agreement.index, game_agreement.values * 100)
            axes[0, 1].set_title('Agreement Rate by Game Type')
            axes[0, 1].set_ylabel('Agreement Rate (%)')
            axes[0, 1].set_xlabel('Game Type')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Agreement rounds distribution
        if 'agreement_round' in self.df.columns:
            agreement_data = self.df[self.df['agreement_reached'] == True]['agreement_round']
            axes[1, 0].hist(agreement_data, bins=range(1, int(agreement_data.max()) + 2), alpha=0.7)
            axes[1, 0].set_title('Distribution of Agreement Rounds')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_xlabel('Round Number')
        
        # Utility comparison
        if 'model_a_final_utility' in self.df.columns and 'model_b_final_utility' in self.df.columns:
            axes[1, 1].scatter(self.df['model_a_final_utility'], self.df['model_b_final_utility'], alpha=0.6)
            axes[1, 1].set_title('Model A vs Model B Final Utilities')
            axes[1, 1].set_xlabel('Model A Utility')
            axes[1, 1].set_ylabel('Model B Utility')
            
            # Add diagonal line for reference
            max_util = max(self.df['model_a_final_utility'].max(), self.df['model_b_final_utility'].max())
            axes[1, 1].plot([0, max_util], [0, max_util], 'r--', alpha=0.5, label='Equal utility')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('agreement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Saved chart as: agreement_analysis.png")
    
    def create_model_comparison_chart(self):
        """Create detailed model comparison charts."""
        if self.df.empty:
            print("⚠️ No data to plot")
            return
        
        # Create model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Utility surplus comparison
        utility_cols = ['model_a_utility_surplus', 'model_b_utility_surplus']
        if all(col in self.df.columns for col in utility_cols):
            model_names_a = self.df['model_a_name'].unique()
            model_names_b = self.df['model_b_name'].unique()
            
            # Group by model name and calculate average utility surplus
            avg_utilities = {}
            for _, row in self.df.iterrows():
                for model_col, name_col in [('model_a', 'model_a_name'), ('model_b', 'model_b_name')]:
                    model_name = row[name_col]
                    utility_col = f'{model_col}_utility_surplus'
                    if pd.notna(row[utility_col]) and pd.notna(model_name):
                        if model_name not in avg_utilities:
                            avg_utilities[model_name] = []
                        avg_utilities[model_name].append(row[utility_col])
            
            # Calculate averages
            model_names = list(avg_utilities.keys())
            avg_values = [np.mean(values) for values in avg_utilities.values()]
            
            if model_names and avg_values:
                axes[0, 0].bar(range(len(model_names)), avg_values)
                axes[0, 0].set_title('Average Utility Surplus by Model')
                axes[0, 0].set_ylabel('Utility Surplus')
                axes[0, 0].set_xticks(range(len(model_names)))
                axes[0, 0].set_xticklabels([name.split('/')[-1] for name in model_names], rotation=45)
        
        # Agreement success rate by model
        success_rates = {}
        for _, row in self.df.iterrows():
            for name_col in ['model_a_name', 'model_b_name']:
                model_name = row[name_col]
                if pd.notna(model_name):
                    if model_name not in success_rates:
                        success_rates[model_name] = {'total': 0, 'success': 0}
                    success_rates[model_name]['total'] += 1
                    if row['agreement_reached']:
                        success_rates[model_name]['success'] += 1
        
        if success_rates:
            model_names = list(success_rates.keys())
            success_percentages = [(success_rates[name]['success'] / success_rates[name]['total'] * 100) 
                                 for name in model_names]
            
            axes[0, 1].bar(range(len(model_names)), success_percentages)
            axes[0, 1].set_title('Success Rate by Model')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].set_xticks(range(len(model_names)))
            axes[0, 1].set_xticklabels([name.split('/')[-1] for name in model_names], rotation=45)
        
        # Model size comparison
        if 'model_a_size' in self.df.columns and 'model_b_size' in self.df.columns:
            size_success = {}
            size_utility = {}
            
            for _, row in self.df.iterrows():
                for size_col, utility_col in [('model_a_size', 'model_a_final_utility'), 
                                             ('model_b_size', 'model_b_final_utility')]:
                    size = row[size_col]
                    utility = row[utility_col]
                    
                    if pd.notna(size):
                        # Success rate
                        if size not in size_success:
                            size_success[size] = {'total': 0, 'success': 0}
                        size_success[size]['total'] += 1
                        if row['agreement_reached']:
                            size_success[size]['success'] += 1
                        
                        # Average utility
                        if pd.notna(utility):
                            if size not in size_utility:
                                size_utility[size] = []
                            size_utility[size].append(utility)
            
            # Plot size success rates
            if size_success:
                sizes = list(size_success.keys())
                success_rates = [(size_success[size]['success'] / size_success[size]['total'] * 100) 
                               for size in sizes]
                axes[1, 0].bar(sizes, success_rates)
                axes[1, 0].set_title('Success Rate by Model Size')
                axes[1, 0].set_ylabel('Success Rate (%)')
                axes[1, 0].set_xlabel('Model Size')
            
            # Plot average utilities by size
            if size_utility:
                sizes = list(size_utility.keys())
                avg_utilities = [np.mean(size_utility[size]) for size in sizes]
                axes[1, 1].bar(sizes, avg_utilities)
                axes[1, 1].set_title('Average Utility by Model Size')
                axes[1, 1].set_ylabel('Average Final Utility')
                axes[1, 1].set_xlabel('Model Size')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Saved chart as: model_comparison.png")
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive text summary report."""
        if self.df.empty:
            return "No data available for analysis."
        
        report_lines = [
            "🔬 GAME RESULTS ANALYSIS REPORT",
            "=" * 50,
            f"📊 Total Games Analyzed: {len(self.df)}",
            f"📅 Date Range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}",
            ""
        ]
        
        # Overall statistics
        if 'agreement_reached' in self.df.columns:
            overall_agreement = self.df['agreement_reached'].mean() * 100
            report_lines.append(f"🤝 Overall Agreement Rate: {overall_agreement:.1f}%")
        
        if 'agreement_round' in self.df.columns:
            avg_rounds = self.df[self.df['agreement_reached'] == True]['agreement_round'].mean()
            report_lines.append(f"⚡ Average Rounds to Agreement: {avg_rounds:.1f}")
        
        report_lines.append("")
        
        # Game type analysis
        if 'game_type' in self.df.columns:
            report_lines.append("🎮 PERFORMANCE BY GAME TYPE:")
            for game_type in self.df['game_type'].unique():
                game_data = self.df[self.df['game_type'] == game_type]
                agreement_rate = game_data['agreement_reached'].mean() * 100
                count = len(game_data)
                report_lines.append(f"   {game_type}: {agreement_rate:.1f}% success ({count} games)")
        
        # Model size analysis
        size_cols = [col for col in self.df.columns if col.endswith('_size')]
        if size_cols:
            report_lines.extend(["", "🧠 PERFORMANCE BY MODEL SIZE:"])
            
            # Combine all model sizes
            all_sizes = []
            for col in size_cols:
                all_sizes.extend(self.df[col].dropna().tolist())
            
            unique_sizes = set(all_sizes)
            for size in sorted(unique_sizes):
                # Count games involving this size
                size_games = 0
                size_agreements = 0
                
                for _, row in self.df.iterrows():
                    if row.get('model_a_size') == size or row.get('model_b_size') == size:
                        size_games += 1
                        if row['agreement_reached']:
                            size_agreements += 1
                
                if size_games > 0:
                    success_rate = (size_agreements / size_games) * 100
                    report_lines.append(f"   {size} models: {success_rate:.1f}% success ({size_games} games)")
        
        return "\\n".join(report_lines)
    
    def save_report(self, filename: str = "analysis_report.txt"):
        """Save the analysis report to a file."""
        report = self.generate_summary_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"💾 Saved analysis report as: {filename}")
        return filename

def main():
    """Main function for the analyzer."""
    analyzer = GameResultsAnalyzer()
    
    if analyzer.df.empty:
        print("❌ No data to analyze. Run some games and use simple_game_logger.py first!")
        return
    
    print("📈 Game Results Analyzer")
    print("=" * 30)
    
    analyzer.clean_data()
    
    while True:
        print("\\n📋 Analysis Options:")
        print("1. Generate agreement rate charts")
        print("2. Generate model comparison charts")
        print("3. Show summary report")
        print("4. Save analysis report")
        print("5. Export data to Excel")
        print("6. Exit")
        
        choice = input("\\n❓ Choose option (1-6): ").strip()
        
        if choice == "1":
            analyzer.create_agreement_rate_chart()
        elif choice == "2":
            analyzer.create_model_comparison_chart()
        elif choice == "3":
            print("\\n" + analyzer.generate_summary_report())
        elif choice == "4":
            analyzer.save_report()
        elif choice == "5":
            excel_file = "game_results_analysis.xlsx"
            analyzer.df.to_excel(excel_file, index=False)
            print(f"💾 Exported data to: {excel_file}")
        elif choice == "6":
            print("👋 Analysis complete!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()