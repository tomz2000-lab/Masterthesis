"""
Simple Utility Surplus Tracker
=============================
Tracks utility surplus by model and game type only.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any
import os

class SimpleUtilityTracker:
    """Simple tracker for utility surplus: model × game × metric"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.data_file = os.path.join(results_dir, "utility_surplus_data.csv")
        self.ensure_data_file()
    
    def ensure_data_file(self):
        """Create CSV file with minimal headers"""
        if not os.path.exists(self.data_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'model_id', 'game_type', 'utility_surplus', 'temperature'
            ])
            df.to_csv(self.data_file, index=False)
    
    def record_result(self, session_data: Dict[str, Any]):
        """Record utility surplus results - simplified"""
        timestamp = datetime.now().isoformat()
        game_type = session_data.get('game_type', 'unknown')
        
        # Extract utility surplus data
        surplus_data = session_data.get('metrics', {}).get('utility_surplus', {})
        model_configs = session_data.get('model_configs', {})
        
        records = []
        for model_id, surplus in surplus_data.items():
            # Get temperature from model config
            temperature = 0.5  # default
            if model_id in model_configs:
                temperature = model_configs[model_id].get('config', {}).get('temperature', 0.5)
            
            records.append({
                'timestamp': timestamp,
                'model_id': model_id,
                'game_type': game_type,
                'utility_surplus': surplus,
                'temperature': temperature
            })
        
        # Save to CSV
        if records:
            df_new = pd.DataFrame(records)
            df_new.to_csv(self.data_file, mode='a', header=False, index=False)
            print(f"📊 Recorded utility surplus for {len(records)} models")
    
    def load_data(self) -> pd.DataFrame:
        """Load all data"""
        if os.path.exists(self.data_file):
            return pd.read_csv(self.data_file)
        return pd.DataFrame()
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get simple model × game performance table"""
        df = self.load_data()
        if df.empty:
            return pd.DataFrame()
        
        # Create pivot table: models as rows, games as columns
        summary = df.groupby(['model_id', 'game_type'])['utility_surplus'].mean().unstack(fill_value=0)
        return summary
    
    def generate_simple_report(self) -> str:
        """Generate simple report"""
        df = self.load_data()
        if df.empty:
            return "No data available"
        
        summary = self.get_summary_table()
        
        report = f"""
=== UTILITY SURPLUS PERFORMANCE ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE BY MODEL & GAME:
{summary.to_string()}

AVERAGE BY MODEL:
{df.groupby('model_id')['utility_surplus'].mean().to_string()}

AVERAGE BY GAME:
{df.groupby('game_type')['utility_surplus'].mean().to_string()}
"""
        return report

# Example usage
if __name__ == "__main__":
    tracker = SimpleUtilityTracker()
    
    # Mock data
    mock_session = {
        'game_type': 'resource_allocation',
        'metrics': {
            'utility_surplus': {'model_a': 45.2, 'model_b': 32.8}
        },
        'model_configs': {
            'model_a': {'config': {'temperature': 0.7}},
            'model_b': {'config': {'temperature': 0.5}}
        }
    }
    
    # tracker.record_result(mock_session)
    # print(tracker.generate_simple_report())
