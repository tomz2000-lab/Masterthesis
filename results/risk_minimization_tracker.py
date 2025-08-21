"""
Simple Risk Tracker
==================
Tracks risk percentage by model and game type only.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any
import os

class SimpleRiskTracker:
    """Simple tracker for risk minimization: model × game × metric"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.data_file = os.path.join(results_dir, "risk_minimization_data.csv")
        self.ensure_data_file()
    
    def ensure_data_file(self):
        """Create CSV file with minimal headers"""
        if not os.path.exists(self.data_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'model_id', 'game_type', 'risk_percentage', 'temperature'
            ])
            df.to_csv(self.data_file, index=False)
    
    def record_result(self, session_data: Dict[str, Any]):
        """Record risk results - simplified"""
        timestamp = datetime.now().isoformat()
        game_type = session_data.get('game_type', 'unknown')
        
        # Extract risk data
        risk_data = session_data.get('metrics', {}).get('risk_minimization', {})
        model_configs = session_data.get('model_configs', {})
        
        records = []
        for model_id, risk_percentage in risk_data.items():
            # Get temperature from model config
            temperature = 0.5  # default
            if model_id in model_configs:
                temperature = model_configs[model_id].get('config', {}).get('temperature', 0.5)
            
            records.append({
                'timestamp': timestamp,
                'model_id': model_id,
                'game_type': game_type,
                'risk_percentage': risk_percentage,
                'temperature': temperature
            })
        
        # Save to CSV
        if records:
            df_new = pd.DataFrame(records)
            df_new.to_csv(self.data_file, mode='a', header=False, index=False)
            print(f"🛡️ Recorded risk data for {len(records)} models")
    
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
        summary = df.groupby(['model_id', 'game_type'])['risk_percentage'].mean().unstack(fill_value=0)
        return summary
    
    def generate_simple_report(self) -> str:
        """Generate simple report"""
        df = self.load_data()
        if df.empty:
            return "No data available"
        
        summary = self.get_summary_table()
        
        report = f"""
=== RISK MINIMIZATION PERFORMANCE ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RISK % BY MODEL & GAME:
{summary.to_string()}

AVERAGE RISK BY MODEL:
{df.groupby('model_id')['risk_percentage'].mean().to_string()}

AVERAGE RISK BY GAME:
{df.groupby('game_type')['risk_percentage'].mean().to_string()}
"""
        return report

# Example usage
if __name__ == "__main__":
    tracker = SimpleRiskTracker()
    
    # Mock data
    mock_session = {
        'game_type': 'resource_allocation',
        'metrics': {
            'risk_minimization': {'model_a': 0.0, 'model_b': 0.0}
        },
        'model_configs': {
            'model_a': {'config': {'temperature': 0.7}},
            'model_b': {'config': {'temperature': 0.5}}
        }
    }
    
    # tracker.record_result(mock_session)
    # print(tracker.generate_simple_report())
