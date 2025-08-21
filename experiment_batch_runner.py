#!/usr/bin/env python3
"""
Comprehensive Batch Runner for Negotiation Experiments
=====================================================

This script runs systematic experiments across:
- 3 Game Types: resource_allocation, company_car, integrative_negotiations  
- 3 Model Configurations: model_a (temp=0.7), model_b (temp=0.5), model_c (temp=0.3)
- 25 Sessions per combination = 225 total sessions

Features:
- Progress tracking and resumption
- Error handling and retry logic
- Performance monitoring
- Results aggregation
- HPC cluster optimization
"""

import os
import sys
import json
import time
import argparse
import itertools
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Add parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import platform components
from negotiation_platform.core.llm_manager import LLMManager
from negotiation_platform.core.game_engine import GameEngine
from negotiation_platform.core.metrics_calculator import MetricsCalculator
from negotiation_platform.core.session_manager import SessionManager
from negotiation_platform.core.config_manager import ConfigManager
from results.master_tracker import SimpleMasterTracker

def run_negotiation_session(game_type: str, primary_model: str, session_id: str, 
                           rounds: int = 5, enable_metrics: bool = True) -> Dict[str, Any]:
    """Wrapper function to run a single negotiation session"""
    
    # Initialize configuration
    config_manager = ConfigManager()
    
    # Initialize components
    llm_manager = LLMManager(config_manager.get_config("model_configs"))
    game_engine = GameEngine()
    metrics_calculator = MetricsCalculator()
    session_manager = SessionManager(llm_manager, game_engine, metrics_calculator)
    
    # Determine which models to use
    available_models = ['model_a', 'model_b', 'model_c']
    if primary_model not in available_models:
        primary_model = 'model_a'
    
    # Select two different models for the negotiation
    other_models = [m for m in available_models if m != primary_model]
    secondary_model = other_models[0] if other_models else 'model_b'
    
    players = [primary_model, secondary_model]
    
    # Get game configuration
    game_config = config_manager.get_game_config(game_type)
    
    # Run the negotiation
    result = session_manager.run_negotiation(
        players=players,
        game_type=game_type,
        game_config=game_config,
        num_rounds=rounds
    )
    
    # Add session metadata
    if result:
        result['session_metadata'] = {
            'session_id': session_id,
            'primary_model': primary_model,
            'secondary_model': secondary_model,
            'players': players
        }
        result['model_configs'] = config_manager.get_config("model_configs")
    
    return result

class ExperimentBatchRunner:
    """Manages large-scale negotiation experiments with progress tracking"""
    
    def __init__(self, results_dir: str = "results", checkpoint_interval: int = 5):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Experimental parameters
        self.models = ['model_a', 'model_b', 'model_c']
        self.games = ['resource_allocation', 'company_car', 'integrative_negotiations']
        self.sessions_per_combo = 25
        self.checkpoint_interval = checkpoint_interval
        
        # Progress tracking
        self.progress_file = self.results_dir / "experiment_progress.json"
        self.completed_sessions = self.load_progress()
        
        # Results tracking
        self.tracker = SimpleMasterTracker(str(self.results_dir))
        
        # Performance monitoring
        self.timing_data = []
        self.error_log = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the experiment"""
        log_file = self.results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_progress(self) -> Dict[str, Dict[str, int]]:
        """Load previous progress from checkpoint file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        
        # Initialize progress tracking
        progress = {}
        for game in self.games:
            progress[game] = {}
            for model in self.models:
                progress[game][model] = 0
        return progress
    
    def save_progress(self):
        """Save current progress to checkpoint file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.completed_sessions, f, indent=2)
    
    def get_experiment_plan(self) -> List[Tuple[str, str, int]]:
        """Generate complete experiment plan with remaining sessions"""
        plan = []
        
        for game, model in itertools.product(self.games, self.models):
            completed = self.completed_sessions[game][model]
            remaining = self.sessions_per_combo - completed
            
            for session_num in range(completed, self.sessions_per_combo):
                plan.append((game, model, session_num))
        
        return plan
    
    def estimate_total_time(self, avg_session_time: float = 3.0) -> Dict[str, Any]:
        """Estimate experiment completion time"""
        plan = self.get_experiment_plan()
        total_sessions = len(plan)
        
        # Time estimates (minutes per session)
        estimates = {
            'conservative': avg_session_time * 1.5,  # Account for overhead
            'optimistic': avg_session_time * 0.8,    # Everything goes smoothly
            'realistic': avg_session_time * 1.2      # Some overhead, minor issues
        }
        
        results = {
            'total_sessions': total_sessions,
            'sessions_per_combo': self.sessions_per_combo,
            'combinations': len(self.games) * len(self.models),
        }
        
        for scenario, time_per_session in estimates.items():
            total_minutes = total_sessions * time_per_session
            results[f'{scenario}_total_hours'] = total_minutes / 60
            results[f'{scenario}_total_days'] = total_minutes / (60 * 24)
        
        return results
    
    def run_single_session(self, game_type: str, model_id: str, session_num: int) -> Tuple[bool, float, Any]:
        """Run a single negotiation session with error handling"""
        session_id = f"{game_type}_{model_id}_{session_num:03d}"
        start_time = time.time()
        
        try:
            self.logger.info(f"🎯 Starting session {session_id}")
            
            # Configure the session
            session_config = {
                'game_type': game_type,
                'primary_model': model_id,
                'session_id': session_id,
                'rounds': 5,
                'enable_metrics': True
            }
            
            # Run the negotiation
            result = run_negotiation_session(**session_config)
            
            # Record timing
            duration = time.time() - start_time
            self.timing_data.append({
                'session_id': session_id,
                'game_type': game_type,
                'model_id': model_id,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            })
            
            # Record results
            if result and 'metrics' in result:
                self.tracker.record_session(result)
                self.logger.info(f"✅ Completed {session_id} in {duration:.1f}s")
                return True, duration, result
            else:
                self.logger.warning(f"⚠️ Session {session_id} completed but no metrics found")
                return False, duration, result
                
        except Exception as e:
            duration = time.time() - start_time
            error_info = {
                'session_id': session_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration
            }
            self.error_log.append(error_info)
            self.logger.error(f"❌ Session {session_id} failed: {e}")
            return False, duration, None
    
    def run_experiment_batch(self, max_sessions: int = None, 
                           parallel_models: bool = False) -> Dict[str, Any]:
        """Run the complete experiment with progress tracking"""
        
        plan = self.get_experiment_plan()
        if max_sessions:
            plan = plan[:max_sessions]
        
        total_sessions = len(plan)
        self.logger.info(f"🚀 Starting experiment: {total_sessions} sessions to complete")
        
        # Time estimation
        time_estimates = self.estimate_total_time()
        self.logger.info(f"⏱️ Estimated completion: {time_estimates['realistic_total_hours']:.1f} hours")
        
        start_time = time.time()
        successful_sessions = 0
        failed_sessions = 0
        
        for i, (game_type, model_id, session_num) in enumerate(plan, 1):
            self.logger.info(f"\n📊 Progress: {i}/{total_sessions} sessions")
            
            # Run the session
            success, duration, result = self.run_single_session(game_type, model_id, session_num)
            
            if success:
                successful_sessions += 1
                # Update progress
                self.completed_sessions[game_type][model_id] += 1
            else:
                failed_sessions += 1
            
            # Checkpoint progress regularly
            if i % self.checkpoint_interval == 0 or i == total_sessions:
                self.save_progress()
                self.logger.info(f"💾 Progress saved: {successful_sessions} successful, {failed_sessions} failed")
                
                # Generate intermediate report
                if i % (self.checkpoint_interval * 3) == 0:
                    self.generate_progress_report()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_session_time = total_time / total_sessions if total_sessions > 0 else 0
        
        final_stats = {
            'total_sessions_attempted': total_sessions,
            'successful_sessions': successful_sessions,
            'failed_sessions': failed_sessions,
            'success_rate': successful_sessions / total_sessions if total_sessions > 0 else 0,
            'total_experiment_time_hours': total_time / 3600,
            'average_session_time_minutes': avg_session_time / 60,
            'estimated_vs_actual': {
                'estimated_hours': time_estimates['realistic_total_hours'],
                'actual_hours': total_time / 3600
            }
        }
        
        self.logger.info(f"🎉 Experiment completed! Success rate: {final_stats['success_rate']:.1%}")
        return final_stats
    
    def generate_progress_report(self):
        """Generate a progress report showing completion status"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📈 EXPERIMENT PROGRESS REPORT")
        self.logger.info("="*60)
        
        for game in self.games:
            self.logger.info(f"\n🎮 {game.upper()}:")
            for model in self.models:
                completed = self.completed_sessions[game][model]
                percentage = (completed / self.sessions_per_combo) * 100
                progress_bar = "█" * (completed // 2) + "░" * ((self.sessions_per_combo - completed) // 2)
                self.logger.info(f"  {model}: {completed:2d}/{self.sessions_per_combo} sessions ({percentage:5.1f}%) [{progress_bar}]")
        
        # Performance statistics
        if self.timing_data:
            avg_time = sum(t['duration_seconds'] for t in self.timing_data) / len(self.timing_data)
            self.logger.info(f"\n⏱️ Average session time: {avg_time:.1f} seconds")
        
        if self.error_log:
            error_rate = len(self.error_log) / (len(self.timing_data) + len(self.error_log))
            self.logger.info(f"❌ Error rate: {error_rate:.1%}")
    
    def generate_final_report(self):
        """Generate comprehensive final analysis"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 FINAL EXPERIMENT ANALYSIS")
        self.logger.info("="*60)
        
        # Load and analyze all results
        report = self.tracker.generate_master_report()
        self.logger.info(report)
        
        # Model comparison
        comparison = self.tracker.get_model_comparison()
        self.logger.info(comparison)

def create_model_size_configs():
    """Create configuration files for different model sizes"""
    
    configs = {
        '8B': {
            'model_name': 'meta-llama/Llama-3.1-8B-Instruct',
            'load_in_8bit': True,
            'estimated_memory_gb': 12,
            'estimated_time_per_session_minutes': 3
        },
        '70B': {
            'model_name': 'meta-llama/Llama-3.1-70B-Instruct', 
            'load_in_8bit': True,
            'estimated_memory_gb': 45,
            'estimated_time_per_session_minutes': 8
        }
    }
    
    return configs

def estimate_cluster_requirements(model_size: str = '8B'):
    """Estimate HPC cluster requirements for different model sizes"""
    
    configs = create_model_size_configs()
    if model_size not in configs:
        raise ValueError(f"Model size {model_size} not supported. Choose from: {list(configs.keys())}")
    
    config = configs[model_size]
    
    # L40 GPU specifications (your cluster)
    l40_memory_gb = 48
    l40_compute_capability = 8.9
    
    # Calculate requirements
    sessions_total = 3 * 3 * 25  # games × models × sessions_per_combo
    time_per_session = config['estimated_time_per_session_minutes']
    memory_required = config['estimated_memory_gb']
    
    # Can the model fit on L40?
    fits_on_l40 = memory_required <= l40_memory_gb * 0.9  # 90% utilization safe
    
    estimates = {
        'model_size': model_size,
        'model_name': config['model_name'],
        'memory_required_gb': memory_required,
        'l40_compatible': fits_on_l40,
        'total_sessions': sessions_total,
        'time_per_session_minutes': time_per_session,
        'total_experiment_hours': (sessions_total * time_per_session) / 60,
        'total_experiment_days': (sessions_total * time_per_session) / (60 * 24),
        'parallel_speedup_estimate': {
            'sequential_hours': (sessions_total * time_per_session) / 60,
            'with_3_parallel_hours': (sessions_total * time_per_session) / (60 * 3),  # 3 games parallel
            'with_9_parallel_hours': (sessions_total * time_per_session) / (60 * 9)   # All combos parallel
        }
    }
    
    return estimates

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Run negotiation experiments')
    parser.add_argument('--mode', choices=['estimate', 'run', 'resume'], default='estimate',
                       help='Mode: estimate timing, run experiment, or resume from checkpoint')
    parser.add_argument('--model-size', choices=['8B', '70B'], default='8B',
                       help='Model size for estimation')
    parser.add_argument('--max-sessions', type=int, default=None,
                       help='Maximum number of sessions to run (for testing)')
    parser.add_argument('--results-dir', default='results',
                       help='Directory to store results')
    
    args = parser.parse_args()
    
    if args.mode == 'estimate':
        print("\n🔍 CLUSTER REQUIREMENTS ESTIMATION")
        print("="*50)
        
        for model_size in ['8B', '70B']:
            estimates = estimate_cluster_requirements(model_size)
            print(f"\n📊 {model_size} Model ({estimates['model_name']}):")
            print(f"  Memory Required: {estimates['memory_required_gb']} GB")
            print(f"  L40 Compatible: {'✅ Yes' if estimates['l40_compatible'] else '❌ No'}")
            print(f"  Time per Session: {estimates['time_per_session_minutes']} minutes")
            print(f"  Total Experiment Time: {estimates['total_experiment_hours']:.1f} hours ({estimates['total_experiment_days']:.1f} days)")
            print(f"  With Parallelization:")
            print(f"    3 Games Parallel: {estimates['parallel_speedup_estimate']['with_3_parallel_hours']:.1f} hours")
            print(f"    9 Combos Parallel: {estimates['parallel_speedup_estimate']['with_9_parallel_hours']:.1f} hours")
    
    elif args.mode in ['run', 'resume']:
        runner = ExperimentBatchRunner(results_dir=args.results_dir)
        
        if args.mode == 'run':
            print("\n🚀 STARTING FULL EXPERIMENT")
        else:
            print("\n🔄 RESUMING EXPERIMENT FROM CHECKPOINT")
        
        # Run the experiment
        stats = runner.run_experiment_batch(max_sessions=args.max_sessions)
        
        # Generate final report
        runner.generate_final_report()
        
        print(f"\n🎉 Experiment completed!")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Total time: {stats['total_experiment_time_hours']:.1f} hours")

if __name__ == "__main__":
    main()
