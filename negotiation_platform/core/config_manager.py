import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Manages platform configuration from YAML files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._configs = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all configuration files."""
        config_files = [
            "model_configs.yaml",
            "game_configs.yaml", 
            "platform_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                self._configs[config_file[:-5]] = self.load_config(config_path)
            else:
                self._configs[config_file[:-5]] = self._get_default_config(config_file[:-5])
    
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            print(f"Error loading config {config_path}: {e}")
            return {}
    
    def get_config(self, config_type: str) -> Dict[str, Any]:
        """Get configuration by type."""
        return self._configs.get(config_type, {})
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get specific model configuration."""
        models = self.get_config("model_configs")
        return models.get(model_name)
    
    def get_game_config(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Get specific game configuration."""
        games = self.get_config("game_configs")
        return games.get(game_name)
    
    def _get_default_config(self, config_type: str) -> Dict[str, Any]:
        """Get default configuration if file doesn't exist."""
        defaults = {
            "model_configs": {
                "model_a": {
                    "type": "huggingface",
                    "model_name": "microsoft/DialoGPT-medium",
                    "config": {"device": "auto", "temperature": 0.7}
                },
                "model_b": {
                    "type": "huggingface", 
                    "model_name": "facebook/blenderbot-400M-distill",
                    "config": {"device": "auto", "temperature": 0.8}
                },
                "model_c": {
                    "type": "huggingface",
                    "model_name": "microsoft/DialoGPT-small",
                    "config": {"device": "auto", "temperature": 0.6}
                }
            },
            "game_configs": {
                "company_car": {
                    "starting_price": 45000,
                    "buyer_budget": 40000,
                    "seller_cost": 38000,
                    "buyer_batna": 41000,
                    "seller_batna": 39000,
                    "rounds": 5,
                    "batna_decay": {"buyer": 0.02, "seller": 0.01}
                },
                "resource_allocation": {
                    "total_resources": 100,
                    "constraints": {"gpu_bandwidth": 240, "min_gpu": 20, "min_bandwidth": 15},
                    "batnas": {"development": 300, "marketing": 360},
                    "batna_decay": {"development": 0.03, "marketing": 0.02}
                },
                "integrative_negotiations": {
                    "issues": ["server_room", "meeting_access", "cleaning", "branding"],
                    "rounds": 5,
                    "batna_decay": 0.02
                }
            },
            "platform_config": {
                "logging_level": "INFO",
                "max_retries": 3,
                "timeout_seconds": 30,
                "results_dir": "results"
            }
        }
        return defaults.get(config_type, {})
