import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Manages platform configuration from YAML files."""
    
    def __init__(self, config_dir: str = "configs"):
        script_dir = Path(__file__).parent.parent
        self.config_dir = script_dir / config_dir
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
            #else:
                #self._configs[config_file[:-5]] = self._get_default_config(config_file[:-5])
    
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
    
