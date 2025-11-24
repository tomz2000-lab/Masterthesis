"""
Configuration Manager
====================

The ConfigManager provides centralized configuration management for the entire
negotiation platform. It handles loading, parsing, and accessing YAML configuration
files for models, games, and platform settings.

Key Features:
    - Centralized YAML configuration loading and parsing
    - Type-safe configuration access with fallback defaults
    - Hierarchical configuration organization (models, games, platform)
    - Automatic configuration file discovery and loading
    - Error-tolerant loading with graceful fallbacks

Configuration Structure:
    - model_configs.yaml: AI model definitions and parameters
    - game_configs.yaml: Game-specific settings and rules
    - platform_config.yaml: Global platform settings and defaults

Usage Pattern:
    The ConfigManager follows a lazy-loading pattern where all configuration
    files are loaded during initialization, then cached for efficient access
    throughout the application lifecycle.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """
    Centralized configuration management system for the negotiation platform.
    
    The ConfigManager handles all aspects of configuration loading, parsing, and access
    for the negotiation platform. It provides a unified interface for accessing model
    configurations, game settings, and platform parameters from YAML files.
    
    Architecture:
        Uses a hierarchical configuration system with separate files for different
        concerns (models, games, platform). All configurations are loaded at
        initialization and cached for efficient access.
    
    Key Features:
        - Automatic discovery and loading of YAML configuration files
        - Type-safe configuration access with proper error handling
        - Centralized configuration validation and defaults
        - Efficient caching of parsed configurations
        - Graceful handling of missing or invalid configuration files
        - Hierarchical organization supporting different configuration scopes
    
    Attributes:
        config_dir (Path): Directory path containing configuration files.
        _configs (Dict[str, Dict[str, Any]]): Cache of loaded configurations
            organized by configuration type (model_configs, game_configs, etc.).
    
    Configuration Files:
        - model_configs.yaml: AI model definitions, API keys, and parameters
        - game_configs.yaml: Game-specific rules, defaults, and constraints
        - platform_config.yaml: Global platform settings and behaviors
    
    Example:
        >>> config_manager = ConfigManager()
        >>> model_config = config_manager.get_model_config("model_a")
        >>> print(model_config['model_name'])
        'meta-llama/Llama-2-7b-chat-hf'
        >>> game_config = config_manager.get_game_config("company_car")
        >>> print(game_config['max_rounds'])
        5
    
    Raises:
        FileNotFoundError: If configuration directory doesn't exist.
        yaml.YAMLError: If configuration files contain invalid YAML syntax.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the ConfigManager with automatic configuration loading.
        
        Creates a new ConfigManager instance and immediately loads all available
        configuration files from the specified directory. The directory path is
        resolved relative to the negotiation_platform package root.
        
        Args:
            config_dir (str, optional): Name of the configuration directory relative
                to the negotiation_platform root. Defaults to "configs".
        
        Configuration Loading Order:
            1. model_configs.yaml - AI model configurations
            2. game_configs.yaml - Game-specific settings  
            3. platform_config.yaml - Global platform settings
        
        Example:
            >>> # Use default config directory
            >>> config_manager = ConfigManager()
            >>> # Use custom config directory
            >>> config_manager = ConfigManager("custom_configs")
        
        Note:
            Missing configuration files are handled gracefully - the manager will
            continue loading other files and provide empty dictionaries for missing
            configurations rather than failing completely.
        """
        script_dir = Path(__file__).parent.parent
        self.config_dir = script_dir / config_dir
        self._configs = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """
        Load all platform configuration files into memory for efficient access.
        
        Discovers and loads all recognized configuration files from the config directory.
        Each file is parsed as YAML and stored in the internal configuration cache
        with the filename (minus extension) as the key.
        
        Configuration Files Loaded:
            - model_configs.yaml: AI model definitions and API configurations
            - game_configs.yaml: Game-specific rules and default parameters
            - platform_config.yaml: Global platform settings and behaviors
        
        Error Handling:
            Missing files are skipped gracefully without throwing exceptions.
            Invalid YAML files will log errors but won't prevent other files
            from loading successfully.
        
        Example:
            >>> config_manager = ConfigManager()
            >>> config_manager.load_all_configs()  # Called automatically in __init__
            >>> print(list(config_manager._configs.keys()))
            ['model_configs', 'game_configs', 'platform_config']
        
        Note:
            This method is called automatically during initialization and typically
            doesn't need to be called directly unless reloading configurations.
        """
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
        """
        Retrieve configuration for a specific AI model by name.
        
        Looks up the configuration for the specified model from the loaded
        model_configs.yaml file. Returns the complete configuration dictionary
        for that model including API keys, model parameters, and loading settings.
        
        Args:
            model_name (str): The registered name/identifier of the model to retrieve.
                Must match a key in the model_configs.yaml file.
        
        Returns:
            Optional[Dict[str, Any]]: The model's configuration dictionary if found,
                None if the model name doesn't exist in the configuration.
        
        Configuration Contents:
            Typical model configuration includes:
                - model_name: HuggingFace model identifier
                - type: Model wrapper type (usually 'huggingface')
                - device: GPU/CPU device specification
                - generation_config: Model generation parameters
                - api_key: Authentication token (using environment variables)
        
        Example:
            >>> config_manager = ConfigManager()
            >>> config = config_manager.get_model_config("model_a")
            >>> print(config['model_name'])
            'meta-llama/Llama-2-7b-chat-hf'
            >>> print(config['device'])
            'cuda:0'
        
        Returns:
            None: If model_name is not found in the configuration.
        """
        models = self.get_config("model_configs")
        return models.get(model_name)
    
    def get_game_config(self, game_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve configuration for a specific negotiation game by name.
        
        Looks up the configuration for the specified game from the loaded
        game_configs.yaml file. Returns the complete configuration dictionary
        containing game-specific rules, parameters, and default settings.
        
        Args:
            game_name (str): The registered name/identifier of the game to retrieve.
                Must match a key in the game_configs.yaml file. Common values include
                "company_car", "resource_allocation", "integrative_negotiations".
        
        Returns:
            Optional[Dict[str, Any]]: The game's configuration dictionary if found,
                None if the game name doesn't exist in the configuration.
        
        Configuration Contents:
            Typical game configuration includes:
                - max_rounds: Maximum number of negotiation rounds
                - batna_values: Best Alternative to Negotiated Agreement values
                - resource_limits: Available resources for allocation games
                - scoring_weights: Point values for different outcomes
                - time_limits: Round or session time constraints
        
        Example:
            >>> config_manager = ConfigManager()
            >>> config = config_manager.get_game_config("company_car")
            >>> print(config['max_rounds'])
            5
            >>> print(config['buyer_batna'])
            41000
        
        Returns:
            None: If game_name is not found in the configuration.
        """
        games = self.get_config("game_configs")
        return games.get(game_name)
    
