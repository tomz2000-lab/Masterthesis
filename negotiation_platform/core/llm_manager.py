"""
LLM Manager for handling multiple models with plug-and-play functionality
"""
import yaml
from typing import Dict, List, Any, Optional
from negotiation_platform.models.base_model import BaseLLMModel
from negotiation_platform.models.hf_model_wrapper import HuggingFaceModelWrapper

class LLMManager:
    """Manages multiple LLM models with plug-and-play switching"""

    def __init__(self, model_configs: dict):
        print(f"[DEBUG] LLMManager received model_configs: {model_configs}")
        print(f"[DEBUG] Type of model_configs: {type(model_configs)}")
        print(f"[DEBUG] Keys in model_configs: {list(model_configs.keys()) if model_configs else 'None or empty'}")
        self.models: Dict[str, BaseLLMModel] = {}
        self.active_model: Optional[str] = None
        self.model_configs = model_configs
        self.max_loaded_models = 2  # Keep max 2 models in memory at once
        self.loaded_order = []  # Track loading order for LRU eviction

        # Register all models from config on init (but don't load them yet)
        self._register_all_models()

    def _register_all_models(self):
        if not self.model_configs:
            print("[ERROR] No model configs provided to LLMManager - registration skipped")
            return

        print(f"[DEBUG] Attempting to register {len(self.model_configs)} models...")
        # Your config might be a dict with keys == model IDs, values == config dict
        for model_id, model_config in self.model_configs.items():
            print(f"[DEBUG] Processing model: {model_id}")
            print(f"[DEBUG] Model config: {model_config}")
            try:
                self.register_model(model_id, model_config)
            except Exception as e:
                print(f"[ERROR] Failed to register model {model_id}: {e}")

    def _load_model_configs(self):
        """Load model configurations from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.model_configs = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found")
            self.model_configs = {"models": {}}

    def register_model(self, model_id: str, model_config: Dict[str, Any]):
        model_type = model_config.get('type', 'huggingface')

        if model_type == 'huggingface':
            model = HuggingFaceModelWrapper(
                model_name=model_config['model_name'],
                config=model_config.get('config', {})
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.models[model_id] = model
        print(f"📝 Registered model: {model_id}")

    def load_model(self, model_id: str):
        """Load a specific model with smart memory management"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")

        # If already loaded, just update order and return the model
        if self.models[model_id].is_loaded:
            if model_id in self.loaded_order:
                self.loaded_order.remove(model_id)
            self.loaded_order.append(model_id)
            return self.models[model_id]

        # Check if we need to unload old models to free memory
        loaded_count = sum(1 for model in self.models.values() if model.is_loaded)
        
        if loaded_count >= self.max_loaded_models:
            # Unload the least recently used model
            lru_model_id = self.loaded_order.pop(0)
            print(f"🗑️  Unloading LRU model: {lru_model_id} to free memory")
            self.models[lru_model_id].unload_model()

        # Load the requested model
        print(f"🚀 Loading model: {model_id}")
        self.models[model_id].load_model()
        self.loaded_order.append(model_id)
        
        # Return the loaded model
        return self.models[model_id]

    def switch_model(self, model_id: str):
        """Switch to a different model (plug-and-play functionality)"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")

        # Unload current model if exists
        if self.active_model and self.models[self.active_model].is_loaded:
            self.models[self.active_model].unload_model()

        # Load new model if not loaded
        if not self.models[model_id].is_loaded:
            self.load_model(model_id)

        self.active_model = model_id
        print(f"🔄 Switched to model: {model_id}")

    def get_response(self, prompt: str, model_id: str = None, **kwargs) -> str:
        """Get response from a specific model or the currently active model"""
        target_model = model_id or self.active_model
        
        if not target_model:
            raise RuntimeError("No model specified and no active model selected")
        
        if target_model not in self.models:
            raise ValueError(f"Model {target_model} not registered")
        
        # Lazy loading: only load the model when actually needed
        if not self.models[target_model].is_loaded:
            print(f"🔄 Lazy loading model: {target_model}")
            self.load_model(target_model)
            self.active_model = target_model

        return self.models[target_model].generate_response(prompt, **kwargs)

    def unload_model(self, model_id: str):
        """Unload a specific model from memory"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
        
        if self.models[model_id].is_loaded:
            self.models[model_id].unload_model()
            # Remove from loaded order if present
            if model_id in self.loaded_order:
                self.loaded_order.remove(model_id)
            print(f"🗑️  Unloaded model: {model_id}")
        else:
            print(f"⚠️  Model {model_id} was not loaded")

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models and their status"""
        return {
            model_id: model.get_model_info()
            for model_id, model in self.models.items()
        }

    def cleanup(self):
        """Unload all models and cleanup resources"""
        for model in self.models.values():
            if model.is_loaded:
                model.unload_model()
        print("🧹 All models unloaded")