"""
LLM Manager for handling multiple models with plug-and-play functionality
"""
import yaml
import threading
from typing import Dict, List, Any, Optional
from negotiation_platform.models.base_model import BaseLLMModel
from negotiation_platform.models.hf_model_wrapper import HuggingFaceModelWrapper

class LLMManager:
    """Manages multiple LLM models with plug-and-play switching"""

    def __init__(self, model_configs: dict):
        print(f"[DEBUG] LLMManager received model_configs: {model_configs}")
        print(f"[DEBUG] Type of model_configs: {type(model_configs)}")
        print(f"[DEBUG] Keys in model_configs: {list(model_configs.keys()) if model_configs else 'None or empty'}")
        # model_id -> wrapper (may be shared across ids)
        self.models: Dict[str, BaseLLMModel] = {}
        # model_name -> shared wrapper instance
        self.shared_models: Dict[str, BaseLLMModel] = {}
        # model_id -> model_name (alias mapping)
        self.model_aliases: Dict[str, str] = {}
        self.active_model: Optional[str] = None
        self.model_configs = model_configs
        self.max_loaded_models = 2  # Keep max N unique models in memory at once
        # Track loading order by model_name (unique) for LRU eviction
        self.loaded_order: List[str] = []
        # Manager-level lock to serialize load/unload operations and avoid races
        self.manager_lock = threading.Lock()

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

        if model_type != 'huggingface':
            raise ValueError(f"Unsupported model type: {model_type}")

        model_name = model_config['model_name']

        # If we've already created a wrapper for this model_name, reuse it
        if model_name in self.shared_models:
            wrapper = self.shared_models[model_name]
            self.models[model_id] = wrapper
            self.model_aliases[model_id] = model_name
            print(f"📝 Registered model alias: {model_id} -> {model_name}")
            return

        # Otherwise create a new wrapper and register it as shared
        wrapper = HuggingFaceModelWrapper(
            model_name=model_name,
            config=model_config.get('config', {})
        )
        self.models[model_id] = wrapper
        self.shared_models[model_name] = wrapper
        self.model_aliases[model_id] = model_name
        print(f"📝 Registered model: {model_id} (new instance for {model_name})")

    def load_model(self, model_id: str):
        """Load a specific model with smart memory management"""
        # Serialize load/unload operations to avoid concurrent from_pretrained calls
        with self.manager_lock:
            # Inner function does the actual work while under lock
            def _do_load():
                if model_id not in self.models:
                    raise ValueError(f"Model {model_id} not registered")

                # Map model_id -> model_name (shared key)
                model_name = self.model_aliases.get(model_id)
                if not model_name:
                    # fallback: try to get model_name from the wrapper if possible
                    model_name = getattr(self.models[model_id], 'model_name', None)

                wrapper = self.models[model_id]

                # If already loaded, just update order and return the wrapper
                if wrapper.is_loaded:
                    if model_name in self.loaded_order:
                        self.loaded_order.remove(model_name)
                    self.loaded_order.append(model_name)
                    return wrapper

                # Check if we need to unload old models to free memory
                # Count unique loaded wrappers (by model_name)
                loaded_count = sum(1 for wrapper in self.shared_models.values() if wrapper.is_loaded)

                if loaded_count >= self.max_loaded_models:
                    # Unload the least recently used shared model by model_name
                    lru_model_name = self.loaded_order.pop(0)
                    print(f"🗑️  Unloading LRU model: {lru_model_name} to free memory")
                    shared_wrapper = self.shared_models.get(lru_model_name)
                    if shared_wrapper and shared_wrapper.is_loaded:
                        shared_wrapper.unload_model()

                # Load the requested model
                print(f"🚀 Loading model: {model_id} (shared name: {model_name})")
                wrapper.load_model()
                # ensure model_name is in loaded order (unique)
                if model_name in self.loaded_order:
                    self.loaded_order.remove(model_name)
                self.loaded_order.append(model_name)

                # Return the loaded wrapper
                return wrapper

            # Execute the protected load and return its result
            return _do_load()

    def switch_model(self, model_id: str):
        """Switch to a different model (plug-and-play functionality)"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")

        # Do not eagerly unload on switch; rely on LRU eviction instead.
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
        
        model_name = self.model_aliases.get(model_id)
        wrapper = self.models[model_id]

        # If other aliases reference the same shared model, warn and skip
        aliases_using = [mid for mid, mname in self.model_aliases.items() if mname == model_name and mid != model_id]
        if aliases_using:
            print(f"⚠️  Model {model_id} shares wrapper '{model_name}' with aliases {aliases_using}; skipping unload to avoid breaking aliases")
            return

        if wrapper.is_loaded:
            wrapper.unload_model()
            # Remove from loaded order if present
            if model_name in self.loaded_order:
                self.loaded_order.remove(model_name)
            print(f"🗑️  Unloaded model: {model_id} (shared name: {model_name})")
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
        for model_name, wrapper in self.shared_models.items():
            if wrapper.is_loaded:
                wrapper.unload_model()
        print("🧹 All shared models unloaded")