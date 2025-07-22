"""
LLM Manager for handling multiple models with plug-and-play functionality
"""
import yaml
from typing import Dict, List, Any, Optional
from negotiation_platform.models.base_model import BaseLLMModel
from negotiation_platform.models.hf_model_wrapper import HuggingFaceModelWrapper

class LLMManager:
    """Manages multiple LLM models with plug-and-play switching"""

    def __init__(self, config_path: str = 'configs/model_configs.yaml'):
        self.models: Dict[str, BaseLLMModel] = {}
        self.active_model: Optional[str] = None
        self.config_path = config_path
        self._load_model_configs()

    def _load_model_configs(self):
        """Load model configurations from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.model_configs = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found")
            self.model_configs = {"models": {}}

    def register_model(self, model_id: str, model_config: Dict[str, Any]):
        """Register a new model for plug-and-play usage"""
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
        """Load a specific model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")

        self.models[model_id].load_model()

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

    def get_response(self, prompt: str, **kwargs) -> str:
        """Get response from the currently active model"""
        if not self.active_model:
            raise RuntimeError("No active model selected")

        return self.models[self.active_model].generate_response(prompt, **kwargs)

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