"""
Base model interface for plug-and-play LLM integration
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseLLMModel(ABC):
    """Abstract base class for all LLM models in the platform"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.is_loaded = False

    @abstractmethod
    def load_model(self):
        """Load the model into memory"""
        pass

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the model"""
        pass

    @abstractmethod
    def unload_model(self):
        """Unload model from memory"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "loaded": self.is_loaded,
            "config": self.config
        }