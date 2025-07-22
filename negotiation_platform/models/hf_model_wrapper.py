"""
Hugging Face model wrapper for plug-and-play integration
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base_model import BaseLLMModel
from typing import Dict, Any

class HuggingFaceModelWrapper(BaseLLMModel):
    """Wrapper for Hugging Face models with plug-and-play functionality"""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.tokenizer = None
        self.model = None
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        """Load Hugging Face model and tokenizer"""
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.config.get('trust_remote_code', True)
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
                trust_remote_code=self.config.get('trust_remote_code', True)
            )

            # Add padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.is_loaded = True
            print(f"✅ {self.model_name} loaded successfully")

        except Exception as e:
            print(f"❌ Error loading {self.model_name}: {str(e)}")
            raise

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using the loaded model"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")

        # Default generation parameters
        generation_params = {
            'max_new_tokens': kwargs.get('max_new_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'do_sample': kwargs.get('do_sample', True),
            'pad_token_id': self.tokenizer.eos_token_id
        }

        try:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            if self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_params
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error: Could not generate response from {self.model_name}"

    def unload_model(self):
        """Unload model from memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.is_loaded = False
        print(f"🗑️  {self.model_name} unloaded")