"""
Hugging Face model wrapper for plug-and-play integration
"""
#from dotenv import load_dotenv
#load_dotenv()
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
            # Get the token from config or from environment variable
            hf_token = self.config.get('api_token') or os.getenv("HUGGINGFACE_TOKEN")

            # Configure quantization for faster loading
            quantization_config = None
            if self.config.get('load_in_8bit', False):
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                print("🔧 Using 8-bit quantization for faster loading")
            elif self.config.get('load_in_4bit', False):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                print("🔧 Using 4-bit quantization for ultra-fast loading")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token,  # Updated from use_auth_token
                trust_remote_code=self.config.get('trust_remote_code', True)
            )

            model_kwargs = {
                'token': hf_token,  # Updated from use_auth_token
                'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
                'device_map': 'auto' if self.device == 'cuda' else None,
                'trust_remote_code': self.config.get('trust_remote_code', True),
                'low_cpu_mem_usage': True,  # Reduces memory usage during loading
            }
            
            # Add quantization if configured
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
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

        # Helper: read only from self.config (top-level) or nested 'generation' dict
        gen_conf = self.config.get('generation', {}) if isinstance(self.config.get('generation', {}), dict) else {}
        def _cfg(key, default):
            if key in self.config:
                return self.config[key]
            if key in gen_conf:
                return gen_conf[key]
            return default

        # Build generation parameters strictly from YAML-configurable values (kwargs will NOT override these)
        generation_params = {
            'max_new_tokens': _cfg('max_new_tokens', 120),
            'temperature': _cfg('temperature', 0.3),
            'top_p': _cfg('top_p', 0.8),
            'do_sample': _cfg('do_sample', True),
            'pad_token_id': self.tokenizer.eos_token_id,
            'repetition_penalty': _cfg('repetition_penalty', 1.1),
            'num_beams': _cfg('num_beams', 1),
        }

        # Remove parameters that may not be supported by all models
        if 'early_stopping' in kwargs:
            generation_params['early_stopping'] = kwargs['early_stopping']
        if 'length_penalty' in kwargs:
            generation_params['length_penalty'] = kwargs['length_penalty']

        try:
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors='pt')
            
            # Get the actual device the model is on
            model_device = next(self.model.parameters()).device
            print(f"🔧 [DEBUG] Model device: {model_device}")
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            print(f"🔧 [DEBUG] Moved inputs to device: {model_device}")
            
            # Ensure pad_token_id is an integer, not a tensor
            if isinstance(generation_params['pad_token_id'], torch.Tensor):
                generation_params['pad_token_id'] = generation_params['pad_token_id'].item()

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    **generation_params
                )

            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Clean up the response but preserve JSON
            response = response.strip()
            
            # First, try to extract JSON if it exists at the beginning
            import re
            json_match = re.match(r'^(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response)
            if json_match:
                # Return just the JSON part if found at the start
                json_part = json_match.group(1)
                print(f"🧹 [DEBUG] Extracted JSON from start: {repr(json_part)}")
                return json_part
            
            # Remove common prefixes that models might add
            prefixes_to_remove = [
                "Here's my response:",
                "My action:",
                "Response:",
                "JSON:",
                "Answer:",
                "\n\n",
                "```json",
                "```"
            ]
            
            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
                    # Check again for JSON after removing prefix
                    json_match = re.match(r'^(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response)
                    if json_match:
                        json_part = json_match.group(1)
                        print(f"🧹 [DEBUG] Extracted JSON after prefix removal: {repr(json_part)}")
                        return json_part
            
            # If response ends with closing markdown, remove it
            if response.endswith("```"):
                response = response[:-3].strip()
            
            print(f"🧹 [DEBUG] Cleaned response: {repr(response)}")
            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error: Could not generate response from {self.model_name}"

    def parse_action(self, response: str, game_type: str = None) -> Dict[str, Any]:
        """Parse LLM response into a structured action with Pydantic validation"""
        import json
        import re
        
        print(f"🔍 [DEBUG] Raw LLM response: {repr(response)}")
        
        try:
            # Clean up the response
            response_clean = response.strip()
            
            # First priority: Look for JSON at the start of the response
            json_at_start = re.match(r'^(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response_clean)
            if json_at_start:
                try:
                    parsed = json.loads(json_at_start.group(1))
                    print(f"✅ [DEBUG] Successfully parsed JSON from start: {parsed}")
                    
                    # Apply Pydantic validation if game_type is provided
                    if game_type:
                        from .action_schemas import validate_and_constrain_action
                        try:
                            validated = validate_and_constrain_action(json_at_start.group(1), game_type)
                            print(f"✅ [DEBUG] Pydantic validation successful: {validated}")
                            return validated
                        except ValueError as e:
                            print(f"⚠️ [DEBUG] Pydantic validation failed, using original: {e}")
                    
                    return parsed
                except json.JSONDecodeError as e:
                    print(f"⚠️ [DEBUG] JSON decode error for start match: {e}")
            
            # Try to find JSON in the response using various patterns
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON pattern
                r'\{[^{}]*\}',                        # Simple JSON pattern
                r'```json\s*(.*?)\s*```',             # JSON in code blocks
                r'```\s*(.*?)\s*```'                  # Any code block
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response_clean, re.DOTALL)
                if matches:
                    for match in matches:
                        try:
                            # Clean the match
                            match_clean = match.strip()
                            parsed = json.loads(match_clean)
                            print(f"✅ [DEBUG] Successfully parsed JSON: {parsed}")
                            
                            # Apply Pydantic validation if game_type is provided
                            if game_type:
                                from .action_schemas import validate_and_constrain_action
                                try:
                                    validated = validate_and_constrain_action(match_clean, game_type)
                                    print(f"✅ [DEBUG] Pydantic validation successful: {validated}")
                                    return validated
                                except ValueError as e:
                                    print(f"⚠️ [DEBUG] Pydantic validation failed, using original: {e}")
                            
                            return parsed
                        except json.JSONDecodeError as e:
                            print(f"⚠️ [DEBUG] JSON decode error for '{match[:50]}...': {e}")
                            continue
            
            # Try to parse the entire response as JSON
            if response_clean.startswith('{') and ('}' in response_clean):
                # Find the first complete JSON object
                brace_count = 0
                end_idx = 0
                for i, char in enumerate(response_clean):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > 0:
                    json_candidate = response_clean[:end_idx]
                    try:
                        parsed = json.loads(json_candidate)
                        print(f"✅ [DEBUG] Successfully parsed extracted JSON: {parsed}")
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"⚠️ [DEBUG] Extracted JSON decode error: {e}")
            
            # Try to extract key-value pairs manually if JSON parsing fails
            action_dict = self._extract_action_manually(response_clean)
            if action_dict:
                print(f"✅ [DEBUG] Manually extracted action: {action_dict}")
                return action_dict
            
            # Fallback: return a no-op action
            print(f"❌ [DEBUG] Could not parse action from response: {response}")
            return {"type": "noop", "reason": "failed_to_parse"}
            
        except Exception as e:
            print(f"❌ [DEBUG] Error parsing action: {str(e)}")
            return {"type": "noop", "reason": "parse_error"}
    
    def _extract_action_manually(self, response: str) -> Dict[str, Any]:
        """Try to extract action manually from response text"""
        import re
        
        # Look for common patterns
        action_dict = {}
        
        # Look for type/action
        type_match = re.search(r'(?:type|action)[\s:]*["\']?(\w+)["\']?', response, re.IGNORECASE)
        if type_match:
            action_dict["type"] = type_match.group(1).lower()
        
        # Look for price
        price_match = re.search(r'price[\s:]*["\']?(\d+(?:\.\d+)?)["\']?', response, re.IGNORECASE)
        if price_match:
            action_dict["price"] = float(price_match.group(1))
        
        # Look for offer
        if "offer" in response.lower():
            action_dict["type"] = "offer"
        elif "accept" in response.lower():
            action_dict["type"] = "accept"
        elif "reject" in response.lower():
            action_dict["type"] = "reject"
        
        return action_dict if action_dict else None

    def unload_model(self):
        """Unload model from memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.is_loaded = False
        print(f"🗑️  {self.model_name} unloaded")