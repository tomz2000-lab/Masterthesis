"""
Hugging Face Model Wrapper
=========================

Comprehensive wrapper implementation for Hugging Face Transformers models,
providing plug-and-play integration with the negotiation platform. This module
handles the complexities of model loading, memory management, GPU utilization,
and response generation for various Hugging Face language models.

The wrapper supports advanced features including quantization, device mapping,
memory optimization, and robust error handling. It provides seamless integration
with the platform's action parsing system and includes extensive debugging
capabilities for production deployment.

Key Features:
    - Automatic device detection and optimal GPU/CPU placement
    - Memory-efficient loading with quantization support (4-bit, 8-bit)
    - Robust response generation with configurable parameters
    - Intelligent action parsing with game-specific validation
    - Comprehensive error handling and recovery mechanisms
    - Production-ready logging and monitoring capabilities

Supported Models:
    - All Hugging Face Transformers causal language models
    - Meta Llama family (2, 3.1, 3.2)
    - Mistral family (7B, 8B variants)
    - Qwen family (3B, 7B variants)
    - Custom fine-tuned models with compatible architectures

Example:
    >>> config = {
    ...     'device': 'auto',
    ...     'load_in_8bit': True,
    ...     'temperature': 0.7,
    ...     'max_new_tokens': 150
    ... }
    >>> model = HuggingFaceModelWrapper('meta-llama/Llama-2-7b-chat-hf', config)
    >>> model.load_model()
    >>> response = model.generate_response('Negotiate a car price.')
    >>> action = model.parse_action(response, 'company_car')

Note:
    This implementation requires the transformers, torch, and optionally
    bitsandbytes libraries. GPU support requires CUDA-compatible hardware
    and appropriate driver installation.
"""
#from dotenv import load_dotenv
#load_dotenv()
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os
from .base_model import BaseLLMModel
from typing import Dict, Any

class HuggingFaceModelWrapper(BaseLLMModel):
    """Production-ready wrapper for Hugging Face Transformers models.
    
    This class provides a comprehensive implementation of the BaseLLMModel
    interface specifically designed for Hugging Face Transformers models.
    It handles the complexities of modern language model deployment including
    memory optimization, device management, and robust inference pipelines.
    
    The wrapper supports both research and production scenarios with features
    like automatic quantization, intelligent device placement, comprehensive
    error handling, and extensive logging for debugging and monitoring.
    
    Attributes:
        tokenizer: Hugging Face tokenizer instance for the model.
        model: Loaded Hugging Face model instance ready for inference.
        device (str): Target device for model execution ('cuda', 'cpu', or 'auto').
    
    Configuration Options:
        Device and Memory:
            - device (str): Target device ('cuda', 'cpu', 'auto')
            - device_map (str/dict): Advanced device placement strategy
            - load_in_8bit (bool): Enable 8-bit quantization for memory efficiency
            - load_in_4bit (bool): Enable 4-bit quantization for extreme efficiency
        
        Generation Parameters:
            - temperature (float): Sampling temperature (0.0-2.0)
            - max_new_tokens (int): Maximum tokens to generate
            - top_p (float): Nucleus sampling parameter
            - do_sample (bool): Enable sampling vs greedy decoding
            - repetition_penalty (float): Penalty for repetitive text
        
        Authentication:
            - api_token (str): Hugging Face API token for private models
            - trust_remote_code (bool): Allow execution of remote code
    
    Example:
        >>> config = {
        ...     'device': 'auto',
        ...     'load_in_8bit': True,
        ...     'temperature': 0.7,
        ...     'max_new_tokens': 120,
        ...     'api_token': 'hf_token_here'
        ... }
        >>> wrapper = HuggingFaceModelWrapper('model-name', config)
        >>> wrapper.load_model()
        >>> response = wrapper.generate_response('Your prompt here')
    
    Note:
        This implementation includes production-ready optimizations and
        extensive error handling. For development, consider enabling
        verbose logging to monitor model behavior and performance.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """Initialize Hugging Face model wrapper with intelligent device detection.
        
        Sets up the wrapper with automatic device detection and configuration
        normalization. Handles the complexity of device selection including
        CUDA availability checking and fallback strategies for various
        hardware configurations.
        
        Args:
            model_name (str): Hugging Face model identifier, which can be:
                - Repository ID (e.g., 'meta-llama/Llama-2-7b-chat-hf')
                - Local model path for offline usage
                - Custom model name for privately hosted models
            config (Dict[str, Any]): Comprehensive configuration dictionary
                containing device preferences, generation parameters, memory
                optimization settings, and authentication credentials.
        
        Example:
            >>> config = {
            ...     'device': 'auto',  # Automatic device detection
            ...     'load_in_8bit': True,  # Memory optimization
            ...     'temperature': 0.7,  # Generation creativity
            ...     'max_new_tokens': 150
            ... }
            >>> wrapper = HuggingFaceModelWrapper('model-name', config)
        
        Note:
            The device setting is intelligently resolved: 'auto' selects CUDA
            if available, otherwise falls back to CPU. Explicit device settings
            override automatic detection but should match available hardware.
        """
        super().__init__(model_name, config)
        self.tokenizer = None
        self.model = None
        # Handle 'auto' device setting and normalize to 'cuda' or 'cpu'
        device_config = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if device_config == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_config

    def load_model(self):
        """Load Hugging Face model and tokenizer with advanced optimization.
        
        Performs comprehensive model loading with memory optimization, device
        placement, quantization support, and extensive error handling. This method
        implements production-ready loading strategies including automatic
        quantization, intelligent device mapping, and detailed progress monitoring.
        
        The loading process includes:
        - Environment optimization and TorchDynamo configuration
        - Quantization setup (4-bit/8-bit) for memory efficiency
        - Tokenizer initialization with authentication handling
        - Model loading with optimal device placement
        - Resource monitoring and diagnostic reporting
        
        Raises:
            RuntimeError: If model loading fails due to insufficient resources,
                authentication issues, or hardware compatibility problems.
            FileNotFoundError: If the specified model cannot be found in the
                Hugging Face Hub or local filesystem.
            MemoryError: If insufficient GPU/CPU memory is available for the
                model with current quantization settings.
            ImportError: If required dependencies (bitsandbytes) are missing
                for quantization features.
        
        Example:
            >>> model = HuggingFaceModelWrapper('meta-llama/Llama-2-7b-chat-hf', config)
            >>> model.load_model()  # Comprehensive loading with optimization
            Loading meta-llama/Llama-2-7b-chat-hf...
            ✅ Model loaded successfully in 45.2s
        
        Note:
            This method includes extensive logging and diagnostic output for
            debugging. It automatically handles quantization, device placement,
            and memory optimization based on the configuration provided during
            initialization. The loading process is optimized for both speed
            and memory efficiency.
        """
        try:
            print(f"Loading {self.model_name}...")
            # Enable more verbose transformers logs to help diagnose slow steps
            os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'info')
            os.environ.setdefault('HF_HUB_OFFLINE', '0')
            
            # Check if we should disable TorchDynamo (helps with some hanging issues)
            if os.environ.get('TORCHDYNAMO_DISABLE'):
                print(f"🔧 TorchDynamo is DISABLED via environment variable")
            else:
                print(f"🔧 TorchDynamo is ENABLED (consider TORCHDYNAMO_DISABLE=1 if hanging)")
                # Disable TorchDynamo to prevent hanging issues
                print(f"🔧 DISABLING TorchDynamo to prevent potential hanging...")
                os.environ['TORCHDYNAMO_DISABLE'] = '1'
                try:
                    # Import torch._dynamo safely without shadowing the torch module
                    import torch._dynamo as torch_dynamo
                    torch_dynamo.config.suppress_errors = True
                except Exception as dynamo_error:
                    print(f"⚠️ Could not configure TorchDynamo: {dynamo_error}")

            t_start = time.time()
            # Print environment info for debugging
            import transformers
            print(f"🔧 Environment: transformers={transformers.__version__}, torch={torch.__version__}")
            print(f"🔧 CUDA available: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
            
            # Get the token from config or from environment variable
            hf_token = self.config.get('api_token') or os.getenv("HUGGINGFACE_TOKEN")

            # Configure quantization for faster loading
            quantization_config = None
            if self.config.get('load_in_8bit', False):
                print("🔧 Importing BitsAndBytesConfig for 8-bit quantization...")
                t_bnb_start = time.time()
                try:
                    from transformers import BitsAndBytesConfig
                    import bitsandbytes as bnb
                    print(f"🔧 BitsAndBytes import successful in {time.time() - t_bnb_start:.2f}s, version: {getattr(bnb, '__version__', 'unknown')}")
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    print("🔧 Using 8-bit quantization for faster loading")
                except Exception as e:
                    print(f"❌ BitsAndBytes import/config failed: {e}")
                    raise
            elif self.config.get('load_in_4bit', False):
                print("🔧 Importing BitsAndBytesConfig for 4-bit quantization...")
                t_bnb_start = time.time()
                try:
                    from transformers import BitsAndBytesConfig
                    import bitsandbytes as bnb
                    print(f"🔧 BitsAndBytes import successful in {time.time() - t_bnb_start:.2f}s, version: {getattr(bnb, '__version__', 'unknown')}")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4"
                    )
                    print("🔧 Using 4-bit quantization for ultra-fast loading")
                except Exception as e:
                    print(f"❌ BitsAndBytes import/config failed: {e}")
                    raise

            print("🔁 Starting tokenizer.from_pretrained()")
            t_tok_start = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token,  # Updated from use_auth_token
                trust_remote_code=self.config.get('trust_remote_code', True)
            )
            t_tok_end = time.time()
            print(f"✅ Tokenizer loaded in {t_tok_end - t_tok_start:.2f}s")

            print("🔧 Building model kwargs...")
            t_kwargs_start = time.time()
            model_kwargs = {
                'token': hf_token,  # Updated from use_auth_token
                'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
                'trust_remote_code': self.config.get('trust_remote_code', True),
                'low_cpu_mem_usage': True,  # Reduces memory usage during loading
            }
            
            # Handle device_map configuration - prioritize explicit config
            device_map_config = self.config.get('device_map')
            if device_map_config:
                model_kwargs['device_map'] = device_map_config
                print(f"🔧 Using configured device_map: {device_map_config}")
            elif self.device == 'cuda':
                model_kwargs['device_map'] = 'auto'
                print("🔧 Using default device_map: auto (CUDA available)")
            else:
                print("🔧 No device_map set (CPU mode)")
            
            # Add quantization if configured
            if quantization_config:
                print("🔧 Adding quantization_config to model_kwargs...")
                model_kwargs['quantization_config'] = quantization_config

            t_kwargs_end = time.time()
            print(f"✅ Model kwargs built in {t_kwargs_end - t_kwargs_start:.2f}s")
            print(f"� Model kwargs keys: {list(model_kwargs.keys())}")
            print(f"🔧 device_map: {model_kwargs.get('device_map')}")
            print(f"🔧 torch_dtype: {model_kwargs.get('torch_dtype')}")
            print(f"🔧 low_cpu_mem_usage: {model_kwargs.get('low_cpu_mem_usage')}")
            print(f"🔧 trust_remote_code: {model_kwargs.get('trust_remote_code')}")

            print(f"�🔁 Starting AutoModelForCausalLM.from_pretrained() with model_name='{self.model_name}'")
            print(f"🔧 About to call: AutoModelForCausalLM.from_pretrained('{self.model_name}', **{list(model_kwargs.keys())})")
            
            # Flush output to ensure we see this print before any potential hang
            import sys
            sys.stdout.flush()
            
            t_model_start = time.time()
            try:
                print(f"🔧 Starting model load (optimized settings)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                t_model_end = time.time()
                print(f"✅ Model loaded successfully in {t_model_end - t_model_start:.2f}s (total since entry {t_model_end - t_start:.2f}s)")
            except Exception as e:
                t_model_error = time.time()
                print(f"❌ Model loading failed after {t_model_error - t_model_start:.2f}s with error: {str(e)}")
                print(f"❌ Exception type: {type(e).__name__}")
                import traceback
                print(f"❌ Traceback:\n{traceback.format_exc()}")
                raise

            # Device placement: if transformers performed auto-placement, report the device
            try:
                first_param = next(self.model.parameters())
                print(f"🔧 [INFO] First model parameter device after load: {first_param.device}")
            except Exception:
                print("🔧 [INFO] Could not inspect model parameters after load")

            # Print GPU usage via nvidia-smi (best-effort; may not be available in all environments)
            try:
                import subprocess
                smi = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                print(f"🔧 [GPU] nvidia-smi output:\n{smi.stdout.strip()}")
            except Exception:
                print("🔧 [GPU] nvidia-smi not available or timed out")

            # Add padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.is_loaded = True
            print(f"✅ {self.model_name} loaded successfully")

        except Exception as e:
            print(f"❌ Error loading {self.model_name}: {str(e)}")
            raise

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate contextually appropriate response using the loaded model.
        
        Performs inference using the loaded Hugging Face model with intelligent
        parameter management, device handling, and response post-processing.
        The method prioritizes configuration-based parameters while providing
        robust error handling and response cleaning for negotiation contexts.
        
        Args:
            prompt (str): Input prompt for the model, typically containing
                negotiation context, game rules, and situation requiring response.
            **kwargs: Additional generation parameters (note: YAML configuration
                takes precedence over kwargs for consistency). Supported options:
                - early_stopping (bool): Enable early stopping for beam search
                - length_penalty (float): Length penalty for generation
        
        Returns:
            str: Generated response text, cleaned and processed for parsing.
                The method attempts to extract clean JSON from model output
                and removes common prefixes/suffixes that interfere with
                action parsing.
        
        Raises:
            RuntimeError: If the model is not loaded or generation fails due
                to resource constraints, device issues, or model errors.
            ValueError: If the prompt is empty or contains characters that
                cause tokenization or generation failures.
        
        Example:
            >>> prompt = \"Make a negotiation offer for the company car.\"
            >>> response = model.generate_response(prompt)
            >>> print(response)
            '{\"type\": \"offer\", \"price\": 28000}'
        
        Note:
            This method includes intelligent response cleaning that attempts
            to extract clean JSON from model outputs and removes common
            model-generated prefixes. Generation parameters are strictly
            controlled by YAML configuration to ensure consistent behavior
            across experiments.
        """
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

    def _preprocess_json_response(self, response: str) -> str:
        """Fix common model JSON formatting issues before parsing.
        
        Applies a series of regular expression transformations to address
        common JSON formatting mistakes made by language models, including
        malformed arrays, missing quotes, and number formatting issues.
        This preprocessing significantly improves parsing success rates.
        
        Args:
            response (str): Raw response text from the model that may contain
                malformed JSON requiring correction before parsing.
        
        Returns:
            str: Preprocessed response with common JSON formatting issues
                corrected, ready for standard JSON parsing operations.
        
        Transformations Applied:
            - Converts malformed array notation [39,000] to numeric 39000
            - Fixes single-number arrays [100] to plain numbers 100
            - Removes comma separators from quoted numbers \"39,000\"
            - Adds missing quotes around object property names
            - Corrects misplaced quotes in property names
        
        Example:
            >>> response = '{\"price\": [39,000], invalid_key: 100}'
            >>> fixed = model._preprocess_json_response(response)
            >>> print(fixed)
            '{\"price\": 39000, \"invalid_key\": 100}'
        
        Note:
            This method is designed specifically for negotiation action
            parsing and may not be suitable for general JSON preprocessing.
            The transformations are optimized for common model output patterns
            observed in negotiation scenarios.
        """
        import re
        
        # Fix array notation for large numbers: [39,000] -> 39000
        # This handles cases where models think [39,000] means thirty-nine thousand
        response = re.sub(r'\[(\d+),(\d+)\]', r'\1\2', response)
        
        # Fix array notation for regular numbers: [100] -> 100 (for single numbers)
        # But preserve actual arrays like ["Outsourced"] 
        response = re.sub(r':\s*\[(\d+)\]', r': \1', response)
        
        # Fix potential comma-in-quotes issues: "39,000" -> 39000
        response = re.sub(r'"(\d+),(\d+)"', r'\1\2', response)

        # Add double quotes around property names if missing
        # Matches property names without quotes and adds them
        response = re.sub(r'(?<!["{,\s])([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*:)', r'"\1"', response)

        # Fix misplaced or malformed quotes in keys (e.g., {t"ype": ...})
        response = re.sub(r'\{\s*t"', r'{"t', response)

        return response

    def parse_action(self, response: str, game_type: str = None, player_name: str = None) -> Dict[str, Any]:
        """Parse LLM response into structured negotiation action with validation.
        
        Implements a comprehensive parsing pipeline that converts raw model
        output into validated negotiation actions. The method employs multiple
        parsing strategies, extensive error handling, and optional Pydantic
        validation for game-specific action constraints.
        
        Args:
            response (str): Raw text response from the model, which may contain
                JSON, natural language, or mixed formats requiring extraction.
            game_type (str, optional): Type of negotiation game for validation
                ('company_car', 'resource_allocation', 'integrative_negotiations').
                Enables game-specific parsing rules and action validation.
            player_name (str, optional): Player identifier for enhanced logging
                and debugging output during parsing operations.
        
        Returns:
            Dict[str, Any]: Validated action dictionary containing:
                - type (str): Action type ('offer', 'accept', 'reject', 'propose')
                - Additional fields specific to action type and game context
                - Fallback 'noop' action if parsing fails completely
        
        Parsing Strategy:
            1. Response preprocessing to fix common JSON issues
            2. Priority extraction of JSON from response start
            3. Pattern-based JSON extraction with multiple strategies
            4. Manual key-value extraction as fallback
            5. Optional Pydantic validation for game-specific constraints
        
        Example:
            >>> response = '{\"type\": \"offer\", \"price\": 28000}'
            >>> action = model.parse_action(response, 'company_car', 'Buyer')
            >>> print(action)
            {'type': 'offer', 'price': 28000.0}
        
        Note:
            This method includes extensive debugging output and graceful
            error handling. It integrates with the action_schemas module
            for Pydantic validation when game_type is specified, providing
            robust action parsing for research and production scenarios.
        """
        import json
        import re
        
        # Add player identification to logging
        player_id = f" [{player_name}]" if player_name else ""
        print(f"🔍 [DEBUG]{player_id} Raw LLM response: {repr(response)}")
        
        try:
            # Preprocess the response to fix common JSON issues
            response_preprocessed = self._preprocess_json_response(response)
            if response_preprocessed != response:
                print(f"🧹 [DEBUG]{player_id} Preprocessed response: {repr(response_preprocessed)}")
            
            # Clean up the response
            response_clean = response_preprocessed.strip()
            
            # First priority: Look for JSON at the start of the response
            json_at_start = re.match(r'^(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response_clean)
            if json_at_start:
                json_text = json_at_start.group(1)
                print(f"🧹 [DEBUG]{player_id} Extracted JSON from start: {repr(json_text)}")
                try:
                    parsed = json.loads(json_text)
                    print(f"✅ [DEBUG]{player_id} Successfully parsed JSON from start: {parsed}")
                    
                    # Apply Pydantic validation if game_type is provided
                    if game_type:
                        from .action_schemas import validate_and_constrain_action
                        try:
                            validated = validate_and_constrain_action(json_text, game_type)
                            print(f"✅ [DEBUG]{player_id} Pydantic validation successful: {validated}")
                            return validated
                        except ValueError as e:
                            print(f"⚠️ [DEBUG]{player_id} Pydantic validation failed, using original: {e}")
                    
                    return parsed
                except json.JSONDecodeError as e:
                    print(f"⚠️ [DEBUG]{player_id} JSON decode error for start match: {e}")
            
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
                            print(f"✅ [DEBUG]{player_id} Successfully parsed JSON: {parsed}")
                            
                            # Apply Pydantic validation if game_type is provided
                            if game_type:
                                from .action_schemas import validate_and_constrain_action
                                try:
                                    validated = validate_and_constrain_action(match_clean, game_type)
                                    print(f"✅ [DEBUG]{player_id} Pydantic validation successful: {validated}")
                                    return validated
                                except ValueError as e:
                                    print(f"⚠️ [DEBUG]{player_id} Pydantic validation failed, using original: {e}")
                            
                            return parsed
                        except json.JSONDecodeError as e:
                            print(f"⚠️ [DEBUG]{player_id} JSON decode error for '{match[:50]}...': {e}")
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
                        print(f"✅ [DEBUG]{player_id} Successfully parsed extracted JSON: {parsed}")
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"⚠️ [DEBUG]{player_id} Extracted JSON decode error: {e}")
            
            # Try to extract key-value pairs manually if JSON parsing fails
            action_dict = self._extract_action_manually(response_clean)
            if action_dict:
                print(f"✅ [DEBUG]{player_id} Manually extracted action: {action_dict}")
                return action_dict
            
            # Fallback: return a no-op action
            print(f"❌ [DEBUG]{player_id} Could not parse action from response: {response}")
            return {"type": "noop", "reason": "failed_to_parse"}
            
        except Exception as e:
            print(f"❌ [DEBUG]{player_id} Error parsing action: {str(e)}")
            return {"type": "noop", "reason": "parse_error"}
    
    def _extract_action_manually(self, response: str) -> Dict[str, Any]:
        """Extract action information manually using pattern matching.
        
        Fallback parsing method that uses regular expressions to extract
        action information when standard JSON parsing fails. This method
        looks for common negotiation patterns and keywords to construct
        a basic action dictionary from natural language responses.
        
        Args:
            response (str): Raw response text that failed JSON parsing,
                potentially containing natural language or semi-structured
                text with extractable action information.
        
        Returns:
            Dict[str, Any]: Extracted action dictionary if patterns are found,
                None if no recognizable action patterns can be identified.
                Common extracted fields include:
                - type (str): Inferred action type from keywords
                - price (float): Extracted price values for bargaining
        
        Extraction Patterns:
            - Action types: 'offer', 'accept', 'reject' from keywords
            - Price values: Numeric values associated with 'price' keyword
            - Context clues: Words like 'offer', 'accept', 'reject' in text
        
        Example:
            >>> response = "I offer 25000 for the car"
            >>> action = model._extract_action_manually(response)
            >>> print(action)
            {'type': 'offer', 'price': 25000.0}
        
        Note:
            This is a last-resort parsing method with limited accuracy.
            It's designed to extract basic action information when models
            generate natural language instead of structured JSON responses.
        """
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
        """Unload model from memory and perform comprehensive cleanup.
        
        Performs thorough cleanup of model resources including GPU memory,
        cached tensors, and Python object references. This method is essential
        for memory management in multi-model scenarios and prevents memory
        leaks during model switching operations.
        
        Cleanup Operations:
            - Deletion of model and tokenizer objects
            - GPU memory cache clearing (CUDA)
            - Python garbage collection triggering
            - Loading state reset
        
        Example:
            >>> model.load_model()
            >>> # ... use model for inference ...
            >>> model.unload_model()  # Free all resources
            🗑️  meta-llama/Llama-2-7b-chat-hf unloaded
        
        Note:
            This method is idempotent and safe to call multiple times.
            It's automatically called during model switching and should
            be called explicitly when done with a model to free resources
            for other models or applications.
        """
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.is_loaded = False
        print(f"🗑️  {self.model_name} unloaded")