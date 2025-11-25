"""
Base Model Interface
===================

Abstract base class interface for plug-and-play Large Language Model (LLM)
integration in the negotiation platform. This module defines the standardized
API that all model implementations must follow to ensure compatibility and
interchangeability within the platform architecture.

The interface supports various model types including Hugging Face transformers,
commercial APIs, and custom implementations through a unified abstraction layer.
All concrete implementations must provide model loading, response generation,
action parsing, and resource cleanup capabilities.

Key Features:
    - Standardized model lifecycle management (load/unload)
    - Unified response generation interface with configurable parameters
    - Game-specific action parsing with validation support
    - Memory management and resource optimization hooks
    - Extensible configuration system for model-specific parameters

Example:
    >>> from negotiation_platform.models.hf_model_wrapper import HuggingFaceModelWrapper
    >>> config = {'max_new_tokens': 120, 'temperature': 0.7}
    >>> model = HuggingFaceModelWrapper('meta-llama/Llama-2-7b-chat-hf', config)
    >>> model.load_model()
    >>> response = model.generate_response('Your negotiation prompt here')
    >>> action = model.parse_action(response, 'company_car', 'Player1')

Note:
    This is an abstract base class and cannot be instantiated directly.
    Use concrete implementations like HuggingFaceModelWrapper for actual
    model integration.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseLLMModel(ABC):
    """Abstract base class for all Large Language Model implementations.
    
    This class defines the standardized interface that all LLM model wrappers
    must implement to ensure compatibility and interchangeability within the
    negotiation platform. It provides the foundation for supporting various
    model types and APIs through a unified abstraction layer.
    
    The interface encompasses the complete model lifecycle from initialization
    and loading through response generation and resource cleanup. All methods
    are designed to support both synchronous and configuration-driven usage
    patterns common in negotiation research scenarios.
    
    Attributes:
        model_name (str): Unique identifier for the model (e.g., model path,
            API endpoint, or registry name).
        config (Dict[str, Any]): Model-specific configuration parameters
            including generation settings, API keys, and optimization flags.
        is_loaded (bool): Current loading state of the model, used for
            resource management and error prevention.
    
    Example:
        >>> class CustomModelWrapper(BaseLLMModel):
        ...     def load_model(self):
        ...         # Custom implementation
        ...         self.is_loaded = True
        ...     def generate_response(self, prompt):
        ...         # Custom response generation
        ...         return "Generated response"
        
    Note:
        Concrete implementations should handle model-specific concerns such as
        authentication, memory management, GPU utilization, and error recovery
        while maintaining compatibility with this interface.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """Initialize the base LLM model with configuration.
        
        Sets up the foundational attributes required by all model implementations
        including the model identifier, configuration parameters, and loading
        state tracking. Subclasses should call this constructor before
        performing any model-specific initialization.
        
        Args:
            model_name (str): Unique identifier for the model, which may
                represent a file path, model registry name, API endpoint,
                or other model-specific identifier used for loading.
            config (Dict[str, Any]): Comprehensive configuration dictionary
                containing model-specific parameters such as:
                - Generation settings (temperature, max_tokens, top_p)
                - Hardware preferences (device, precision, memory_limits)
                - API credentials and endpoints (for cloud models)
                - Optimization flags (quantization, batching, caching)
        
        Example:
            >>> config = {
            ...     'temperature': 0.7,
            ...     'max_new_tokens': 120,
            ...     'device': 'cuda',
            ...     'load_in_8bit': True
            ... }
            >>> model = CustomModelWrapper('model-name', config)
        
        Note:
            The model is initialized in an unloaded state. Call load_model() 
            explicitly to prepare the model for inference operations.
        """
        self.model_name = model_name
        self.config = config
        self.is_loaded = False

    @abstractmethod
    def load_model(self):
        """Load the model into memory and prepare for inference.
        
        This method performs all necessary initialization steps to make the
        model ready for generating responses. Implementation should handle
        model downloading, memory allocation, device placement, and any
        required preprocessing or optimization steps.
        
        The method should be idempotent - calling it multiple times should
        not cause errors or unnecessary resource consumption. Implementations
        should update the is_loaded attribute to reflect the current state.
        
        Raises:
            RuntimeError: If model loading fails due to insufficient resources,
                network issues, authentication problems, or other critical errors.
            FileNotFoundError: If the specified model cannot be located or
                accessed from the configured source.
            MemoryError: If insufficient memory is available for model loading.
        
        Example:
            >>> model = HuggingFaceModelWrapper('model-name', config)
            >>> model.load_model()  # Loads model into GPU/CPU memory
            >>> assert model.is_loaded == True
        
        Note:
            Implementations should handle device placement, quantization,
            memory optimization, and other hardware-specific concerns based
            on the configuration provided during initialization.
        """
        pass

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a text response from the loaded model.
        
        This method performs inference using the loaded model to generate
        a text response based on the provided prompt. The response should
        be contextually appropriate for negotiation scenarios and follow
        any game-specific formatting requirements.
        
        Args:
            prompt (str): Input text prompt for the model, typically containing
                negotiation context, game rules, player role, and current
                situation requiring a response.
            **kwargs: Additional generation parameters that may override
                default configuration settings. Common parameters include:
                - temperature (float): Sampling temperature for randomness
                - max_new_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter
                - do_sample (bool): Whether to use sampling vs greedy decoding
        
        Returns:
            str: Generated text response from the model, which should be
                parseable into structured actions for the negotiation game.
                The response may contain JSON, natural language, or mixed
                formats depending on the model and prompt design.
        
        Raises:
            RuntimeError: If the model is not loaded or generation fails
                due to resource constraints or model errors.
            ValueError: If the prompt is empty, too long, or contains
                invalid characters that the model cannot process.
        
        Example:
            >>> prompt = "You are a buyer in a car negotiation. Make an offer."
            >>> response = model.generate_response(prompt, temperature=0.5)
            >>> print(response)
            '{"type": "offer", "price": 25000}'
        
        Note:
            Implementations should handle prompt preprocessing, generation
            parameter validation, output post-processing, and error recovery
            while maintaining consistent response quality.
        """
        pass

    @abstractmethod
    def parse_action(self, response: str, game_type: str = None, player_name: str = None) -> Dict[str, Any]:
        """Parse LLM response into a structured negotiation action.
        
        This method converts the raw text output from the model into a
        standardized action dictionary that can be processed by the game
        engine. It should handle various response formats, apply validation,
        and provide error recovery for malformed responses.
        
        Args:
            response (str): Raw text response from the model, which may
                contain JSON, structured text, or natural language that
                needs to be extracted and parsed into action format.
            game_type (str, optional): Type of negotiation game being played
                (e.g., 'company_car', 'resource_allocation', 'integrative_negotiations'),
                used for game-specific parsing and validation rules.
            player_name (str, optional): Name or identifier of the player
                generating the action, used for debugging and error reporting.
        
        Returns:
            Dict[str, Any]: Structured action dictionary containing at minimum:
                - type (str): Action type (e.g., 'offer', 'accept', 'reject')
                - Additional fields specific to the action type and game:
                  * price (float): For offer/counter actions in price bargaining
                  * gpu_hours/cpu_hours (float): For resource allocation proposals
                  * proposal (dict): For integrative negotiation proposals
        
        Raises:
            ValueError: If the response cannot be parsed into a valid action
                or contains invalid values for the specified game type.
            KeyError: If required action fields are missing from the parsed
                response for the given game context.
        
        Example:
            >>> response = '{"type": "offer", "price": 28000}'
            >>> action = model.parse_action(response, 'company_car', 'Buyer')
            >>> print(action)
            {'type': 'offer', 'price': 28000.0}
        
        Note:
            Implementations should provide robust parsing with fallback
            strategies for common formatting issues, and should integrate
            with validation schemas when available for the game type.
        """
        pass

    @abstractmethod
    def unload_model(self):
        """Unload model from memory and free associated resources.
        
        This method performs cleanup operations to release memory and other
        system resources used by the model. It should handle GPU memory
        cleanup, cache clearing, and any other resource deallocation needed
        to prepare for model switching or application shutdown.
        
        The method should be idempotent - calling it multiple times should
        not cause errors. After successful unloading, the is_loaded attribute
        should be set to False to reflect the model state.
        
        Raises:
            RuntimeError: If unloading fails or resources cannot be properly
                released, though implementations should make best efforts
                to continue with partial cleanup in such cases.
        
        Example:
            >>> model.load_model()
            >>> assert model.is_loaded == True
            >>> model.unload_model()
            >>> assert model.is_loaded == False
        
        Note:
            This method is critical for memory management in multi-model
            scenarios where models are dynamically loaded and switched.
            Implementations should ensure thorough cleanup including GPU
            memory, cached tensors, and any background processes.
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Retrieve comprehensive information about the model instance.
        
        This method provides a standardized way to inspect the current state
        and configuration of a model instance. It returns information useful
        for debugging, monitoring, logging, and administrative purposes.
        
        Returns:
            Dict[str, Any]: Information dictionary containing:
                - name (str): The model identifier used during initialization
                - loaded (bool): Current loading state of the model
                - config (Dict[str, Any]): Complete configuration dictionary
                  including all parameters used for model initialization
                  and generation settings
        
        Example:
            >>> model = HuggingFaceModelWrapper('model-name', {'temp': 0.7})
            >>> info = model.get_model_info()
            >>> print(info)
            {
                'name': 'model-name',
                'loaded': False,
                'config': {'temp': 0.7}
            }
        
        Note:
            This method does not modify the model state and can be called
            safely at any time. Subclasses may extend this method to include
            additional model-specific information such as memory usage,
            device placement, or performance statistics.
        """
        return {
            "name": self.model_name,
            "loaded": self.is_loaded,
            "config": self.config
        }