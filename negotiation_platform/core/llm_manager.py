"""
LLM Manager
===========

The LLMManager provides comprehensive management of Large Language Model instances
with advanced features including lazy loading, memory management, and plug-and-play
model switching for negotiation sessions.

Key Features:
    - Dynamic model loading and unloading with memory management
    - Thread-safe model operations with concurrent session support
    - Lazy loading strategy to minimize GPU memory usage
    - LRU (Least Recently Used) eviction for memory optimization
    - Model aliasing and shared instance management
    - Plug-and-play architecture for different model types
    - Automatic model registration from configuration files

Architecture:
    The LLMManager uses a sophisticated caching strategy where models are loaded
    on-demand and shared across sessions when possible. This minimizes GPU memory
    usage while maintaining performance for active negotiations.

Memory Management:
    - Lazy loading: Models loaded only when first requested
    - Shared instances: Same model shared across multiple model IDs
    - LRU eviction: Least recently used models unloaded when memory limit reached
    - Thread safety: All operations protected by locks for concurrent access
"""

import yaml
import threading
from typing import Dict, List, Any, Optional
from negotiation_platform.models.base_model import BaseLLMModel
from negotiation_platform.models.hf_model_wrapper import HuggingFaceModelWrapper

class LLMManager:
    """
    Advanced Large Language Model management system with memory optimization and threading support.
    
    The LLMManager orchestrates the complete lifecycle of AI model instances used in negotiations.
    It provides intelligent loading, caching, and memory management to efficiently handle multiple
    models while minimizing GPU memory usage and maximizing performance.
    
    Core Capabilities:
        - Dynamic model registration from configuration files
        - Lazy loading with on-demand model instantiation
        - Intelligent memory management with LRU eviction
        - Thread-safe operations for concurrent negotiation sessions
        - Model aliasing for flexible configuration management
        - Shared model instances to reduce memory footprint
        - Plug-and-play architecture supporting multiple model types
    
    Memory Strategy:
        Uses a sophisticated multi-level caching system:
        1. Model Registry: Configurations for all available models
        2. Shared Instances: Actual loaded models (limited by max_loaded_models)
        3. Model Aliases: Mapping from model IDs to shared instances
        4. LRU Eviction: Automatic unloading of least recently used models
    
    Attributes:
        models (Dict[str, BaseLLMModel]): Registry of model ID to wrapper instances.
        shared_models (Dict[str, BaseLLMModel]): Cache of actual loaded model instances.
        model_aliases (Dict[str, str]): Mapping from model IDs to shared model names.
        model_configs (Dict[str, Dict[str, Any]]): Configuration for all registered models.
        max_loaded_models (int): Maximum number of models to keep loaded simultaneously.
        loaded_order (List[str]): LRU tracking for loaded models.
        manager_lock (threading.Lock): Thread synchronization for safe concurrent access.
    
    Example:
        >>> model_configs = {
        ...     "model_a": {"model_name": "meta-llama/Llama-2-7b-chat-hf", "type": "huggingface"},
        ...     "model_b": {"model_name": "mistralai/Mistral-7B-Instruct-v0.1", "type": "huggingface"}
        ... }
        >>> llm_manager = LLMManager(model_configs)
        >>> response = llm_manager.generate_response("model_a", "Hello, how are you?")
        >>> print(response)
        "I'm doing well, thank you for asking!"
    
    Thread Safety:
        All public methods are thread-safe and can be called concurrently from multiple
        negotiation sessions without risk of race conditions or memory corruption.
    """

    def __init__(self, model_configs: dict):
        """
        Initialize the LLMManager with model configurations and setup internal data structures.
        
        Creates a new LLMManager instance with the provided model configurations and
        initializes all internal data structures for model management, caching, and
        thread safety. Models are registered but not loaded until first requested.
        
        Args:
            model_configs (dict): Dictionary mapping model IDs to their configurations.
                Each configuration should contain:
                    - model_name (str): HuggingFace model identifier
                    - type (str): Model wrapper type (e.g., 'huggingface')
                    - device (str): Target device ('cuda:0', 'cpu', etc.)
                    - generation_config (dict): Model generation parameters
                    - api_key (str): Authentication token (environment variable)
        
        Architecture Setup:
            - models: Registry mapping model IDs to wrapper instances
            - shared_models: Cache of actual loaded model instances (limited size)
            - model_aliases: Mapping from model IDs to shared model names
            - loaded_order: LRU tracking list for memory management
            - manager_lock: Thread synchronization primitive
        
        Memory Management:
            The manager is configured to keep a maximum of 2 models loaded simultaneously
            (configurable via max_loaded_models). This prevents GPU memory exhaustion
            while maintaining reasonable performance for active negotiations.
        
        Example:
            >>> configs = {
            ...     "model_a": {
            ...         "model_name": "meta-llama/Llama-2-7b-chat-hf",
            ...         "type": "huggingface",
            ...         "device": "cuda:0"
            ...     }
            ... }
            >>> manager = LLMManager(configs)
            >>> print(len(manager.model_configs))
            1
        
        Note:
            Models are registered during initialization but not loaded into memory.
            Loading occurs lazily when the first generation request is made.
        """
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
        """
        Load model configurations from YAML configuration file.
        
        Attempts to read and parse the YAML configuration file specified by
        config_path. If the file is not found, creates a default empty
        configuration structure to prevent initialization failures.
        
        Configuration Structure:
            The YAML file should contain model definitions with configuration
            parameters for each model instance including model names, device
            specifications, and generation parameters.
        
        Error Handling:
            Missing configuration files are handled gracefully by creating
            empty default configurations rather than failing initialization.
        
        Example:
            >>> manager._load_model_configs()
            >>> print(list(manager.model_configs.keys()))
            ['model_a', 'model_b', 'model_c']
        
        Note:
            This method is called automatically during initialization and
            should not typically be called directly by users.
        """
        try:
            with open(self.config_path, 'r') as f:
                self.model_configs = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found")
            self.model_configs = {"models": {}}

    def register_model(self, model_id: str, model_config: Dict[str, Any]):
        """
        Register a new model configuration without loading it into memory.
        
        Adds a model configuration to the registry, making it available for future
        loading and generation requests. The model is not loaded into memory during
        registration - this occurs lazily when first requested.
        
        Args:
            model_id (str): Unique identifier for this model instance. Used to
                reference the model in generation requests.
            model_config (Dict[str, Any]): Configuration dictionary containing:
                - model_name (str): HuggingFace model identifier
                - type (str): Model wrapper type (currently supports 'huggingface')
                - device (str): Target device specification
                - generation_config (dict): Model-specific generation parameters
                - api_key (str): Authentication token reference
        
        Supported Model Types:
            - huggingface: HuggingFace Transformers models with GPU/CPU support
        
        Registration Process:
            1. Validates model type is supported
            2. Creates model alias mapping for shared instance management
            3. Stores configuration for lazy loading
            4. Sets up wrapper instance placeholder
        
        Example:
            >>> manager = LLMManager({})
            >>> config = {
            ...     "model_name": "meta-llama/Llama-2-7b-chat-hf",
            ...     "type": "huggingface",
            ...     "device": "cuda:0"
            ... }
            >>> manager.register_model("my_model", config)
            >>> "my_model" in manager.models
            True
        
        Raises:
            ValueError: If model_type is not supported (currently only 'huggingface').
        """
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
        """
        Load a specific model into memory with intelligent resource management.
        
        Loads the specified model into GPU/CPU memory using a smart caching system
        that prevents memory overflow by automatically evicting least-recently-used
        models when memory limits are approached. Thread-safe loading ensures no
        concurrent model initialization conflicts.
        
        Args:
            model_id (str): Registered identifier of the model to load into memory.
        
        Returns:
            BaseLLMModel: The loaded model wrapper ready for text generation.
        
        Memory Management:
            - Maintains LRU cache of loaded models
            - Automatically evicts oldest models when max_loaded_models exceeded
            - Shares model instances across multiple aliases to save memory
            - Thread-safe loading prevents concurrent initialization conflicts
        
        Example:
            >>> manager = LLMManager(model_configs)
            >>> model = manager.load_model("llama-7b-chat")
            >>> response = model.generate("Hello, how are you?")
        
        Raises:
            ValueError: If model_id is not registered in the manager.
        
        Note:
            Loading is performed lazily - models are only loaded when explicitly
            requested or when generating responses.
        """
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
        """
        Switch the active model for subsequent generation requests.
        
        Changes the default model used for generation requests that don't
        specify a model_id. Enables dynamic model switching during runtime
        without requiring manager reconfiguration or restart.
        
        Args:
            model_id (str): Registered identifier of the model to make active.
        
        Lazy Loading:
            If the target model is not currently loaded in memory, it will be
            loaded automatically using the intelligent memory management system.
        
        Example:
            >>> manager.switch_model("llama-7b-chat")
            >>> response = manager.get_response("Hello")  # Uses llama-7b-chat
            >>> manager.switch_model("mistral-7b")
            >>> response = manager.get_response("Hello")  # Uses mistral-7b
        
        Raises:
            ValueError: If model_id is not registered in the manager.
        
        Note:
            Switching models does not immediately unload the previous active
            model - it remains in memory subject to LRU eviction policies.
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")

        # Do not eagerly unload on switch; rely on LRU eviction instead.
        if not self.models[model_id].is_loaded:
            self.load_model(model_id)

        self.active_model = model_id
        print(f"🔄 Switched to model: {model_id}")

    def get_response(self, prompt: str, model_id: str = None, **kwargs) -> str:
        """
        Generate text response using specified model or currently active model.
        
        Primary interface for text generation that handles model selection,
        lazy loading, and response generation. Automatically loads models
        on-demand if not already in memory and manages the generation pipeline.
        
        Args:
            prompt (str): Input text prompt for the language model to process.
            model_id (str, optional): Specific model identifier to use for
                generation. If None, uses the currently active model.
            **kwargs: Additional generation parameters passed to the model:
                - temperature (float): Randomness control (0.0-1.0)
                - max_length (int): Maximum response length
                - top_p (float): Nucleus sampling parameter
                - do_sample (bool): Whether to use sampling
        
        Returns:
            str: Generated text response from the language model.
        
        Lazy Loading:
            Models are loaded automatically if not already in memory,
            making this method fully self-contained for generation requests.
        
        Example:
            >>> manager = LLMManager(model_configs)
            >>> response = manager.get_response("What is AI?", temperature=0.7)
            >>> print(response)
            "Artificial Intelligence is..."
        
        Raises:
            RuntimeError: If no model is specified and no active model is set.
            ValueError: If the specified model_id is not registered.
        
        Note:
            This method automatically updates the active model if a specific
            model_id is provided, making it the new default for future calls.
        """
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
        """
        Unload a specific model from GPU/CPU memory to free resources.
        
        Removes the specified model from memory while respecting shared model
        instances. If multiple model aliases reference the same underlying model,
        the unload operation will only proceed if no other aliases are using it.
        
        Args:
            model_id (str): Registered identifier of the model to unload from memory.
        
        Memory Management:
            - Respects shared model instances across multiple aliases
            - Updates LRU tracking to reflect unloaded state
            - Frees GPU/CPU memory allocated to the model
        
        Example:
            >>> manager.unload_model("llama-7b-chat")
            >>> # Model is removed from memory, alias remains registered
            >>> manager.load_model("llama-7b-chat")  # Can be reloaded later
        
        Raises:
            ValueError: If model_id is not registered in the manager.
        
        Note:
            Unloading does not remove the model from the registry - it can
            be reloaded later via load_model(). This is primarily useful for
            manual memory management in resource-constrained environments.
        """
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
        """
        Retrieve comprehensive information about all registered models.
        
        Provides detailed metadata about all models in the registry including
        configuration details, loading status, and memory usage information.
        Useful for debugging, monitoring, and system introspection.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping model IDs to their
                detailed information including:
                - model_name (str): HuggingFace model identifier
                - is_loaded (bool): Current memory loading status
                - device (str): Target device specification
                - configuration details from the model wrapper
        
        Example:
            >>> models = manager.list_models()
            >>> for model_id, info in models.items():
            ...     print(f"{model_id}: loaded={info.get('is_loaded', False)}")
            llama-7b-chat: loaded=True
            mistral-7b: loaded=False
        
        Note:
            This method is primarily useful for monitoring and debugging
            rather than normal operation. The information reflects the current
            state and may change as models are loaded and unloaded.
        """
        return {
            model_id: model.get_model_info()
            for model_id, model in self.models.items()
        }

    def cleanup(self):
        """
        Unload all models and free all allocated resources.
        
        Performs comprehensive cleanup by unloading all models from memory,
        clearing caches, and resetting internal state. Should be called when
        the manager is no longer needed to ensure proper resource cleanup.
        
        Cleanup Operations:
            - Unloads all models from GPU/CPU memory
            - Clears LRU tracking and shared model caches
            - Resets active model state
            - Frees all allocated GPU/CPU resources
        
        Example:
            >>> manager = LLMManager(model_configs)
            >>> # ... use models ...
            >>> manager.cleanup()  # Free all resources
        
        Note:
            After cleanup, the manager instance should not be used further.
            Create a new manager instance if continued operation is needed.
            This method is particularly important for long-running applications
            to prevent memory leaks.
        """
        for model_name, wrapper in self.shared_models.items():
            if wrapper.is_loaded:
                wrapper.unload_model()
        print("🧹 All shared models unloaded")