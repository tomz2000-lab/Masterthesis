Configuration
=============

The Negotiation Platform uses YAML configuration files to manage models, games, and platform settings.

Configuration Files
-------------------

The platform uses three main configuration files:

1. ``model_configs.yaml`` - Model definitions and parameters
2. ``game_configs.yaml`` - Game-specific settings 
3. ``platform_config.yaml`` - General platform settings

Model Configuration
-------------------

**File**: ``negotiation_platform/configs/model_configs.yaml``

.. code-block:: yaml

   models:
     # Hugging Face models
     dialogpt_medium:
       type: "huggingface"
       model_name: "microsoft/DialoGPT-medium"
       device_map: "auto"
       torch_dtype: "float16"
       load_in_8bit: false
       max_new_tokens: 256
       temperature: 0.7
       do_sample: true
       
     dialogpt_large:
       type: "huggingface" 
       model_name: "microsoft/DialoGPT-large"
       device_map: "auto"
       torch_dtype: "float16"
       load_in_8bit: true  # Enable 8-bit quantization
       max_new_tokens: 512
       temperature: 0.8
       
     # Custom model example
     custom_negotiator:
       type: "huggingface"
       model_name: "path/to/your/model"
       device_map: "cuda:0"
       torch_dtype: "float32"
       custom_parameters:
         repetition_penalty: 1.1
         top_k: 50
         top_p: 0.9

**Model Configuration Parameters**:

* ``type``: Model type (currently supports "huggingface")
* ``model_name``: Hugging Face model identifier or local path
* ``device_map``: Device allocation ("auto", "cuda:0", "cpu", etc.)
* ``torch_dtype``: Data type ("float16", "float32", "auto")
* ``load_in_8bit``: Enable 8-bit quantization for memory efficiency
* ``max_new_tokens``: Maximum tokens to generate
* ``temperature``: Sampling temperature (0.1-2.0)
* ``do_sample``: Enable sampling vs greedy decoding

Game Configuration
------------------

**File**: ``negotiation_platform/configs/game_configs.yaml``

.. code-block:: yaml

   games:
     company_car:
       starting_price: 45000
       buyer_budget: 40000
       seller_cost: 38000  
       buyer_batna: 44000
       seller_batna: 39000
       rounds: 5
       batna_decay:
         buyer: 0.05    # 5% decay per round
         seller: 0.03   # 3% decay per round
         
     resource_allocation:
       total_gpu_hours: 80
       total_cpu_hours: 160
       deadline_pressure: 0.1
       dev_team_priorities:
         gpu_weight: 0.8
         cpu_weight: 0.2
       marketing_team_priorities:
         gpu_weight: 0.3
         cpu_weight: 0.7
         
     integrative_negotiations:
       total_budget: 100000
       project_duration_weeks: 12
       quality_importance:
         buyer: 0.6
         seller: 0.4
       timeline_flexibility:
         buyer: 0.3
         seller: 0.8

**Company Car Parameters**:

* ``starting_price``: Initial asking price
* ``buyer_budget``: Maximum buyer can spend
* ``seller_cost``: Minimum seller will accept
* ``buyer_batna``: Buyer's best alternative (fallback option cost)
* ``seller_batna``: Seller's minimum acceptable price
* ``rounds``: Maximum negotiation rounds
* ``batna_decay``: Per-round degradation of BATNA values

**Resource Allocation Parameters**:

* ``total_gpu_hours``: Available GPU compute time
* ``total_cpu_hours``: Available CPU compute time  
* ``deadline_pressure``: Urgency factor (0.0-1.0)
* Team priorities define relative importance of resources

Platform Configuration
----------------------

**File**: ``negotiation_platform/configs/platform_config.yaml``

.. code-block:: yaml

   platform:
     # Logging configuration
     logging:
       level: "INFO"        # DEBUG, INFO, WARNING, ERROR
       file: "negotiation.log"
       console_output: true
       
     # Session management
     session:
       max_retries: 3
       timeout_seconds: 300
       save_full_history: true
       
     # Performance settings  
     performance:
       batch_size: 1
       parallel_sessions: 1
       memory_cleanup: true
       
     # Output settings
     output:
       save_results: true
       results_directory: "results/"
       export_format: "json"  # json, csv, both

Environment Variables
---------------------

You can override configuration values using environment variables:

.. code-block:: bash

   # Model configuration
   export NEGOTIATION_MODEL_DEVICE="cuda:1"
   export NEGOTIATION_MODEL_DTYPE="float32"
   
   # Platform settings
   export NEGOTIATION_LOG_LEVEL="DEBUG"
   export NEGOTIATION_RESULTS_DIR="/path/to/results"

Configuration Loading
---------------------

The ``ConfigManager`` class handles configuration loading:

.. code-block:: python

   from negotiation_platform.core.config_manager import ConfigManager
   
   # Load default configurations
   config = ConfigManager()
   
   # Load custom configuration directory
   config = ConfigManager(config_dir="/path/to/custom/configs")
   
   # Get specific configurations
   model_configs = config.get_config("model_configs")
   game_config = config.get_game_config("company_car")
   
   # Override specific values
   custom_game_config = config.get_game_config("company_car")
   custom_game_config["rounds"] = 10  # Extend negotiation rounds

Validation
----------

Configurations are automatically validated on load:

* Model names must be valid Hugging Face identifiers or local paths
* Game parameters must be within reasonable ranges
* Required fields must be present
* Data types must match expectations

**Common Validation Errors**:

* ``ModelNotFoundError``: Invalid model name or path
* ``ConfigValidationError``: Missing required fields
* ``InvalidParameterError``: Parameter values out of valid range

Best Practices
--------------

1. **Model Selection**:
   
   * Use smaller models (medium) for development/testing
   * Use 8-bit quantization for larger models to save memory
   * Set appropriate temperature values (0.7-0.9 for negotiation)

2. **Game Configuration**:
   
   * Ensure BATNA values create realistic negotiation zones
   * Set appropriate decay rates to create time pressure
   * Balance team priorities for fair resource allocation

3. **Performance**:
   
   * Enable memory cleanup for long-running sessions
   * Use appropriate batch sizes based on available GPU memory
   * Monitor resource usage during model comparison runs

4. **Development**:
   
   * Use DEBUG logging during development
   * Save full action history for analysis
   * Create separate configs for different experimental conditions