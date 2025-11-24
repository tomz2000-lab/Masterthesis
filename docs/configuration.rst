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

   # GPU configuration for model comparison with Llama and Qwen models
   model_a:
     type: "huggingface"
     model_name: "meta-llama/Llama-3.1-8B-Instruct"
     config:
       device: "cuda"
       device_map: "auto"
       temperature: 0.7
       max_length: 16
       do_sample: true
       trust_remote_code: true
       load_in_8bit: false

   model_b:
     type: "huggingface"
     model_name: "Qwen/Qwen2.5-7B"
     config:
       device: "cuda"
       device_map: "auto"
       temperature: 0.7
       max_length: 16
       do_sample: true
       trust_remote_code: true
       load_in_8bit: false

   model_c:
     type: "huggingface"
     model_name: "meta-llama/Llama-3.2-3B-Instruct"
     config:
       device: "cuda"
       device_map: "auto"
       temperature: 0.0
       max_length: 16
       do_sample: false
       pad_token_id: 50256
       trust_remote_code: true
       load_in_8bit: false

**Model Configuration Parameters**:

* ``type``: Model type (currently supports "huggingface")
* ``model_name``: Hugging Face model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
* ``device``: Target device ("cuda", "cpu")
* ``device_map``: Device allocation strategy ("auto" for automatic GPU distribution)
* ``temperature``: Sampling temperature (0.0 for deterministic, 0.7 for creative)
* ``max_length``: Maximum sequence length for generation
* ``do_sample``: Enable sampling vs greedy decoding
* ``trust_remote_code``: Allow execution of remote code for certain models
* ``load_in_8bit``: Enable 8-bit quantization for memory efficiency
* ``pad_token_id``: Token ID for padding (model-specific)

Game Configuration
------------------

**File**: ``negotiation_platform/configs/game_configs.yaml``

.. code-block:: yaml

   company_car:
     starting_price: 45000
     buyer_budget: 42000
     seller_cost: 36000
     buyer_batna: 41000
     seller_batna: 39000
     rounds: 5
     batna_decay:
       buyer: 0.015
       seller: 0.015
     acceptance_training:
       profit_threshold: 0.10
       urgency_multiplier: 1.5
       risk_aversion: 0.8

   resource_allocation:
     total_resources: 100
     constraints:
       gpu_bandwidth: 380
       min_gpu: 5
       min_cpu: 5
     batnas:
       development: 300
       marketing: 270
     batna_decay:
       development: 0.015
       marketing: 0.015
     rounds: 5
     utility_functions:
       development:
         gpu_coefficient: 8
         cpu_coefficient: 6
         uncertainty_min: -2
         uncertainty_max: 2
       marketing:
         gpu_coefficient: 6
         cpu_coefficient: 8
         uncertainty_min: -2
         uncertainty_max: 2

   integrative_negotiations:
     issues:
       server_room:
         options: [50, 100, 150]
         points: [10, 30, 60]
       meeting_access:
         options: [2, 4, 7]
         points: [10, 30, 60]
       cleaning:
         options: ["IT", "Shared", "Outsourced"]
         points: [10, 30, 60]
       branding:
         options: ["Minimal", "Moderate", "Prominent"]
         points: [10, 30, 60]
     weights:
       IT:
         server_room: 0.4
         meeting_access: 0.1
         cleaning: 0.3
         branding: 0.2
       Marketing:
         server_room: 0.1
         meeting_access: 0.4
         cleaning: 0.2
         branding: 0.3
     batnas:
       IT: 27
       Marketing: 19
     rounds: 5
     batna_decay: 0.015

**Company Car Parameters**:

* ``starting_price``: Initial asking price (€45,000)
* ``buyer_budget``: Maximum buyer can afford (€42,000)
* ``seller_cost``: Seller's minimum cost (€36,000)
* ``buyer_batna``: Buyer's best alternative cost (€41,000)
* ``seller_batna``: Seller's minimum acceptable price (€39,000)
* ``rounds``: Maximum negotiation rounds (5)
* ``batna_decay``: Per-round BATNA degradation (1.5% for both parties)
* ``acceptance_training``: Parameters to encourage realistic acceptance behavior

**Resource Allocation Parameters**:

* ``total_resources``: Total available resource pool (100 units)
* ``constraints``: Resource allocation constraints and minimums
* ``batnas``: Team-specific BATNA values (Development: 300, Marketing: 270)
* ``utility_functions``: Team-specific utility calculations with coefficients
* ``uncertainty``: Market volatility and stochastic demand modeling

**Integrative Negotiation Parameters**:

* ``issues``: Multi-issue negotiation topics (server room, meetings, cleaning, branding)
* ``weights``: Team-specific importance weights for each issue
* ``batnas``: Minimum acceptable outcomes (IT: 27, Marketing: 19)
* ``options/points``: Available choices and point values for each issue

Platform Configuration
----------------------

**File**: ``negotiation_platform/configs/platform_config.yaml``

.. code-block:: yaml

   platform:
     logging_level: "INFO"
     max_retries: 3
     timeout_seconds: 30
     results_dir: "results"

   models:
     memory_management: true
     auto_unload: true
     default_models: ["model_a", "model_b", "model_c"]

   games:
     default_rounds: 5
     allow_early_termination: true

   metrics:
     calculate_all: true
     export_detailed: true

   experiments:
     runs_per_comparison: 10
     statistical_significance: 0.05

Environment Variables
---------------------

You can override configuration values using environment variables:

.. code-block:: bash

   # HuggingFace API token for model access
   export HF_TOKEN="your_huggingface_token_here"
   
   # Model configuration overrides
   export NEGOTIATION_MODEL_DEVICE="cuda"
   export NEGOTIATION_MODEL_TEMPERATURE="0.8"
   
   # Platform settings
   export NEGOTIATION_LOG_LEVEL="DEBUG"
   export NEGOTIATION_RESULTS_DIR="/path/to/results"
   export NEGOTIATION_MAX_RETRIES="5"

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

* Model names must be valid Hugging Face identifiers (e.g., "meta-llama/Llama-3.1-8B-Instruct")
* Game parameters must be within realistic ranges (BATNA values, resource constraints)
* Required configuration fields must be present
* Device specifications must match available hardware
* Temperature values must be appropriate for the sampling strategy

**Common Validation Errors**:

* ``ModelNotFoundError``: Invalid Hugging Face model identifier
* ``ConfigValidationError``: Missing required fields (batna_decay, utility_functions)
* ``InvalidParameterError``: Temperature out of range or invalid device specification
* ``GPUMemoryError``: Insufficient GPU memory for selected models

Best Practices
--------------

1. **Model Selection**:
   
   * Current setup uses state-of-the-art models: Llama-3.1-8B, Qwen2.5-7B, Llama-3.2-3B
   * Temperature 0.7 for creative negotiation, 0.0 for deterministic behavior
   * GPU memory optimization through ``device_map: "auto"``
   * Trust remote code enabled for modern model architectures

2. **Game Configuration**:
   
   * BATNA values optimized through empirical testing for realistic negotiations
   * Balanced decay rates (0.015) create appropriate time pressure
   * Acceptance training parameters encourage realistic negotiation behavior
   * Multi-issue weights reflect real-world team priorities

3. **Performance**:
   
   * Memory management and auto-unload enabled for efficient GPU usage
   * Short timeout (30s) prevents hanging sessions
   * Detailed metrics export for comprehensive analysis
   * Statistical significance testing at p < 0.05 level

4. **Research Configuration**:
   
   * 10 runs per comparison for statistical reliability
   * Bias detection through model position swapping
   * Comprehensive logging for research reproducibility
   * Uncertainty modeling in resource allocation scenarios