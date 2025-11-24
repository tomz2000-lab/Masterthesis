Games
=====

The Negotiation Platform includes several built-in negotiation games, each designed to test different aspects of negotiation behavior and strategy.

Available Games
---------------

1. **Company Car Negotiation** (``company_car``)
2. **Resource Allocation** (``resource_allocation``)  
3. **Integrative Negotiations** (``integrative_negotiations``)

Company Car Negotiation
-----------------------

**Game Type**: ``company_car``

A bilateral negotiation over the price of a company car between a buyer and seller.

**Scenario**:
- Company car worth €45,000 starting price
- Buyer has maximum budget of €42,000 (contextual information)
- Seller has minimum cost of €36,000 (contextual information)
- Both parties have BATNA (Best Alternative) options that decay over time
- Game mechanics driven by BATNA values, not budget/cost constraints

**Key Features**:
- Time pressure through BATNA decay
- Asymmetric information (each player knows only their own constraints)
- Utility calculations based on BATNA values only
- Randomized role assignment to eliminate role bias
- Acceptance training parameters to encourage realistic behavior
- JSON-based structured proposal system

**Configuration Parameters**:

.. code-block:: yaml

   company_car:
     starting_price: 45000      # Initial asking price
     buyer_budget: 42000        # Maximum buyer can afford (context only)
     seller_cost: 36000         # Seller's minimum cost (context only)
     buyer_batna: 41000         # Cost of buyer's alternative option
     seller_batna: 39000        # Seller's minimum if no deal
     rounds: 5                  # Maximum negotiation rounds
     batna_decay:
       buyer: 0.015             # 1.5% decay per round (balanced)
       seller: 0.015            # 1.5% decay per round (balanced)
     acceptance_training:
       profit_threshold: 0.10   # Accept within 10% of BATNA
       urgency_multiplier: 1.5  # Increased acceptance over time
       risk_aversion: 0.8       # Risk tolerance factor

**Mechanics**:
- Players can make offers, accept, or reject proposals
- Each player has limited number of proposals (rounds - 1)
- BATNA values decay each round, increasing time pressure
- Agreement reached when one player accepts the other's offer
- No agreement if all rounds expire without acceptance

**Winning Conditions**:
- Players win by achieving positive utility surplus over their BATNA
- Buyer surplus = BATNA - agreed_price (money saved)
- Seller surplus = agreed_price - BATNA (profit over minimum)

Resource Allocation
-------------------

**Game Type**: ``resource_allocation``

Multi-resource allocation negotiation between Development and Marketing teams.

**Scenario**:
- Limited GPU and CPU resources (100 total units) available for projects
- Development Team vs Marketing Team with different resource priorities
- Teams must negotiate optimal allocation under constraints
- Complex utility functions with uncertainty modeling
- Mathematical constraints ensure realistic resource distributions

**Configuration Parameters**:

.. code-block:: yaml

   resource_allocation:
     total_resources: 100           # Total resource pool
     constraints:
       gpu_bandwidth: 380           # 4x + 4y <= 380 constraint
       min_gpu: 5                   # Minimum GPU allocation
       min_cpu: 5                   # Minimum CPU allocation
     batnas:
       development: 300             # Development team BATNA
       marketing: 270               # Marketing team BATNA
     batna_decay:
       development: 0.015           # 1.5% decay per round
       marketing: 0.015             # 1.5% decay per round
     rounds: 5                      # Maximum negotiation rounds
     utility_functions:
       development:
         gpu_coefficient: 8         # 8x + 6y utility function
         cpu_coefficient: 6
         uncertainty_min: -2        # Uncertainty range
         uncertainty_max: 2
       marketing:
         gpu_coefficient: 6         # 6x + 8y utility function
         cpu_coefficient: 8
         uncertainty_min: -2        # Uncertainty range
         uncertainty_max: 2
     uncertainty:
       stochastic_demand:
         type: "normal"
         mean: 0
         std: 5
       market_volatility:
         type: "uniform"
         min: -0.08
         max: 0.08

**Mechanics**:
- Teams propose GPU/CPU allocations within constraints
- Utility calculated using linear functions with uncertainty
- Different team coefficients create integrative potential
- BATNA decay creates time pressure
- Mathematical validation ensures feasible proposals

**Key Features**:
- Two-dimensional resource negotiation (GPU x, CPU y)
- Asymmetric utility functions enable win-win solutions
- Constraint-based validation (bandwidth, minimums)
- Stochastic uncertainty modeling for realism
- BATNA-driven outcome evaluation

Integrative Negotiations
-----------------------

**Game Type**: ``integrative_negotiations``

Complex multi-issue office space negotiation between IT and Marketing teams.

**Scenario**:
- IT Team and Marketing Team negotiating office space and collaborative arrangements
- Four distinct issues with multiple options and point values
- Teams have asymmetric preferences enabling integrative solutions
- Opportunity for win-win solutions through strategic issue trading

**Configuration Parameters**:

.. code-block:: yaml

   integrative_negotiations:
     issues:
       server_room:
         options: [50, 100, 150]     # Square meters
         points: [10, 30, 60]       # Point values
       meeting_access:
         options: [2, 4, 7]          # Days per week
         points: [10, 30, 60]
       cleaning:
         options: ["IT", "Shared", "Outsourced"]
         points: [10, 30, 60]
       branding:
         options: ["Minimal", "Moderate", "Prominent"]
         points: [10, 30, 60]
     weights:
       IT:
         server_room: 0.4           # Server room critical for IT
         meeting_access: 0.1        # Low meeting priority
         cleaning: 0.3              # Moderate cleaning concern
         branding: 0.2              # Low branding priority
       Marketing:
         server_room: 0.1           # Low server room priority
         meeting_access: 0.4        # High meeting room needs
         cleaning: 0.2              # Moderate cleaning concern
         branding: 0.3              # High branding importance
     batnas:
       IT: 27                      # Optimized BATNA values
       Marketing: 19               # Reduced tie rates
     rounds: 5                      # Maximum negotiation rounds
     batna_decay: 0.015             # 1.5% decay per round

**Issues Negotiated**:
1. **Server Room Size**: 50, 100, or 150 square meters
2. **Meeting Room Access**: 2, 4, or 7 days per week
3. **Cleaning Responsibility**: IT handles, shared, or outsourced
4. **Branding Visibility**: Minimal, moderate, or prominent visibility

**Key Features**:
- Multiple interdependent issues
- Different team priorities enable integrative solutions
- Complex scoring based on multiple dimensions
- Requires sophisticated trade-off analysis

Game Implementation
-------------------

All games inherit from the ``BaseGame`` class and implement standardized interfaces.
For detailed implementation information, see :doc:`api/games`.

Custom Game Development
-----------------------

To create a new game, extend the base game class and implement the required methods.
See the :class:`negotiation_platform.games.base_game.BaseGame` API documentation for details.

Action Formats
--------------

Games use standardized JSON action formats:

**Company Car Actions**:

.. code-block:: json

   {
     "type": "offer",
     "price": 42000
   }
   
   {
     "type": "accept"
   }
   
   {
     "type": "reject"
   }

**Resource Allocation Actions**:

.. code-block:: json

   {
     "type": "propose",
     "gpu": 40,
     "cpu": 60
   }
   
   {
     "type": "accept"
   }

**Integrative Negotiation Actions**:

.. code-block:: json

   {
     "type": "propose",
     "server_room": 100,
     "meeting_access": 4,
     "cleaning": "Shared",
     "branding": "Moderate"
   }
   
   {
     "type": "accept"
   }

Bias Mitigation Features
-----------------------

All games implement bias reduction techniques:

1. **Randomized Role Assignment**: Player roles assigned randomly each game
2. **Neutral Role Labels**: Display "Role A/B" instead of "Buyer/Seller"
3. **Turn Order Randomization**: First-move advantage eliminated
4. **Balanced Configurations**: Parameters ensure fair negotiation zones

Metrics and Analysis
--------------------

Games support comprehensive metric calculation:

- **Feasibility**: How realistic/achievable outcomes are
- **Utility Surplus**: Player gains over their BATNA
- **Risk Minimization**: Conservative vs aggressive strategies  
- **Deadline Sensitivity**: Response to time pressure
- **Agreement Rate**: Frequency of successful negotiations
- **Pareto Efficiency**: How close to optimal mutual outcomes

Each game provides rich data for statistical analysis of negotiation behaviors and potential biases.