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
- Buyer has maximum budget of €40,000
- Seller has minimum cost of €38,000
- Both parties have BATNA (Best Alternative) options that decay over time

**Key Features**:
- Time pressure through BATNA decay
- Asymmetric information (each player knows only their own constraints)
- Clear win/lose outcomes based on final price
- Randomized role assignment to eliminate role bias

**Configuration Parameters**:

.. code-block:: yaml

   company_car:
     starting_price: 45000      # Initial asking price
     buyer_budget: 40000        # Maximum buyer can afford
     seller_cost: 38000         # Seller's minimum acceptable price
     buyer_batna: 44000         # Cost of buyer's alternative option
     seller_batna: 39000        # Seller's minimum if no deal
     rounds: 5                  # Maximum negotiation rounds
     batna_decay:
       buyer: 0.05             # 5% decay per round
       seller: 0.03            # 3% decay per round

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

Multi-issue negotiation between development teams over limited computing resources.

**Scenario**:
- Limited GPU and CPU hours available for projects
- Development Team vs Marketing Team with different priorities
- Teams must negotiate fair allocation of resources
- Each team values resources differently based on their project needs

**Configuration Parameters**:

.. code-block:: yaml

   resource_allocation:
     total_gpu_hours: 80              # Available GPU compute time
     total_cpu_hours: 160             # Available CPU compute time  
     deadline_pressure: 0.1           # Urgency factor
     dev_team_priorities:
       gpu_weight: 0.8               # High GPU importance
       cpu_weight: 0.2               # Lower CPU importance
     marketing_team_priorities:
       gpu_weight: 0.3               # Lower GPU importance  
       cpu_weight: 0.7               # High CPU importance

**Mechanics**:
- Teams propose resource allocations (GPU hours, CPU hours)
- Proposals must not exceed total available resources
- Each team has different valuation weights for resources
- Success measured by how well final allocation matches preferences

**Key Features**:
- Multi-dimensional negotiation (2+ resources)
- Integrative potential (teams value resources differently)
- Zero-sum constraint (total resources are fixed)
- Deadline pressure affects urgency

Integrative Negotiations
-----------------------

**Game Type**: ``integrative_negotiations``

Complex multi-issue business negotiation with potential for mutual gains.

**Scenario**:
- IT Team and Marketing Team negotiating project parameters
- Multiple issues: budget, timeline, quality requirements, features
- Teams have different priorities across issues
- Opportunity for win-win solutions through trade-offs

**Configuration Parameters**:

.. code-block:: yaml

   integrative_negotiations:
     total_budget: 100000             # Project budget constraint
     project_duration_weeks: 12       # Timeline constraint
     quality_importance:
       buyer: 0.6                    # Quality priority for buyer
       seller: 0.4                   # Quality priority for seller
     timeline_flexibility:
       buyer: 0.3                    # Timeline flexibility for buyer
       seller: 0.8                   # Timeline flexibility for seller

**Issues Negotiated**:
1. **Budget allocation** across project components
2. **Timeline** and milestone dates
3. **Quality standards** and testing requirements  
4. **Feature scope** and deliverables

**Key Features**:
- Multiple interdependent issues
- Different team priorities enable integrative solutions
- Complex scoring based on multiple dimensions
- Requires sophisticated trade-off analysis

Game Implementation
-------------------

All games inherit from the ``BaseGame`` class and implement:

.. code-block:: python

   class BaseGame:
       def initialize_game(self, players: List[str]) -> Dict[str, Any]:
           """Set up initial game state."""
           
       def is_valid_action(self, player: str, action: Dict, state: Dict) -> bool:
           """Validate player actions."""
           
       def process_actions(self, actions: Dict, state: Dict) -> Dict[str, Any]:
           """Process all player actions and update state."""
           
       def is_game_over(self, state: Dict[str, Any]) -> bool:
           """Check if game should end."""
           
       def get_winner(self, state: Dict[str, Any]) -> Optional[str]:
           """Determine winner if applicable."""
           
       def get_game_prompt(self, player_id: str) -> str:
           """Generate player-specific prompt."""

Custom Game Development
-----------------------

To create a new game:

1. **Create game class**:

.. code-block:: python

   from negotiation_platform.games.base_game import BaseGame
   
   class MyCustomGame(BaseGame):
       def __init__(self, config: Dict[str, Any]):
           super().__init__(game_id="my_game", config=config)
           # Initialize game-specific parameters
           
       def initialize_game(self, players: List[str]) -> Dict[str, Any]:
           # Set up initial state
           return {
               "players": players,
               "current_round": 1,
               "game_specific_data": {}
           }

2. **Register game in GameEngine**:

.. code-block:: python

   from negotiation_platform.core.game_engine import GameEngine
   
   engine = GameEngine()
   engine.register_game_type("my_game", MyCustomGame)

3. **Add configuration**:

.. code-block:: yaml

   games:
     my_game:
       parameter1: value1
       parameter2: value2

Action Formats
--------------

Games use standardized action formats:

**Offer Actions**:

.. code-block:: json

   {
     "type": "offer",
     "price": 42000
   }

**Resource Allocation Actions**:

.. code-block:: json

   {
     "type": "propose",
     "allocation": {
       "gpu_hours": 40,
       "cpu_hours": 80
     }
   }

**Response Actions**:

.. code-block:: json

   {
     "type": "accept"
   }
   
   {
     "type": "reject"
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