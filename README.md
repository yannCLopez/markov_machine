# Markov Game Analysis Framework

## Overview

This project implements a framework for analyzing Markov games with alternating moves, particularly focused on verifying equilibrium properties. The framework handles symbolic computation to derive equilibrium values and check incentive compatibility constraints.

## Features

- Analyze 2-player, zero-sum Markov games with alternating moves
- Support for games with:
  - Multiple position states
  - Arbitrary move sets
  - Probabilistic transitions
  - Terminal states with defined payoffs
- Symbolic computation using SymPy to:
  - Calculate equilibrium continuation values
  - Verify deviation incentives
  - Check equilibrium constraints
- Numerical validation of equilibrium conditions

## Repository Structure

- **Core Analysis Engine**:
  - `general_markov_machine.py`: Main engine that creates transition matrices, computes payoffs, and analyzes deviation incentives
  - `numerical_checker.py`: Provides methods to find parameter values satisfying equilibrium constraints

- **Validation Tools**:
  - `point_checker_analytic.py`: Checks whether specific parameter configurations form equilibria using symbolic analysis
  - `point_checker_numeric.py`: Numerically validates equilibrium constraints

- **Equivalence Checks**:
  - `Equivalence checks/`: Mathematica scripts that verify equivalence between expressions derived by hand and expressions derived by the machine. Specific to a particular game -- need to be adapted to your context of interest.
  
- **Game-Specific Implementation**:
  - `Pandolfini_specifics/`: Specific implementation details and documentation for a particular game
- **Legacy Code**:
  - `archived/`: Previous versions of the framework

## Core Components

### `GeneralMarkovGameAnalyzer` Class

The main engine that provides:

- Creation of transition matrices from game specification
- Computation of equilibrium payoffs using matrix inversion
- Analysis of deviation incentives
- Recursive formulation of continuation values

### `EquilibriumConstraintChecker` Class

Verifies if strategy profiles satisfy equilibrium constraints by:

- Converting deviation gains into formal constraints
- Checking if there exist parameter values satisfying all constraints
- Generating comprehensive equilibrium analysis

### `OptimizedNumericalChecker` Class

Finds parameter values satisfying equilibrium constraints through:

- Grid search with vectorized operations
- Automatic bound inference
- Numerically stable constraint evaluation

## Mathematical Framework

The analysis framework builds on finite state Markov chain theory:

1. Representing games as transition matrices
2. Calculating expected payoffs using matrix inversion: V = (I-Q)^(-1)R
3. Deriving recursive formulations for continuation values: v = Qv + Rc
4. Checking one-shot deviation incentives

Each game state is characterized by:
- Current position
- Player to move
- Transition probabilities to other states or terminal outcomes

## Usage

### Basic Analysis

```python
from general_markov_machine import GeneralMarkovGameAnalyzer, EquilibriumConstraintChecker
from numerical_checker import OptimizedNumericalChecker, SearchConfig

# 1. Define game parameters
num_positions = 2  # Number of position states
possible_moves = ["S", "C"]  # Possible moves
position_names = {0: 's', 1: 'c'}  # Names for states

# 2. Define symbolic parameters
import sympy as sp
b_sS, b_sC, b_cS, b_cC, tau = sp.symbols("b_sS b_sC b_cS b_cC tau")

# 3. Define transition probabilities
transition_probs = {
    # Player 1's transition probabilities
    # Format: (position, move) -> {next_state: probability, ...}
}
transition_probs_2 = {
    # Player 2's transition probabilities
}

# 4. Define terminal states and their values
terminal_states = {"win": 1, "lose": 0}

# 5. Create analyzer
analyzer = GeneralMarkovGameAnalyzer(
    num_positions,
    possible_moves,
    transition_probs,
    transition_probs_2,
    terminal_states,
    constant=1,  # Zero-sum value
    position_names=position_names
)

# 6. Analyze a strategy profile
strategy_profile = "CS,CS"  # Format: "p1_moves,p2_moves"
P = analyzer.create_transition_matrix(strategy_profile)
equilibrium_values, _ = analyzer.compute_equilibrium_payoffs(strategy_profile, constant=1)
deviations = analyzer.check_deviations(strategy_profile)
```

### Checking Equilibrium Constraints

```python
# Define parameter constraints
assumptions = [
    b_sC > b_sS,
    b_cC > b_cS,
    # Additional constraints...
]

# Create constraint checker
checker = EquilibriumConstraintChecker(analyzer, assumptions, steps_per_var=10)

# Analyze profile
analysis = checker.analyze_profile(strategy_profile)

# Check if constraints are satisfiable
if analysis["satisfiability"]["is_satisfiable"]:
    print("Strategy profile is a potential equilibrium")
    print("Sample solution:", analysis["satisfiability"]["sample_solution"])
else:
    print("Strategy profile cannot be an equilibrium")
```

## Output

The framework produces:
- Symbolic expressions for equilibrium values
- Deviation incentive analysis
- Parameter constraints for equilibrium
- LaTeX documents with complete mathematical analysis
- CSV files with valid parameter combinations

## Requirements

- Python 3.6+
- SymPy
- NumPy
- SciPy
