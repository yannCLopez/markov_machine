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

## Core Components

### `general_markov_machine.py`

The main analysis engine that provides:

- `GeneralMarkovGameAnalyzer`: Creates transition matrices, computes equilibrium payoffs, and analyzes deviation incentives
- `EquilibriumConstraintChecker`: Verifies if strategy profiles satisfy equilibrium constraints

### `numerical_checker.py`

Provides numerical methods to find parameter values that satisfy equilibrium constraints:

- `OptimizedNumericalChecker`: Implements grid search with vectorized operations for finding feasible parameter values
- `SearchConfig`: Configuration for numerical search parameters

## Auxiliary Files

- `point_checker_analytic.py`: Checks whether a specific parameter configuration forms an equilibrium using symbolic analysis
- `point_checker_numeric.py`: Numerically validates equilibrium constraints for a specific parameter configuration

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

## Mathematical Framework

The framework builds on finite state Markov chain theory to:
1. Represent games as transition matrices
2. Calculate expected payoffs using matrix inversion
3. Derive recursive formulations for continuation values
4. Check one-shot deviation incentives for equilibrium verification

Each game state is characterized by:
- Current position (e.g., "simple" or "complex")
- Player to move (Player 1 or Player 2)
- Transition probabilities to other states or terminal outcomes

## Example Game Structure

The framework is particularly suited for analyzing games where:
- Players alternate moves
- Each player has a set of possible actions in each position
- Actions probabilistically determine the next position
- There is always some probability of game termination

## Output

The framework produces:
- Symbolic expressions for equilibrium values
- Deviation incentive analysis
- Parameter constraints for equilibrium
- LaTeX documents with complete mathematical analysis

## Requirements

- Python 3.6+
- SymPy
- NumPy
- SciPy

## Getting Started

1. Define your game using the structure in the examples
2. Run analysis on specific strategy profiles
3. Use the constraint checker to find parameter ranges that support equilibrium

**Equivalence checks**: Mathematica scripts checking equivalence between expressions derived by hand, and expressions derived by the machine



