import sympy as sp
from general_markov_machine import GeneralMarkovGameAnalyzer, EquilibriumConstraintChecker
from numerical_checker import OptimizedNumericalChecker, SearchConfig

def check_point_numerically():
    # Create symbolic parameters
    b_sS, b_sC, b_cS, b_cC, tau = sp.symbols("b_sS b_sC b_cS b_cC tau")

    # Define the point to check
    point = {
        b_sS: 59/512,
        b_sC: 15/128,
        b_cS: 1/8,
        b_cC: 3/16,
        tau: 1/2
    }

    # Game setup
    num_positions = 2
    possible_moves = ["S", "C"]
    CONSTANT = 1
    position_names = {0: 's', 1: 'c'}

    # Define base constraints - same as in the original code
    assumptions = [
        b_sC > b_sS,
        b_cC > b_cS,
        b_sS < 1,
        b_sC < 1,
        b_cS < 1,
        b_cC < 0.25,
        tau < 1,
        b_sS > 0,
        b_sC > 0,
        b_cS > 0,
        b_cC > 0,
        tau > 0,
    ]

    # Set up the game analyzer
    transition_probs = {
        (0, "S"): {
            "win": (1 - b_sS) * tau,
            "lose": b_sS,
            0: (1 - b_sS) * (1 - tau),
            1: 0,
        },
        (0, "C"): {
            "win": (1 - b_sC) * tau,
            "lose": b_sC,
            0: 0,
            1: (1 - b_sC) * (1 - tau),
        },
        (1, "S"): {
            "win": (1 - b_cS) * tau,
            "lose": b_cS,
            0: (1 - b_cS) * (1 - tau),
            1: 0,
        },
        (1, "C"): {
            "win": (1 - b_cC) * tau,
            "lose": b_cC,
            0: 0,
            1: (1 - b_cC) * (1 - tau),
        },
    }

    transition_probs_2 = {
        (0, "S"): {
            "win": b_sS + (1 - b_sS) * tau,
            "lose": 0,
            0: (1 - b_sS) * (1 - tau),
            1: 0,
        },
        (0, "C"): {
            "win": b_sC + (1 - b_sC) * tau,
            "lose": 0,
            0: 0,
            1: (1 - b_sC) * (1 - tau),
        },
        (1, "S"): {
            "win": b_cS + (1 - b_cS) * tau,
            "lose": 0,
            0: (1 - b_cS) * (1 - tau),
            1: 0,
        },
        (1, "C"): {
            "win": b_cC + (1 - b_cC) * tau,
            "lose": 0,
            0: 0,
            1: (1 - b_cC) * (1 - tau),
        },
    }

    terminal_states = {"win": 1, "lose": 0}

    # Create analyzer and constraint checker
    analyzer = GeneralMarkovGameAnalyzer(
        num_positions,
        possible_moves,
        transition_probs,
        transition_probs_2,
        terminal_states,
        CONSTANT,
        position_names,
    )

    # Create constraint checker with appropriate configuration
    config = SearchConfig(epsilon=1e-5, early_stop_threshold=0.01, buffer=0.01)
    checker = EquilibriumConstraintChecker(analyzer, assumptions, steps_per_var=2)

    # Get equilibrium constraints for CS,CS
    strategy_profile = "CS,CS"
    constraints, _ = checker.get_equilibrium_constraints(strategy_profile)

    # Combine base constraints and equilibrium constraints
    all_constraints = assumptions + constraints

    # Create numerical checker instance
    variables = [b_sS, b_sC, b_cS, b_cC, tau]
    numerical_checker = OptimizedNumericalChecker(variables, assumptions, config=config)

    # Check the specific point
    print(f"\nChecking point for strategy profile {strategy_profile}:")
    print("\nParameters:")
    for var, val in point.items():
        print(f"{var} = {val}")

    # Check constraints
    satisfies_constraints = numerical_checker.check_constraints_at_point(point, all_constraints)

    print("\nConstraint Verification:")
    
    # Check each constraint individually for detailed output
    for i, constraint in enumerate(all_constraints, 1):
        # Substitute values into constraint
        lhs_val = float(constraint.lhs.subs(point).evalf())
        rhs_val = float(constraint.rhs.subs(point).evalf())
        
        # Evaluate constraint
        satisfied = True
        if isinstance(constraint, sp.StrictGreaterThan):
            satisfied = lhs_val > rhs_val
        elif isinstance(constraint, sp.GreaterThan):
            satisfied = lhs_val >= rhs_val
        elif isinstance(constraint, sp.StrictLessThan):
            satisfied = lhs_val < rhs_val
        elif isinstance(constraint, sp.LessThan):
            satisfied = lhs_val <= rhs_val
        
        print(f"\nConstraint {i}: {constraint}")
        print(f"LHS value: {lhs_val}")
        print(f"RHS value: {rhs_val}")
        print(f"Satisfied: {'✓' if satisfied else '✗'}")

    print("\nOverall result:")
    if satisfies_constraints:
        print("✓ This point satisfies all constraints for CS,CS")
    else:
        print("✗ This point does NOT satisfy all constraints for CS,CS")

if __name__ == "__main__":
    check_point_numerically()