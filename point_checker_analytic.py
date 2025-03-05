import sympy as sp
from general_markov_machine import GeneralMarkovGameAnalyzer, EquilibriumConstraintChecker

def check_point_for_equilibrium():
    # Create symbolic parameters
    b_sS, b_sC, b_cS, b_cC, tau = sp.symbols("b_sS b_sC b_cS b_cC tau")

    # Define the point to check
    point = {
        b_sS: sp.Rational(59, 512),
        b_sC: sp.Rational(15, 128),
        b_cS: sp.Rational(1, 8),
        b_cC: sp.Rational(3, 16),
        tau: sp.Rational(1, 2)
    }

    # Game setup
    num_positions = 2
    possible_moves = ["S", "C"]
    CONSTANT = 1
    position_names = {0: 's', 1: 'c'}

    # Define transition probabilities
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

    # Create analyzer instance
    analyzer = GeneralMarkovGameAnalyzer(
        num_positions,
        possible_moves,
        transition_probs,
        transition_probs_2,
        terminal_states,
        CONSTANT,
        position_names,
    )

    # Check deviations for CS,CS
    strategy_profile = "CS,CS"
    
    # Get equilibrium values
    equilibrium_values, _ = analyzer.compute_equilibrium_payoffs(strategy_profile, CONSTANT)
    
    # Get deviations
    P = analyzer.create_transition_matrix(strategy_profile)
    Q, R = analyzer.get_Q_R(P)
    deviations = analyzer.check_deviations(strategy_profile)

    print(f"\nChecking point for strategy profile {strategy_profile}:")
    print("\nParameters:")
    for var, val in point.items():
        print(f"{var} = {val}")

    print("\nDeviation analysis:")
    for (player, pos, move), results in deviations.items():
        # Substitute the point values into the gain expression
        gain = results['gain_from_deviation'].subs(point)
        print(f"\nPlayer {player} at position {pos} deviating to {move}:")
        print(f"Gain from deviation: {gain}")
        
        # Check if this deviation violates equilibrium
        is_profitable = (gain > 0 if player == 1 else gain < 0)
        if is_profitable:
            print(f"WARNING: Profitable deviation found!")

    # Overall check
    all_constraints_satisfied = all(
        results['gain_from_deviation'].subs(point) <= 0 if player == 1 else results['gain_from_deviation'].subs(point) >= 0
        for (player, pos, move), results in deviations.items()
    )

    print("\nOverall result:")
    if all_constraints_satisfied:
        print("✓ This point is a valid equilibrium for CS,CS")
    else:
        print("✗ This point is NOT a valid equilibrium for CS,CS")

if __name__ == "__main__":
    check_point_for_equilibrium()