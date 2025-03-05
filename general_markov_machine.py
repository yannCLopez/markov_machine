"""
Payoffs seem to correspond with our derivation 
Matrix P appears to be correct.
Deviation seem to correspond with our by-hand derivation
TO-DO: CHECK deviation checker.
 
CHECK THESE: 
2. Constraint System (EquilibriumConstraintChecker class)
* Defines symbolic variables (b_sS, b_sC, b_cS, b_cC, tau) and their base constraints
* get_equilibrium_constraints method converts deviation gains into formal constraints
* Player 1's gains should be ≤ 0, Player 2's gains should be ≥ 0 for equilibrium
3. Solution Verification (ImprovedNumericalChecker class)
* Uses grid search to find parameter values satisfying all constraints
* infer_bounds method determines reasonable search ranges for parameters
* check_constraints_at_point verifies if a specific point satisfies all constraints
"""

import sympy as sp
from sympy import simplify
import os
from datetime import datetime
from sympy.logic.boolalg import And
import itertools
from typing import Dict, List, Tuple, Any
from sympy.core.relational import (
    Relational,
    StrictGreaterThan,
    StrictLessThan,
    GreaterThan,
    LessThan,
    Equality,
)
from numerical_checker import OptimizedNumericalChecker, SearchConfig
from typing import List, Dict, Tuple, Generator, Optional
from sympy.core.relational import Relational, StrictLessThan, StrictGreaterThan, LessThan, GreaterThan, Equality
import numpy as np
from scipy.optimize import brute
import itertools
from dataclasses import dataclass
import warnings


class GeneralMarkovGameAnalyzer:
    def __init__(
        self,
        num_positions,
        possible_moves,
        transition_probs,
        transition_probs_2,
        terminal_states,
        constant,
        position_names,
    ):
        """
        Initialize the analyzer for a general 2-player zero-sum game.

        Args:
            num_positions (int): Number of possible position states
            possible_moves (list): List of possible moves
            transition_probs (dict): Mapping (current_pos, move) -> {next_pos: prob}
                - current_pos is an integer 0 to num_positions-1
                - move is from possible_moves
                - next_pos is either an integer 0 to num_positions-1 or a terminal state name
                - probabilities should sum to 1 for each (current_pos, move) pair
            terminal_states (dict): Mapping terminal_state_name -> value from player 1's perspective
            constant (numeric):  # Constant to which player 1 and player 2's payoffs should add up. Used to compute player 2's payoffs as constant-(player 1's payoffs).
        """
        self.num_positions = num_positions
        self.possible_moves = possible_moves
        self.transition_probs = transition_probs
        self.transition_probs_2 = transition_probs_2
        self.terminal_states = terminal_states
        self.constant = constant
        self.position_names = position_names

        # Calculate total number of non-terminal states
        # For each position, we need states for both players
        self.num_non_terminal = 2 * num_positions

        # Total states includes terminal states
        self.total_states = self.num_non_terminal + len(terminal_states)

        # Create ordered list of terminal states for consistent indexing
        self.terminal_state_list = sorted(terminal_states.keys())
        self.terminal_values = [
            terminal_states[state] for state in self.terminal_state_list
        ]

    def create_transition_matrix(self, strategy_profile):
        """
        Create the transition matrix P for a given strategy profile.
        """
        # Initialize transition matrix
        P = sp.zeros(self.total_states, self.total_states)

        # Parse strategy profile
        p1_strategy, p2_strategy = self.parse_strategy_profile(strategy_profile)

        # Fill transitions for non-terminal states
        for player in [1, 2]:
            for pos in range(self.num_positions):
                # Calculate state index
                from_state = pos if player == 1 else pos + self.num_positions 

                # Get move from strategy
                strategy = p1_strategy if player == 1 else p2_strategy
                move = strategy[pos]

                # Get transition probabilities
                trans_probs = (
                    self.transition_probs if player == 1 else self.transition_probs_2
                )
                position_trans_probs = trans_probs[(pos, move)]

                # Fill matrix
                for next_state, prob in position_trans_probs.items():
                    if next_state in self.terminal_states:
                        # If terminal state, use its index in terminal_state_list
                        terminal_idx = self.terminal_state_list.index(next_state)
                        P[from_state, self.num_non_terminal + terminal_idx] = prob
                    else:
                        # For non-terminal transitions, adjust index based on next player
                        next_player = 2 if player == 1 else 1
                        to_state = (
                            next_state
                            if next_player == 1
                            else next_state + self.num_positions
                        )
                        P[from_state, to_state] = prob

        # Terminal states transition to themselves with probability 1
        for i in range(len(self.terminal_states)):
            P[self.num_non_terminal + i, self.num_non_terminal + i] = 1

        return P

    def parse_strategy_profile(self, strategy_profile):
        """
        Parse a strategy profile into a usable format.

        Args:
            strategy_profile (str): Format "p1_moves,p2_moves" where each player's moves
                                  is a string of length num_positions specifying their
                                  move choice in each position

        Returns:
            tuple: (player1_strategy, player2_strategy) where each strategy is a dict
                  mapping position -> move
        """
        p1_strat, p2_strat = strategy_profile.split(",")

        # Verify lengths match number of positions
        if len(p1_strat) != self.num_positions or len(p2_strat) != self.num_positions:
            raise ValueError("Strategy length must match number of positions")

        # Create strategy dictionaries
        p1_strategy = {pos: move for pos, move in enumerate(p1_strat)}
        p2_strategy = {pos: move for pos, move in enumerate(p2_strat)}

        return p1_strategy, p2_strategy

    def get_Q_R(self, P):
        """
        Extract Q and R matrices from transition matrix P.
        Q contains transitions between non-terminal states.
        R contains transitions to terminal states.

        Args:
            P: Full transition matrix

        Returns:
            tuple: (Q, R) matrices
        """
        Q = P[: self.num_non_terminal, : self.num_non_terminal]
        R = P[: self.num_non_terminal, self.num_non_terminal :]
        return Q, R
    
    def compute_equilibrium_payoffs_recursive(self, strategy_profile, constant):
        """
        Compute equilibrium continuation values using the recursive formulation:
        v = Qv + R[1,0]^T
        
        This gives values in terms of symbolic V_player^{pos,profile} variables
        rather than computing the explicit solution via matrix inversion.
        """
        # Get transition matrices
        P = self.create_transition_matrix(strategy_profile)
        Q, R = self.get_Q_R(P)
        
        # Create symbolic variables for each non-terminal state value
        value_symbols = []
        for player in [1, 2]:
            for pos in range(self.num_positions):
                pos_name = self.position_names[pos]
                symbol = sp.Symbol(f'V_{player}^{{{pos_name};{strategy_profile}}}')
                value_symbols.append(symbol)
        
        # Create the vector v of symbolic variables
        v = sp.Matrix(value_symbols)
        
        # Create terminal values vector
        
        # Right-hand side of equation: Qv + R[1,0]
        rhs = Q * v + R * sp.Matrix(self.terminal_values)
        
        # The equations v = Qv + R[1,0] give us our recursive formulation
        # Each component gives one equation
        equations = []
        for i in range(len(v)):
            equations.append(sp.Eq(v[i], rhs[i]))
            
        return equations, v

    def compute_equilibrium_payoffs(self, strategy_profile, constant):
        """
        Compute equilibrium continuation values for all states under a strategy profile.

        Args:
            strategy_profile (str): Strategy profile (e.g. "CS,CS")

        Returns:
            sympy Matrix: Vector of continuation values for all non-terminal states
        """
        # Get equilibrium transition matrix
        P = self.create_transition_matrix(strategy_profile)
        Q, R = self.get_Q_R(P)

        # Compute continuation values
        I = sp.eye(self.num_non_terminal)
        v_1 = (I - Q).inv() * R * sp.Matrix(self.terminal_values)
        v_1 = v_1.applyfunc(sp.simplify)

        
        v_2 = [constant - v_1[i, 0] for i in range(v_1.shape[0])]
        v_2 = v_2 = sp.Matrix(v_2)  # CHECK that this gives v_2 in the same form as v_1

        return v_1, v_2
    
    def compute_payoffs_both_methods(self, strategy_profile, constant):
        """
        Compute payoffs using both matrix inversion and recursive formulation.
        Returns both results for comparison and verification.
        """
        # Get matrix-based solution
        v_matrix, v2_matrix = self.compute_equilibrium_payoffs(strategy_profile, constant)
        
        # Get recursive formulation
        equations, v_symbols = self.compute_equilibrium_payoffs_recursive(strategy_profile, constant)
        
        # For verification: substitute the matrix solution into the recursive equations
        substitutions = {}
        for i, symbol in enumerate(v_symbols):
            substitutions[symbol] = v_matrix[i]
        
        # Check if equations are satisfied by matrix solution
        verification = []
        for eq in equations:
            lhs = eq.lhs.subs(substitutions)
            rhs = eq.rhs.subs(substitutions)
            verification.append(sp.simplify(lhs - rhs) == 0)
            
        return {
            'matrix_solution': (v_matrix),
            'recursive_equations': equations,
            'symbolic_variables': v_symbols,
            'verification': verification,
        }


    def compute_deviation_payoff_symbolic( #checked
        self, strategy_profile, player, pos, deviation_move
    ):
        """
        Compute expected payoff from one-shot deviation.
        Args:
            strategy_profile (str): Current strategy profile
            player (int): Player making deviation (1 or 2)
            pos (int): Position state where deviation occurs
            deviation_move (str): The deviating move
        Returns:
            sympy expression: Expected payoff for the deviating player
        """
        if player == 2:
            # First, player 2's deviation move
            trans_probs_2 = self.transition_probs_2
            position_trans_probs_2 = trans_probs_2[(pos, deviation_move)]
            
            # Initialize total payoff
            total_payoff = 0
            
            # Handle immediate outcomes from player 2's move
            if 'win' in position_trans_probs_2:
                total_payoff += position_trans_probs_2['win'] * 1  # Player 2 blunders (player 1 wins)
            if 'lose' in position_trans_probs_2:
                total_payoff += position_trans_probs_2['lose'] * 0  # Game ends, player 1 loses
                
            # For non-terminal transitions, compute player 1's response
            for next_pos, prob1 in position_trans_probs_2.items():
                if isinstance(next_pos, int):
                    # Get player 1's strategy in this position
                    p1_strategy = self.parse_strategy_profile(strategy_profile)[0]
                    p1_move = p1_strategy[next_pos]
                    
                    # Get player 1's transition probabilities
                    trans_probs_1 = self.transition_probs
                    position_trans_probs_1 = trans_probs_1[(next_pos, p1_move)]
                    
                    # Handle player 1's outcomes
                    if 'lose' in position_trans_probs_1:
                        total_payoff += prob1 * position_trans_probs_1['lose'] * 0  # Player 1 blunders
                    if 'win' in position_trans_probs_1:
                        total_payoff += prob1 * position_trans_probs_1['win'] * 1  # Game ends, player 1 wins
                        
                    # For non-terminal transitions after player 1's move
                    for final_pos, prob2 in position_trans_probs_1.items():
                        if isinstance(final_pos, int):
                            # Use V_2^{pos,strategy_profile} as we're returning to player 2's turn
                            v_symbol = sp.Symbol(f'V_2^{{{self.position_names[final_pos]},{strategy_profile}}}')
                            total_payoff += prob1 * prob2 * v_symbol

        else:  # player 1
            # Compute full cycle
            # First, player 1's deviation move
            trans_probs_1 = self.transition_probs
            position_trans_probs_1 = trans_probs_1[(pos, deviation_move)]
            
            # Initialize total payoff
            total_payoff = 0
            
            # Handle immediate outcomes from player 1's move
            # Hardcoded. Change accordingly
            # The payoffs should not be hardcoded, nor should the names of terminal states be hardcoded.
            if 'lose' in position_trans_probs_1:
                total_payoff += position_trans_probs_1['lose'] * 0  # Player 1 blunders
            if 'win' in position_trans_probs_1:
                total_payoff += position_trans_probs_1['win'] * 1   # Game ends, player 1 wins
                
            # For non-terminal transitions, compute player 2's response
            for next_pos, prob1 in position_trans_probs_1.items():
                if isinstance(next_pos, int):
                    # Get player 2's strategy in this position
                    p2_strategy = self.parse_strategy_profile(strategy_profile)[1] 
                    p2_move = p2_strategy[next_pos]
                    
                    # Get player 2's transition probabilities
                    trans_probs_2 = self.transition_probs_2
                    position_trans_probs_2 = trans_probs_2[(next_pos, p2_move)]
                    
                    # Handle player 2's outcomes
                    if 'win' in position_trans_probs_2:
                        total_payoff += prob1 * position_trans_probs_2['win'] * 1  # Player 2 blunders 
                    if 'lose' in position_trans_probs_2:
                        total_payoff += prob1 * position_trans_probs_2['lose'] * 0  # Game ends, player 1 loses
                        
                    # For non-terminal transitions after player 2's move
                    for final_pos, prob2 in position_trans_probs_2.items():
                        if isinstance(final_pos, int):
                            # Use V^{0,strategy_profile} or V^{1,strategy_profile} based on final_pos
                            v_symbol = sp.Symbol(f'V_1^{{{self.position_names[final_pos]},{strategy_profile}}}')
                            total_payoff += prob1 * prob2 * v_symbol
            
        return total_payoff.simplify()
       


    def compute_deviation_payoff(
        self, strategy_profile, player, pos, deviation_move
    ):  # checked
        """
        Compute expected payoff from one-shot deviation.

        Args:
            strategy_profile (str): Current strategy profile
            player (int): Player making deviation (1 or 2)
            pos (int): Position state where deviation occurs
            deviation_move (str): The deviating move

        Returns:
            sympy expression: Expected payoff for the deviating player
        """
        # Get transition probabilities for deviation move
        trans_probs = self.transition_probs if player == 1 else self.transition_probs_2
        position_trans_probs = trans_probs[(pos, deviation_move)]
        #print(f"CHECK_player: {player}")
        #print(f"CHECK_deviation_move: {deviation_move}")
        #print(f"CHECK_position_trans_probs: {position_trans_probs}")
        ##print(f"check_position: {pos}")

        # Initialize d vector for non-terminal states
        # Size is 2 * num_positions to account for both players' states
        d = [0] * (2 * self.num_positions)

        # Fill d based on who's deviating
        next_player = 2 if player == 1 else 1

        # For each non-terminal state transition in position_trans_probs
        for next_pos, prob in position_trans_probs.items():
            if isinstance(
                next_pos, int
            ):  # Only handle non-terminal transitions here (non terminal states are given as integers, and should start at 0)
                # Calculate index in d vector based on player and position
                # For player 1: indices 0 to num_positions-1
                # For player 2: indices num_positions to 2*num_positions-1
                if next_player == 1:
                    d[next_pos] = prob
                else:
                    d[self.num_positions + next_pos] = prob

        d = sp.Matrix(d)
        #print(f"CHECK_d: {d}")

        # Handle terminal states separately using terminal_state_list for consistency
        d_terminal = []
        for state in self.terminal_state_list:
            if state not in position_trans_probs:
                raise KeyError(f"State '{state}' not found in position_trans_probs")
            d_terminal.append(position_trans_probs[state])
        d_terminal = sp.Matrix(d_terminal)

        # Get terminal state values from the stored mapping
        v_terminal = sp.Matrix(
            [self.terminal_states[state] for state in self.terminal_state_list]
        )

        # Get equilibrium continuation values
        v, _ = self.compute_equilibrium_payoffs(strategy_profile, self.constant)
        #print(f"Check_v[2]: {v[2]}")

        # Debugging checks:

        # print(f'player = {player}')
        # print(f'v_terminal = {v_terminal}')
        # print(f'd_terminal = {d_terminal}')
        # print(f'v = {v}')
        # print(f'd = {d}')

        # Compute expected payoff
        # First term: expected payoff from non-terminal transitions
        non_terminal_payoff = sum(d[i] * v[i] for i in range(len(d)))

        # Second term: expected payoff from terminal transitions
        terminal_payoff = d_terminal.dot(v_terminal)

        total_payoff = non_terminal_payoff + terminal_payoff

        return total_payoff.simplify()

    def check_deviations(self, strategy_profile):  # checked
        """
        Systematically check all possible deviations from a strategy profile.

        Args:
            strategy_profile (str): Strategy profile to analyze (e.g., "CS,CS")

        Returns:
            dict: Mapping (player, position, move) -> (equilibrium_payoff, deviation_payoff)
        """
        # Parse strategy profile to get baseline strategies
        p1_strategy, p2_strategy = self.parse_strategy_profile(strategy_profile)
        # print(f"p2_strategy: {p2_strategy}")

        # Get equilibrium values for comparison
        equilibrium_values_p1, _ = self.compute_equilibrium_payoffs(
            strategy_profile, self.constant
        )

        # Initialize results dictionary
        deviation_analysis = {}

        # Check deviations for each player
        for player in [1, 2]:
            # Get current strategy and state offset for this player
            strategy = p1_strategy if player == 1 else p2_strategy
            state_offset = 0 if player == 1 else self.num_positions

            # Check each position
            for pos in range(self.num_positions):
                # Get equilibrium move and value
                equilibrium_move = strategy[pos]
                equilibrium_value = equilibrium_values_p1[pos + state_offset]  # here
            
                # Create symbolic equilibrium value
                pos_name = self.position_names[pos]
                symbolic_equilibrium_value = sp.Symbol(f'V_{player}^{{{pos_name};{strategy_profile}}}')


                # Check each possible deviation move
                for move in self.possible_moves:
                    if move != equilibrium_move:  # Only check actual deviations
                        # Compute deviation payoff
                        deviation_value = self.compute_deviation_payoff(
                            strategy_profile, player, pos, move
                        )
                        
                        symbolic_deviation_value = self.compute_deviation_payoff_symbolic(
                            strategy_profile, player, pos, move
                        )

                        gain_from_deviation = simplify(
                            deviation_value - equilibrium_value
                        )
                        
                        symbolic_gain_from_deviation = symbolic_deviation_value - symbolic_equilibrium_value
                              

                        equilibrium_value = equilibrium_value.simplify()

                        # Store results
                        deviation_analysis[(player, pos, move)] = {
                            "equilibrium_move": equilibrium_move,
                            "equilibrium_value": equilibrium_value,
                            "deviation_value": deviation_value,
                            "symbolic_equilibrium_value": symbolic_equilibrium_value,
                            "symbolic_gain_from_deviation": symbolic_gain_from_deviation,
                            "symbolic_deviation_value": symbolic_deviation_value,
                            "gain_from_deviation": gain_from_deviation,
                            
                        }

        return deviation_analysis


class EquilibriumConstraintChecker:
    def __init__(self, analyzer, assumptions, steps_per_var):
        """
        Initialize the constraint checker with a GeneralMarkovGameAnalyzer instance.

        Args:
            analyzer: Instance of GeneralMarkovGameAnalyzer
        """
        self.analyzer = analyzer
        # Define basic parameters
        self.b_sS, self.b_sC, self.b_cS, self.b_cC, self.tau = sp.symbols(
            "b_sS b_sC b_cS b_cC tau"
        )

        # Create variables list for numerical checker
        self.variables = [self.b_sS, self.b_sC, self.b_cS, self.b_cC, self.tau]

        # Create base constraints
        self.base_constraints = assumptions

        config = SearchConfig(epsilon=1e-5, early_stop_threshold=0.01, buffer=0.01)
        self.numerical_checker = OptimizedNumericalChecker(self.variables, self.base_constraints, config=config)

    def get_equilibrium_constraints(self, strategy_profile): #checked
        """
        Get all equilibrium constraints for a strategy profile.
        For player 1, gain from deviation should be <= 0
        For player 2, gain from deviation should be >= 0
        """
        deviations = self.analyzer.check_deviations(strategy_profile)
        constraints = []
        symbolic_constraints = []
        for (player, pos, move), results in deviations.items():
            gain_p1 = results["gain_from_deviation"]
            symbolic_gain = results["symbolic_gain_from_deviation"]
            if player == 1:
                constraints.append(gain_p1 <= 0)
                symbolic_constraints.append(symbolic_gain <= 0)
            else:  # player 2
                constraints.append(gain_p1 >= 0)
                symbolic_constraints.append(symbolic_gain >= 0)

        return constraints, symbolic_constraints

    def check_satisfiability(self, strategy_profile, steps_per_var, additional_constraints=None): #checked
        """
        Check if there exist parameter values satisfying all equilibrium constraints.
        Uses systematic grid search with bound inference.

        Args:
            strategy_profile (str): Strategy profile to analyze
            additional_constraints: Optional additional constraints to include

        Returns:
            dict: Results of satisfiability check including:
                - is_satisfiable: Boolean indicating if constraints can be satisfied
                - constraints_processed: The processed constraints for inspection
                - explanation: Human-readable explanation of results
                - sample_solution: If solution found, example values that work
        """
        # Get equilibrium constraints
        constraints, _ = self.get_equilibrium_constraints(strategy_profile)

        # Add any additional constraints
        if additional_constraints:
            constraints.extend(additional_constraints)

        # Try to find a feasible point
        found, point = self.numerical_checker.find_feasible_point(constraints, steps_per_var)

        if found:
            print("\nFeasible solution found:")
            for var, val in point.items():
                 print(f"{var} = {val:.6f}")
            return {
                "is_satisfiable": True,
                "constraints_processed": constraints,
                "explanation": "Found solution via grid search.",
                "sample_solution": point,
            }

        else:
            return {
                "is_satisfiable": None,
                "constraints_processed": constraints,
                "explanation": "No solution found via systematic search.",
                "sample_solution": None,
                }

    def analyze_profile(self, strategy_profile):
        """
        Comprehensive analysis of a strategy profile.

        Args:
            strategy_profile (str): Strategy profile to analyze

        Returns:
            dict: Complete analysis including:
                - equilibrium_values: Continuation values
                - constraints: All equilibrium constraints
                - satisfiability: Results of satisfiability check
        """
        # Get equilibrium values
        equilibrium_values, _ = self.analyzer.compute_equilibrium_payoffs(
            strategy_profile, constant
        )
        simplified_values = [value.simplify() for value in equilibrium_values]  

        # Get constraints
        constraints, symbolic_constraints = self.get_equilibrium_constraints(strategy_profile)

        # Check satisfiability
        satisfiability = self.check_satisfiability(strategy_profile, steps_per_var)

        # Return comprehensive analysis
        return {
            "equilibrium_values": simplified_values,
            "constraints": constraints,
            "symbolic_constraints": symbolic_constraints,
            "satisfiability": satisfiability,
        }

def save_analysis_to_tex(analyzer, strategy_profile, P, Q, R, deviations, analysis, timestamp):
    """
    Save complete analysis results to a LaTeX file.
    
    Args:
        analyzer: Instance of GeneralMarkovGameAnalyzer
        strategy_profile (str): The strategy profile being analyzed
        P, Q, R: Matrices from the analysis
        deviations: Dictionary of deviation results
        analysis: Dictionary containing complete analysis results
        timestamp: Timestamp string for filename
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, f"complete_analysis_{timestamp}.tex")
    
    tex_content = []
    
    # Document header
    tex_content.extend([
        "\\documentclass{article}",
        "\\usepackage{amsmath}",
        "\\usepackage{amssymb}",
        "\\usepackage{booktabs}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{etoolbox}",
        "\\allowdisplaybreaks",
        "\\setlength{\\parindent}{0pt}",  # Remove paragraph indentation
        "\\AtBeginEnvironment{align*}{\\setlength{\\abovedisplayskip}{0pt}}",
        "\\AtBeginEnvironment{align*}{\\setlength{\\belowdisplayskip}{0pt}}",
        "\\AtBeginEnvironment{align*}{\\setlength{\\abovedisplayshortskip}{0pt}}",
        "\\AtBeginEnvironment{align*}{\\setlength{\\belowdisplayshortskip}{0pt}}",
        "\\begin{document}",
        f"\\section*{{Complete Analysis of Strategy Profile {strategy_profile}}}",
        
        "\\subsection*{Transition Matrices}",
        "\\setlength{\\abovedisplayskip}{12pt}",
        "\\setlength{\\belowdisplayskip}{12pt}",
        f"\\text{{Transition matrix P:}} \\\\",
        "\\begin{align*}",
        f"&{sp.latex(P)} \\\\[2em]",
        "\\end{align*}",
        f"\\text{{Matrix Q:}} \\\\",
        "\\begin{align*}",
        f"&{sp.latex(Q)} \\\\[2em]",
        "\\end{align*}",
        f"\\text{{Matrix R:}} \\\\",
        "\\begin{align*}",
        f"&{sp.latex(R)}",
        "\\end{align*}",
        
        "\\subsection*{Deviation Analysis}"
    ])
    
    # Add deviation results
    for (player, pos, move), results in deviations.items():
        position_name = ["simple", "complex"][pos]
        pos_name = analyzer.position_names[pos]     # For symbolic notation
        tex_content.extend([
            f"\\paragraph{{Player {player}, {position_name} position, deviation to {move}}}\\", 
            f"Equilibrium move: \\\\",
            f"{results['equilibrium_move']} \\\\[1em]",
            f"Equilibrium value: \\\\",
            "\\begin{align*}",
            f"&V_{player}^{{{pos_name};{strategy_profile}}} = {sp.latex(results['equilibrium_value'])} \\\\[1em]",
            "\\end{align*}",
            f"Deviation value: \\\\",
            "\\begin{align*}",
            f"&{sp.latex(results['deviation_value'])} \\\\[1em]",
            "\\end{align*}",
            "\\begin{align*}",
            f"&={sp.latex(results['symbolic_deviation_value'])} \\\\[1em]",
            "\\end{align*}",
            f"Gain from deviation: \\\\",
            "\\begin{align*}",
            f"&{sp.latex(results['gain_from_deviation'])} \\\\[1em]",
            "\\end{align*}",
            "\\begin{align*}",
            f"&={sp.latex(results['symbolic_deviation_value'])}-V_{player}^{{{pos_name},{strategy_profile}}} \\\\[1em]",
            "\\end{align*}",
        ])
    
# Add equilibrium values
    tex_content.extend([
        "\\subsection*{Equilibrium Values (Non-terminal States)}"
        "\\subsubsection*{Closed Form Solution}"

    ])

    # Handle first num_positions entries (Player 1)
    for i in range(analyzer.num_positions):
        pos_name = analyzer.position_names[i]
        value = analysis['equilibrium_values'][i]
        tex_content.extend([
            f"$V_1^{{{pos_name};{strategy_profile}}}$: \\\\",
            "\\begin{align*}",
            f"&{sp.latex(value)} \\\\[1em]",
            "\\end{align*}"
        ])

    # Handle second num_positions entries (Player 2)
    for i in range(analyzer.num_positions):
        pos_name = analyzer.position_names[i]
        value = analysis['equilibrium_values'][i + analyzer.num_positions]
        tex_content.extend([
            f"$V_2^{{{pos_name};{strategy_profile}}}$: \\\\",
            "\\begin{align*}",
            f"&{sp.latex(value)} \\\\[1em]",
            "\\end{align*}"
        ])

            # Add recursive formulation
    tex_content.extend([
        "\\subsubsection*{Recursive Formulation}",
        "The continuation values satisfy the following system of equations: \\\\"
    ])

    # Get recursive equations from compute_equilibrium_payoffs_recursive
    equations, _ = analyzer.compute_equilibrium_payoffs_recursive(strategy_profile, constant)

        # Add equations to LaTeX
    tex_content.extend([
        "\\begin{align*}"
    ])
    
    for eq in equations:
        tex_content.extend([
            f"{sp.latex(eq)} \\\\[1em]"
        ])
    
    tex_content.extend([
        "\\end{align*}"
    ])

    # Add equilibrium constraints
    tex_content.extend([
        "\\subsection*{Equilibrium Constraints}"
    ])


    
    for i, (constraint, symbolic_constraint) in enumerate(zip(analysis['constraints'], analysis['symbolic_constraints']), 1):
        tex_content.extend([
            f"Constraint {i}: \\\\",
            "\\begin{align*}",
            f"&{sp.latex(constraint)} \\\\[1em]",
            "\\end{align*}",
            "\\begin{align*}",
            f"&{sp.latex(symbolic_constraint)} \\\\[1em]",
            "\\end{align*}"
        ])
    
    # Add satisfiability results
    tex_content.extend([
        "\\subsection*{Satisfiability Results}",
        f"{analysis['satisfiability']['explanation']}"
    ])
    
    # If solution found, add it
    if analysis['satisfiability'].get('sample_solution'):
        tex_content.extend([
            "\\paragraph{Sample Solution}",
            "\\begin{align*}"
        ])
        for var, val in analysis['satisfiability']['sample_solution'].items():
            tex_content.append(f"{var} &= {val:.6f} \\\\")
        tex_content.append("\\end{align*}")
    
    # Document footer
    tex_content.append("\\end{document}")
    
    # Write to file
    with open(filename, "w") as f:
        f.write("\n".join(tex_content))
    
    return filename

if __name__ == "__main__":

    # Create symbolic parameters with assumptions
    b_sS, b_sC, b_cS, b_cC, tau = sp.symbols(
        "b_sS b_sC b_cS b_cC tau",
    )    

    # Add additional assumptions about relative blunder probabilities using relational objects
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

    # Game setup
    num_positions = 2  # simple (0) and complex (1)
    possible_moves = ["S", "C"]
    constant = 1  # Constant to which player 1 and player 2's payoffs should add up. Used to compute player 2's payoffs as constant-(player 1's payoffs) 
    position_names = {0: 's', 1: 'c'} # simple (0) and complex (1)
    # Define transition probabilities
    # Format: (pos, move) -> {next_pos: prob, 'win': prob, 'lose': prob}
    # GUIDELINES: non terminal states are to be given as INTEGERS, STARTING AT 0. Terminal states should NOT be given as integers.
    # terminal states should be given as LETTERS/WORDS
    # IMPORTANT: Order positions in the same order as positions in the strategy profiles. E.g., is we have a strategy profile CC, \\
    # where the first letter represents the strategy when the game is in a simple pos. and the second letter represents the strategy when the game is a complex pos., \\
    # You should fill  transition_probs such that 0  denotes a simple pos., and 1 denotes a complex pos.
    transition_probs = {
        # From simple position (0)
        (0, "S"): {
            "win": (1 - b_sS) * tau,  # no blunder and game ends
            "lose": b_sS,  # blunder
            0: (1 - b_sS) * (1 - tau),  # no blunder, continue to simple
            1: 0,  # no transitions to complex with simple move
        },
        (0, "C"): {
            "win": (1 - b_sC) * tau,
            "lose": b_sC,
            0: 0,  # no transitions to simple with complex move
            1: (1 - b_sC) * (1 - tau),  # no blunder, continue to complex
        },
        # From complex position (1)
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

    # Transition probabilities when player 2 moves.
    # Format: (pos, move) -> {next_pos: prob, 'win': prob, 'lose': prob}
    # GUIDELINES: non terminal states are to be given as INTEGERS, STARTING AT 0. Terminal states should NOT be given as integers.
    # terminal states should be given as LETTERS/WORDS
    # REMEMBER: Even here, payoffs are from player 1's perspective
    transition_probs_2 = {
        # From simple position (0)
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
        # From complex position (1)
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

    # Define terminal states and their values from player 1's perspective
    # terminal states should be given as LETTERS/WORDS
    terminal_states = {"win": 1, "lose": 0}

    # Create analyzer instance with terminal states
    analyzer = GeneralMarkovGameAnalyzer(
        num_positions,
        possible_moves,
        transition_probs,  # Player 1's transition probabilities
        transition_probs_2,  # Player 2's transition probabilities
        terminal_states,
        constant,
        position_names,
    )
    # Analyze strategy profile
    # Note: in our encoding, position 0 = simple, 1 = complex
    all_strategy_profiles = ("CS,SS","CC,SS", "CC,CS", "CS,CS",)

    for strat_profile in all_strategy_profiles:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        strategy_profile = strat_profile

        steps_per_var = 2 #grid density per variable

        # Get transition matrix
        P = analyzer.create_transition_matrix(strategy_profile)

        # Get Q and R matrices for value computation
        Q, R = analyzer.get_Q_R(P)

        print("Transition matrix P:")
        print(P)
        print("\nQ matrix:")
        print(Q)
        print("\nR matrix:")
        print(R)

        deviations = analyzer.check_deviations(strategy_profile)

        # Print results
        for (player, pos, move), results in deviations.items():
            print(f"\nPlayer {player} at position {pos} deviating to {move}:")
            print(f"Equilibrium move was: {results['equilibrium_move']}")
            print(f"Equilibrium value: {results['symbolic_equilibrium_value']}={results['equilibrium_value'].simplify()}")
            print(f"Symbolic deviation value: {results['symbolic_deviation_value'].simplify()}")           
            print(f"Deviation value: {results['deviation_value'].simplify()}")
            print(f"Gain from deviation: {results['gain_from_deviation'].simplify()}")
            print(f"= {results['symbolic_gain_from_deviation']}")

        # Create constraint checker
        checker = EquilibriumConstraintChecker(analyzer, assumptions, steps_per_var)

        # Analyze a strategy profile
        analysis = checker.analyze_profile(strategy_profile)

        # Print results
        print(f"\nAnalysis of strategy profile {strategy_profile}:")
        print("\nEquilibrium values (non-terminal states):")
        print(analysis["equilibrium_values"])

        print("\nEquilibrium constraints:")
        for i, (constraint, symbolic_constraint) in enumerate(zip(analysis["constraints"], analysis["symbolic_constraints"]), 1):
            print(f"\nConstraint {i}:")
            print(constraint)
            print(f"= {symbolic_constraint}")


        print("\nSatisfiability results:")
        print(analysis["satisfiability"]["explanation"])

        # Add the new code here
        if analysis["satisfiability"]["is_satisfiable"]:
            results_filename = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"valid_parameters_{strategy_profile.replace(',', '_')}_{timestamp}.csv"
            )
            
            satisfaction_percentage = checker.numerical_checker.save_results_to_csv(results_filename)
            
            print(f"\nResults saved to: {results_filename}")
            print(f"Satisfaction percentage: {satisfaction_percentage:.2f}%")
            print(f"Total valid combinations found: {len(checker.numerical_checker.valid_points)}")

            # At the end of each iteration, save complete analysis
        output_file = save_analysis_to_tex(
            analyzer,
            strategy_profile,
            P,
            Q,
            R,
            deviations,
            analysis,
            timestamp,
        )
        print(f"\nComplete analysis has been saved to '{output_file}'")

        results = analyzer.compute_payoffs_both_methods(strategy_profile, constant)

        # Print recursive equations
        print("Recursive formulation of continuation values:")
        for eq in results['recursive_equations']:
            print(eq)

        # Verify that matrix solution satisfies recursive equations
        print("\nVerification that matrix solution satisfies recursive equations:")
        for verified in results['verification']:
            print(verified)
