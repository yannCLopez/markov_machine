import sympy as sp
import os
from datetime import datetime


class MarkovGameAnalyzer:
    def __init__(self, transition_params):
        """
        Initialize with game-specific transition parameters.
        
        Args:
            transition_params: dictionary containing game-specific transition parameters
                             (for our specific game, this includes blunder probs and tau)
        """
        self.params = transition_params
        
    def get_transition_prob(self, from_state, move, params):
        """
        Calculate transition probabilities for a given state and move.
        To be implemented by specific games.
        
        Args:
            from_state: current state
            move: chosen move
            params: game parameters
            
        Returns:
            Dictionary mapping destination states to probabilities
        """
        raise NotImplementedError("Must be implemented by specific game class")

    def create_transition_matrix(self, strategy_profile, num_states):
        """
        Creates transition matrix P for a given strategy profile.
        
        Args:
            strategy_profile: string or other format specifying strategy
            num_states: total number of states including terminal states
            
        Returns:
            sympy Matrix: Transition matrix P
        """
        raise NotImplementedError("Must be implemented by specific game class")

    def get_Q_R(self, P, num_non_terminal):
        """
        Extracts Q and R matrices from transition matrix P.
        
        Args:
            P: full transition matrix
            num_non_terminal: number of non-terminal states
            
        Returns:
            Q: transitions between non-terminal states
            R: transitions to terminal states
        """
        Q = P[:num_non_terminal, :num_non_terminal]
        R = P[:num_non_terminal, num_non_terminal:]
        return Q, R

class TwoPlayerAlternatingGameAnalyzer(MarkovGameAnalyzer):
    def __init__(self, transition_params):
        """
        Specific implementation for our two-player alternating move game.
        
        Args:
            transition_params: dict with 'b_sS', 'b_sC', 'b_cS', 'b_cC', 'tau'
        """
        super().__init__(transition_params)
        
    def get_transition_prob(self, state_info, move, params):
        """
        Calculate transition probabilities for our specific game.
        
        Args:
            state_info: tuple (player, position) where player is 1/2 and position is 's'/'c'
            move: 'S' or 'C'
            params: game parameters including blunder probabilities and tau
            
        Returns:
            dict mapping destination states to probabilities
        """
        player, pos = state_info
        # Get blunder probability based on position and move
        if pos == 's':
            b = params['b_sS'] if move == 'S' else params['b_sC']
        else:
            b = params['b_cS'] if move == 'S' else params['b_cC']
            
        # Calculate transition probabilities
        result = {}
        no_blunder = 1 - b
        
        if player == 1:
            result['lose'] = b  # blunder leads to loss
            result['win'] = no_blunder * params['tau']  # game ends naturally with win
            # Continue to next state
            next_pos = 's' if move == 'S' else 'c'
            result[f'p2{next_pos}'] = no_blunder * (1 - params['tau'])
        else:  # player 2
            result['win'] = b + no_blunder * params['tau']  # blunder or natural end
            # Continue to next state
            next_pos = 's' if move == 'S' else 'c'
            result[f'p1{next_pos}'] = no_blunder * (1 - params['tau'])
            
        return result

    def create_transition_matrix(self, strategy_profile):
        """
        Create transition matrix for our specific game.
        
        Args:
            strategy_profile: string "XY,ZW" where:
                XY = player 1's responses to (simple, complex)
                ZW = player 2's responses to (simple, complex)
                Each of X,Y,Z,W is either 'S' or 'C'
        """
        # Parse strategy profile
        p1_strat, p2_strat = strategy_profile.split(',')
        p1_responses = {'s': p1_strat[0], 'c': p1_strat[1]}
        p2_responses = {'s': p2_strat[0], 'c': p2_strat[1]}
        
        # Create 6x6 matrix (4 non-terminal states + 2 terminal states)
        P = sp.zeros(6, 6)
        
        # State mapping for matrix indices
        state_to_idx = {'p1s': 0, 'p1c': 1, 'p2s': 2, 'p2c': 3, 'win': 4, 'lose': 5}
        
        # Fill transitions for non-terminal states
        for player in [1, 2]:
            for pos in ['s', 'c']:
                from_state = f'p{player}{pos}'
                from_idx = state_to_idx[from_state]
                
                # Get move from strategy
                if player == 1:
                    move = p1_responses[pos]
                else:
                    move = p2_responses[pos]
                
                # Get transition probabilities
                trans_probs = self.get_transition_prob((player, pos), move, self.params)
                
                # Fill matrix
                for to_state, prob in trans_probs.items():
                    to_idx = state_to_idx[to_state]
                    P[from_idx, to_idx] = prob
        
        # Terminal states
        P[4, 4] = 1  # win stays win
        P[5, 5] = 1  # lose stays lose
        
        return P
    

def matrix_to_latex(P, strategy_profile):
    """
    Convert sympy matrix P to LaTeX file with nicely formatted output.
    Save in the same directory as the script with timestamp and strategy profile in filename.
    """
    # Generate timestamp and filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"transition_matrix_{strategy_profile.replace(',', '_')}_{timestamp}.tex"
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create full path for output file
    output_path = os.path.join(script_dir, filename)
    
    # LaTeX document preamble
    latex_str = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage[margin=1in]{{geometry}}
\\begin{{document}}

\\section*{{Transition Matrix P for Strategy Profile {strategy_profile}}}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

State ordering: 
\\begin{{itemize}}
\\item States 0-3: Non-terminal states (p1s, p1c, p2s, p2c)
\\item States 4-5: Terminal states (win, lose)
\\end{{itemize}}

\\[
P = \\begin{{pmatrix}}
"""
    
    # Add matrix entries
    rows, cols = P.shape
    for i in range(rows):
        row = [str(P[i,j]) for j in range(cols)]
        latex_str += " & ".join(row)
        if i < rows-1:
            latex_str += "\\\\"
        latex_str += "\n"
    
    # Close matrix and document
    latex_str += """\\end{pmatrix}
\\]

\\end{document}"""
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"LaTeX file saved to: {output_path}")

# Create symbolic parameters
b_sS = sp.Symbol('b_sS')
b_sC = sp.Symbol('b_sC')
b_cS = sp.Symbol('b_cS')
b_cC = sp.Symbol('b_cC')
tau = sp.Symbol('tau')

# Package parameters in dictionary
params = {
    'b_sS': b_sS,
    'b_sC': b_sC,
    'b_cS': b_cS,
    'b_cC': b_cC,
    'tau': tau
}

# Create analyzer instance
analyzer = TwoPlayerAlternatingGameAnalyzer(params)
strategy_profile = "CS,CS"

# Get transition matrix for CC,SS
P = analyzer.create_transition_matrix(strategy_profile)

# Get Q and R matrices
Q, R = analyzer.get_Q_R(P, 4)  # 4 non-terminal states

matrix_to_latex(P, strategy_profile)


# Print results
print("P =")
print(P)
print("\nQ =")
print(Q)
print("\nR =")
print(R)