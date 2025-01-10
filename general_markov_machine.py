import sympy as sp

class GeneralMarkovGameAnalyzer:
    def __init__(self, num_positions, possible_moves, transition_probs):
        """
        Initialize the analyzer for a general 2-player zero-sum game.
        
        Args:
            num_positions (int): Number of possible position states
            possible_moves (list): List of possible moves
            transition_probs (dict): Mapping (current_pos, move) -> {next_pos: prob, 'win': prob, 'lose': prob}
                - current_pos is an integer 0 to num_positions-1
                - move is from possible_moves
                - next_pos is an integer 0 to num_positions-1
                - probabilities should sum to 1 for each (current_pos, move) pair
        """
        self.num_positions = num_positions
        self.possible_moves = possible_moves
        self.transition_probs = transition_probs
        
        # Calculate total number of non-terminal states
        # For each position, we need states for both players
        self.num_non_terminal = 2 * num_positions
        
        # Total states includes terminal states (win/lose)
        self.total_states = self.num_non_terminal + 2
        
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
        p1_strat, p2_strat = strategy_profile.split(',')
        
        # Verify lengths match number of positions
        if len(p1_strat) != self.num_positions or len(p2_strat) != self.num_positions:
            raise ValueError("Strategy length must match number of positions")
            
        # Create strategy dictionaries
        p1_strategy = {pos: move for pos, move in enumerate(p1_strat)}
        p2_strategy = {pos: move for pos, move in enumerate(p2_strat)}
        
        return p1_strategy, p2_strategy
        
    def create_transition_matrix(self, strategy_profile):
        """
        Create the transition matrix P for a given strategy profile.
        
        Args:
            strategy_profile (str): Strategy profile in the format described above
            
        Returns:
            sympy.Matrix: The transition matrix P
        """
        # Initialize transition matrix
        P = sp.zeros(self.total_states, self.total_states)
        
        # Parse strategy profile
        p1_strategy, p2_strategy = self.parse_strategy_profile(strategy_profile)
        
        # Fill transitions for non-terminal states
        for player in [1, 2]:
            for pos in range(self.num_positions):
                # Calculate state index
                # Player 1's states come first, then player 2's states
                from_state = pos if player == 1 else pos + self.num_positions
                
                # Get move from strategy
                strategy = p1_strategy if player == 1 else p2_strategy
                move = strategy[pos]
                
                # Get transition probabilities
                trans_probs = self.transition_probs[(pos, move)]
                
                # Fill matrix
                for next_pos, prob in trans_probs.items():
                    if next_pos == 'win':
                        P[from_state, -2] = prob  # Win state is second to last
                    elif next_pos == 'lose':
                        P[from_state, -1] = prob  # Lose state is last
                    else:
                        # For non-terminal transitions, need to adjust index based on next player
                        next_player = 2 if player == 1 else 1
                        to_state = next_pos if next_player == 1 else next_pos + self.num_positions
                        P[from_state, to_state] = prob
        
        # Terminal states transition to themselves with probability 1
        P[-2, -2] = 1  # win stays win
        P[-1, -1] = 1  # lose stays lose
        
        return P
    
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
        Q = P[:self.num_non_terminal, :self.num_non_terminal]
        R = P[:self.num_non_terminal, -2:]  # Last two columns for win/lose states
        return Q, R

# Example usage:
if __name__ == "__main__":
   
    # Create symbolic parameters
    b_sS = sp.Symbol('b_sS')
    b_sC = sp.Symbol('b_sC')
    b_cS = sp.Symbol('b_cS')
    b_cC = sp.Symbol('b_cC')
    tau = sp.Symbol('tau')

    # Game setup
    num_positions = 2  # simple (0) and complex (1)
    possible_moves = ['S', 'C']

    # Define transition probabilities
    # Format: (pos, move) -> {next_pos: prob, 'win': prob, 'lose': prob}
    transition_probs = {
        # From simple position (0)
        (0, 'S'): {
            'win': (1 - b_sS) * tau,      # no blunder and game ends
            'lose': b_sS,                 # blunder
            0: (1 - b_sS) * (1 - tau),    # no blunder, continue to simple
            1: 0                          # no transitions to complex with simple move
        },
        (0, 'C'): {
            'win': (1 - b_sC) * tau,
            'lose': b_sC,
            0: 0,                         # no transitions to simple with complex move
            1: (1 - b_sC) * (1 - tau)     # no blunder, continue to complex
        },
        # From complex position (1)
        (1, 'S'): {
            'win': (1 - b_cS) * tau,
            'lose': b_cS,
            0: (1 - b_cS) * (1 - tau),
            1: 0
        },
        (1, 'C'): {
            'win': (1 - b_cC) * tau,
            'lose': b_cC,
            0: 0,
            1: (1 - b_cC) * (1 - tau)
        }
    }

    # Create analyzer instance
    analyzer = GeneralMarkovGameAnalyzer(num_positions, possible_moves, transition_probs)

    # Analyze strategy profile "CS,CS"
    # Note: in our encoding, position 0 = simple, 1 = complex
    # So "CS" means C in simple position (0), S in complex position (1)
    strategy_profile = "CS,CS"

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

    # To compute values, you would then use:
    I = sp.eye(analyzer.num_non_terminal)
    I_minus_Q = I - Q
    v = I_minus_Q.inv() * R * sp.Matrix([1, 0])  # [1,0] for terminal values (win,lose)
    print("\nValues v:")
    print(v)