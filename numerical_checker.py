from typing import List, Dict, Tuple, Generator, Optional
import sympy as sp
from sympy.core.relational import Relational, StrictLessThan, StrictGreaterThan, LessThan, GreaterThan, Equality
import numpy as np
from scipy.optimize import brute
import itertools
from dataclasses import dataclass
import warnings

@dataclass
class SearchConfig:
    """Configuration for the numerical search"""
    epsilon: float = 1e-3
    early_stop_threshold: float = 0.5  # Early stopping if violation > threshold
    buffer: float = 0  # Buffer zone around constraint boundaries (as percentage of range)


class OptimizedNumericalChecker:
    """
    Optimized numerical checker for finding feasible solutions to inequality systems.
    Uses vectorized operations, early stopping, and improved numerical stability.
    """
    
    def __init__(self, 
                 variables: List[sp.Symbol], 
                 base_constraints: List[Relational],
                 config: Optional[SearchConfig] = None):
        """
        Initialize checker with improved numerical handling.

        Args:
            variables: List of SymPy symbols to solve for
            base_constraints: List of base constraints that must be satisfied
            config: Search configuration parameters
        """
        self.variables = variables
        self.base_constraints = base_constraints
        self.config = config or SearchConfig()
        self.total_points_checked = 0
        self.valid_points = []

        # Validate inputs
        if not all(isinstance(v, sp.Symbol) for v in variables):
            raise ValueError("All variables must be SymPy symbols")
        if not all(isinstance(c, Relational) for c in base_constraints):
            raise ValueError("All constraints must be SymPy Relational")

    def relative_comparison(self, lhs: float, rhs: float) -> bool: # checked
        """
        Performs numerically stable floating-point comparison using relative tolerance.
        For numbers close to zero (< epsilon), checks absolute difference.
        Otherwise, compares the relative difference: |a-b|/max(|a|,|b|) against epsilon.
        This handles both very small and very large numbers appropriately.

        Args:
            lhs: Left-hand side value
            rhs: Right-hand side value
            
        Returns:
            bool: True if values are equal within relative tolerance

        
        """
        if abs(lhs) < self.config.epsilon and abs(rhs) < self.config.epsilon:
            return True
        return abs(lhs - rhs) / max(abs(lhs), abs(rhs)) < self.config.epsilon

    def find_feasible_point(self, 
                           additional_constraints: List[Relational], 
                           steps_per_var: int = 20) -> Tuple[bool, Optional[Dict[sp.Symbol, float]]]: #checked. 
        """
        Find a point that satisfies all constraints using optimized search.
        
        Args:
            additional_constraints: Additional constraints beyond base constraints
            steps_per_var: Number of grid points per variable
            
        Returns:
            Tuple of (found_solution, solution_point)
            If no solution found, returns (False, None)
        """
        all_constraints = self.base_constraints + additional_constraints
        bounds = self.infer_bounds(all_constraints)
        
        # Convert bounds to format suitable for scipy.optimize.brute
        ranges = [(bounds[var][0], bounds[var][1], complex(steps_per_var)) 
                 for var in self.variables]
        
        # Define objective function for optimization
        def objective(x):
            point = dict(zip(self.variables, x))
            #print(f"Testing point: {point}")  # Debug print
            if self.check_constraints_at_point(point, all_constraints):
                return 0.0
            return 1.0
        
        try:
            # Use scipy's brute force optimizer with improved efficiency
            result = brute(objective, ranges, finish=None)
            point = dict(zip(self.variables, result))
            
            # Verify solution
            if self.check_constraints_at_point(point, all_constraints):
                print("Final verified solution:", point)  # Print final verified solution
                return True, point
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
        
        return False, None

    def check_constraints_at_point(self, 
                                 point: Dict[sp.Symbol, float], 
                                 constraints: List[Relational]) -> bool:
        """
        Check if a point satisfies all constraints with improved numerical stability.
        
        Args:
            point: Dictionary mapping variables to values
            constraints: List of constraints to check
            
        Returns:
            bool: True if all constraints are satisfied, False otherwise
        """
        self.total_points_checked += 1

        try:
            bounds = self.infer_bounds(constraints)
            
            # First check if point is sufficiently far from variable bounds
            for var, value in point.items():
                var_range = bounds[var][1] - bounds[var][0]
                min_buffer = bounds[var][0] + var_range * self.config.buffer
                max_buffer = bounds[var][1] - var_range * self.config.buffer
                
                if value < min_buffer or value > max_buffer:
                    return False
            for constraint in constraints:
                lhs_val = float(constraint.lhs.subs(point).evalf())
                rhs_val = float(constraint.rhs.subs(point).evalf())
                
                if isinstance(constraint, StrictGreaterThan):
                    if not lhs_val > rhs_val + self.config.epsilon:
                        return False
                elif isinstance(constraint, GreaterThan):
                    if not lhs_val >= rhs_val:
                        return False
                elif isinstance(constraint, StrictLessThan):
                    if not lhs_val < rhs_val - self.config.epsilon:
                        return False
                elif isinstance(constraint, LessThan):
                    if not lhs_val <= rhs_val:
                        return False

            self.valid_points.append(point.copy())
            return True
            
        except (TypeError, ValueError):
            return False  # Handle invalid evaluations

    def infer_bounds(self, all_constraints: List[Relational]) -> Dict[sp.Symbol, List[float]]: #checked
        """Keep existing implementation"""
        bounds = {var: [0, 1] for var in self.variables}
        
        for constraint in all_constraints:
            if isinstance(constraint, Relational):
                lhs, rhs = constraint.lhs, constraint.rhs
                
                if lhs in self.variables and rhs.is_number:
                    value = float(rhs)
                    if isinstance(constraint, (StrictLessThan, LessThan)):
                        bounds[lhs][1] = min(bounds[lhs][1], value)
                    elif isinstance(constraint, (StrictGreaterThan, GreaterThan)):
                        bounds[lhs][0] = max(bounds[lhs][0], value)
                
                elif rhs in self.variables and lhs.is_number:
                    value = float(lhs)
                    if isinstance(constraint, (StrictLessThan, LessThan)):
                        bounds[rhs][0] = max(bounds[rhs][0], value)
                    elif isinstance(constraint, (StrictGreaterThan, GreaterThan)):
                        bounds[rhs][1] = min(bounds[rhs][1], value)
        
        return bounds
    
    def save_results_to_csv(self, filename: str):
        """Save valid points to CSV and return satisfaction percentage."""
        import csv
        import os
        
        # Calculate satisfaction percentage
        satisfaction_percentage = (len(self.valid_points) / self.total_points_checked * 100) if self.total_points_checked > 0 else 0
        
        # Prepare CSV header and rows
        header = list(self.variables)
        rows = [[point[var] for var in header] for point in self.valid_points]
        
        # Save to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([str(var) for var in header])  # Convert symbols to strings
            writer.writerows(rows)
        
        return satisfaction_percentage