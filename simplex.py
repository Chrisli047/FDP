from scipy.optimize import linprog
import numpy as np
import time

class SimplexWrapper:
    total_simplex_time = 0
    total_simplex_calls = 0

    @classmethod
    def check_constraints_feasibility(cls, constraints, function, var_min, var_max):
        cls.total_simplex_calls += 1
        """
        Checks if a given function satisfies min < 0 and max > 0 under given constraints.

        Parameters:
            constraints: List of tuples representing constraints in the format:
                         (coef1, coef2, ..., constant), interpreted as:
                         coef1 * x1 + coef2 * x2 + ... > constant.
            function: List of coefficients for the function to evaluate (objective function),
                      where the last element is the constant term.
            var_min: Lower bound for all variables.
            var_max: Upper bound for all variables.

        Returns:
            Boolean indicating whether the function satisfies the condition under the constraints.
        """
        start_time = time.time()

        try:
            # Start timer
            start_time = time.time()

            # Determine the number of variables (excluding the constant term)
            num_vars = len(function) - 1

            # Extract coefficients (excluding the constant term)
            function_coefficients = function[:-1]
            function_constant = function[-1]

            # Reformat constraints into A_ub and b_ub
            A_ub = []
            b_ub = []

            for constraint in constraints:
                *coefficients, constant = constraint
                A_ub.append([-coef for coef in coefficients])  # Negate coefficients for linprog
                b_ub.append(constant)  # Negate constant for linprog

            # Convert to numpy arrays
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)


            # Set variable bounds based on var_min and var_max
            var_bounds = [(var_min, var_max)] * num_vars

            # Solve for minimum value of the function
            result_min = linprog(function_coefficients, A_ub=A_ub, b_ub=b_ub, bounds=var_bounds, method='highs')

            # Solve for maximum value of the function by negating it
            function_neg = [-coef for coef in function_coefficients]
            result_max = linprog(function_neg, A_ub=A_ub, b_ub=b_ub, bounds=var_bounds, method='highs')

            # Check if both optimizations were successful
            if not (result_min.success and result_max.success):
                # print(f"Simplex failed: {result_min.message if not result_min.success else result_max.message}")
                return False

            # Extract the minimum and maximum values of the function
            min_value = result_min.fun + function_constant
            max_value = -result_max.fun + function_constant # Negate back to get the actual maximum

            cls.total_simplex_time += time.time() - start_time

            # Check the condition: min_value < 0 and max_value > 0
            if min_value < 0 and max_value > 0:
                # print(f"Function satisfies condition under constraints. Checked in {time.time() - start_time:.6f} seconds.")
                return True
            else:
                # print(f"Function does not satisfy condition. Min: {min_value}, Max: {max_value}")
                return False

        except Exception as e:
            print(f"Error in check_constraints_feasibility: {e}")
            return False
