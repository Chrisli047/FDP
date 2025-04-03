import sqlite3
import time
from copy import deepcopy
from typing import Any

import cdd
import numpy as np
from numpy import bool
from concurrent.futures import ProcessPoolExecutor
import jax.numpy as jnp
from typing_extensions import override

from executor_manager import ExecutorManager
from sqlite_utils import SQLiteReader

# ✅ Move compute_values outside the class to allow pickling
def compute_values(vertices, coefficients, constant, start_idx, end_idx):
    """
    Compute the min and max values for a chunk of vertices.
    Must be a top-level function to be pickled by multiprocessing.
    """
    values_chunk = vertices[start_idx:end_idx].dot(coefficients) + constant
    return values_chunk.min(), values_chunk.max()


def find_overlapping_equations(equations):
    """
    Finds overlapping 2D linear equations of the form ax1 + bx2 = 0 and counts the number of overlaps.

    Parameters:
        equations (list of tuples): List of equations in the form (a, b, c).

    Returns:
        int: Total count of overlaps.
    """
    seen_ratios = []
    overlap_count = 0

    for a, b, c in equations:
        if b != 0:
            ratio = a / b
        else:
            ratio = float('inf')  # Handle vertical lines where b = 0

        # Check if the ratio is already in the seen list
        if ratio in seen_ratios:
            overlap_count += 1
        else:
            seen_ratios.append(ratio)

    print("overlapping equations: ", overlap_count)
    return overlap_count


def generate_constraints(n, var_min, var_max):
    """
    Generate constraints for an n-dimensional space with var_min and var_max.
    Returns a list of tuples in the format (coe1, coe2, ..., coen, constant).
    # Define inequalities in the form A x + b > 0
    """
    constraints = []

    for i in range(n):
        # Lower bound: x_i >= var_min -> x_i - var_min >= 0
        lower_bound = [0] * n
        lower_bound[i] = 1
        constraints.append((*lower_bound, -var_min))

        # Upper bound: x_i <= var_max -> -x_i + var_max >= 0
        upper_bound = [0] * n
        upper_bound[i] = -1
        constraints.append((*upper_bound, var_max))

    return constraints


def merge_constraints(node_constraints, init_constraints):
    """
    Merge node.constraints with init_constraints by fetching records from the database.
    Parameters:
        node_constraints (list): Constraints for the current node.
        init_constraints (list): Global initial constraints.
        m (int): Number of functions.
        n (int): Dimension of functions.
        db_name (str): Database file name.
        conn: SQLite database connection.
    Returns:
        list of tuples: Merged constraints.
    """
    # Deep-copy init_constraints to avoid modifying the original

    merged_constraints = deepcopy(init_constraints)


    for record_id in node_constraints:
        # Fetch record from the database
        if record_id < 0:
            # record = FunctionProfiler.read_from_sqlite(m=m, n=n, db_name=db_name, record_id=-record_id, conn=conn)
            record = SQLiteReader.get_record_by_id(-record_id)
            # Negate coefficients, keep constant unchanged
            record = tuple(-coeff for coeff in record[:-3]) + (-record[-3],)  # Convert to a tuple
        else:
            # record = FunctionProfiler.read_from_sqlite(m=m, n=n, db_name=db_name, record_id=record_id, conn=conn)
            record = SQLiteReader.get_record_by_id(record_id)
            # Keep coefficients, negate constant
            record = tuple(record[:-3]) + (record[-3],)  # Convert to a tuple

        # Append the modified record as a tuple to merged_constraints
        merged_constraints.append(record)

    # print("merged_constraints: ", merged_constraints)
    return merged_constraints


class FunctionProfiler:
    total_time_compute_vertices = 0.0
    total_time_check_function = 0.0
    total_time_read_from_sqlite = 0.0
    total_time_satisfies_all_constraints = 0.0  # New accumulator for satisfies_all_constraints
    total_time_get_right_vertices = 0.0
    total_time_remove_redundant_constraints = 0.0

    total_feasibility_check_calls = 0
    total_compute_vertices_calls = 0
    total_get_right_vertices_calls = 0

    compute_vertex_set = []

    total_time_compute_vertices_mp = 0.0

    @classmethod
    def compute_vertices(cls, constraints):
        # print("constraints: ", constraints)
        # print("len(constraints): ", len(constraints))
        start_time = time.time()
        try:
            rows = []
            for constraint in constraints:
                # Split the tuple into coefficients and the constant
                *coefficients, constant = constraint
                row = [constant] + coefficients  # Include constant as the first element
                rows.append(row)

            # Convert to cdd matrix
            mat = cdd.matrix_from_array(rows, rep_type=cdd.RepType.INEQUALITY)

            # cdd.matrix_redundancy_remove(mat)

            # Create polyhedron from the matrix
            poly = cdd.polyhedron_from_matrix(mat)
            ext = cdd.copy_generators(poly)

            vertices = []
            for row in ext.array:
                if row[0] == 1.0:  # This indicates a vertex
                    vertex = [round(coord, 8) for coord in row[1:]]
                    # vertex = [coord for coord in row[1:]]
                    vertices.append(vertex)

        except Exception as e:
            print(f"Error in compute_vertices: {e}")
            vertices = []

        elapsed_time = time.time() - start_time
        cls.total_time_compute_vertices += elapsed_time
        cls.total_compute_vertices_calls += 1
        return vertices


    @classmethod
    def check_sign_change_single(cls, func, vertices, cache=None) -> tuple[bool | Any, bool, bool]:
        if len(vertices) == 0:
            return False, False, False
        """
        Checks if the function evaluates to both >0 and <0 across the vertex set,
        with timer and cache logic included.

        Parameters:
            func (list or tuple): A linear function represented as coefficients followed by a constant term.
            vertices (list or numpy.ndarray): A list or array of vertices, where each vertex is an n-dimensional point.
            cache (dict, optional): A dictionary for caching results.

        Returns:
            bool: True if the function evaluates to both positive and negative values, False otherwise.
        """


        start_time = time.time()

        # print("func: ", func)
        # Separate coefficients and constant
        coefficients = np.array(func[:-3], dtype=float)
        constant = float(func[-3])

        # print("coefficients: ", coefficients, "constant: ", constant)
        # print("vertices: ", vertices)
        # Convert vertices to a numpy array if it's a list
        vertices = np.asarray(vertices, dtype=float)

        # Vectorized evaluation: values = (vertices @ coefficients) + constant
        values = vertices.dot(coefficients) + constant

        # Check overall min and max
        val_min = values.min()
        val_max = values.max()

        # We have a sign change if val_min < 0 and val_max > 0
        result = (val_min < 0.0) and (val_max > 0.0)
        # print("result: ", result)
        # print("_" * 50)

        # all_positive = np.all(values >= 0)
        # # # print("all_positive: ", all_positive)
        # all_negative = np.all(values <= 0)
        # # print("all_negative: ", all_negative)

        elapsed_time = time.time() - start_time
        cls.total_time_check_function += elapsed_time
        cls.total_feasibility_check_calls += 1

        return result, False, False

    # @classmethod
    # def check_sign_change_single(cls, func, vertices, cache=None) -> tuple[bool, bool, bool, np.ndarray, np.ndarray]:
    #     if len(vertices) == 0:
    #         return False, False, False, np.array([]), np.array([])
    #
    #     start_time = time.time()
    #
    #     # Separate coefficients and constant.
    #     # (Note: your original code uses func[:-3] and func[-3].)
    #     coefficients = np.array(func[:-3], dtype=float)
    #     constant = float(func[-3])
    #
    #     # Convert vertices to a numpy array (if not already).
    #     vertices = np.asarray(vertices, dtype=float)
    #
    #     # Evaluate the function for each vertex: f(v) = (vertices @ coefficients) + constant.
    #     values = vertices.dot(coefficients) + constant
    #
    #     # Determine overall min and max.
    #     val_min = values.min()
    #     val_max = values.max()
    #
    #     # There is a sign change if some values are negative and some are positive.
    #     result = (val_min < 0.0) and (val_max > 0.0)
    #
    #     # Check if all values are non-negative or non-positive.
    #     all_positive = np.all(values >= 0)
    #     all_negative = np.all(values <= 0)
    #
    #     # Split the vertices into those with strictly positive and strictly negative function values.
    #     vertices_positive = np.asarray(vertices[values > 0])
    #     vertices_negative = np.asarray(vertices[values < 0])
    #
    #     elapsed_time = time.time() - start_time
    #     cls.total_time_check_function += elapsed_time
    #     cls.total_feasibility_check_calls += 1
    #
    #     return result, all_positive, all_negative, vertices_positive, vertices_negative

    @classmethod
    def check_sign_change(cls, func, vertices, cache=None) -> tuple[bool | Any, bool, bool]:
        """
        Checks if the function evaluates to both >0 and <0 across the vertex set,
        with timer and cache logic included.

        Parameters:
            func (list or tuple): A linear function represented as coefficients followed by a constant term.
            vertices (list or numpy.ndarray): A list or array of vertices, where each vertex is an n-dimensional point.
            cache (dict, optional): A dictionary for caching results.

        Returns:
            tuple: (has_sign_change, all_positive, all_negative)
        """
        start_time = time.time()

        # Separate coefficients and constant
        coefficients = np.array(func[:-3], dtype=float)
        constant = float(func[-3])

        # Convert vertices to a numpy array if it's a list
        vertices = np.asarray(vertices, dtype=float)


        num_workers = min(16, len(vertices) // 20)  # Adjusting number of processes dynamically
        if num_workers <= 1:
            values = vertices.dot(coefficients) + constant
            val_min, val_max = values.min(), values.max()
        else:
            chunk_size = (len(vertices) + num_workers - 1) // num_workers
            # tasks = []
            #
            # with ProcessPoolExecutor(max_workers=num_workers) as executor:
            #     for i in range(0, len(vertices), chunk_size):
            #         tasks.append(executor.submit(compute_values, i, min(i + chunk_size, len(vertices))))

            executor = ExecutorManager.get_executor()  # ✅ Get the shared executor

            # ✅ Prepare arguments as iterables
            start_indices = list(range(0, len(vertices), chunk_size))
            end_indices = [min(i + chunk_size, len(vertices)) for i in start_indices]
            coefficients_list = [coefficients] * len(start_indices)  # Broadcast coefficients
            constant_list = [constant] * len(start_indices)  # Broadcast constant
            vertices_list = [vertices] * len(start_indices)  # Broadcast vertices

            # ✅ Use executor.map() with multiple iterables
            results = list(executor.map(compute_values, vertices_list, coefficients_list, constant_list, start_indices,
                                        end_indices))

            # tasks = [executor.submit(compute_values, i, min(i + chunk_size, len(vertices))) for i in range(0, len(vertices), chunk_size)]

            # Collect min and max across all chunks
            # results = [task.result() for task in tasks]

            val_min = min(r[0] for r in results)
            val_max = max(r[1] for r in results)


        # Sign check
        result = (val_min < 0.0) and (val_max > 0.0)
        all_positive = val_min >= 0.0
        all_negative = val_max <= 0.0

        elapsed_time = time.time() - start_time
        cls.total_time_check_function += elapsed_time
        cls.total_feasibility_check_calls += 1

        return result, all_positive, all_negative


    @classmethod
    def read_from_sqlite(cls, m, n, db_name="test_intersections.db", record_id=None, conn=None):
        """
        Read records from a dynamically named SQLite table based on m and n.
        Optionally filter by ID. Use an existing connection if provided.
        Returns a single record (without the index) as a tuple or a list of tuples.
        """
        start_time = time.time()

        close_conn = False
        if conn is None:
            conn = sqlite3.connect(db_name)
            close_conn = True

        cursor = conn.cursor()
        table_name = f"intersections_m{m}_n{n}"

        # Query records
        if record_id is not None:
            cursor.execute(f"SELECT * FROM {table_name} WHERE id = ?", (record_id,))
            result = cursor.fetchone()  # Fetch a single record
            result = tuple(result[1:]) if result else None  # Skip the index (position 0)
        else:
            cursor.execute(f"SELECT * FROM {table_name}")
            result = [tuple(row[1:]) for row in cursor.fetchall()]  # Skip the index for all rows

        # Close the connection if it was created in this function
        if close_conn:
            conn.close()

        elapsed_time = time.time() - start_time
        cls.total_time_read_from_sqlite += elapsed_time
        return result


    @classmethod
    def satisfies_all_constraints(cls, vertex, inequalities):
        """
        Check if the given vertex satisfies all linear inequalities of the form:
        a1*x1 + a2*x2 + ... + c > 0
        """
        start_time = time.time()

        print("inequalities: ", inequalities)
        print("vertex: ", vertex)
        # Convert vertex to a NumPy array for efficient computation
        vertex = np.array(vertex)

        for constraint in inequalities:
            # Separate coefficients and constant
            coefficients = np.array(constraint[:-1])
            c = constraint[-1]

            # Compute lhs using dot product for efficiency
            lhs = np.dot(coefficients, vertex) + c

            # Check if lhs <= 0
            if lhs <= 0:
                elapsed_time = time.time() - start_time
                cls.total_time_satisfies_all_constraints += elapsed_time
                return False

        elapsed_time = time.time() - start_time
        cls.total_time_satisfies_all_constraints += elapsed_time
        return True

    # @classmethod
    # def get_right_vertices(cls, left_vertices, current_vertices, new_vertices, func, cache=None):
    #     """
    #     Determines the right vertices from the combined set of left + current vertices
    #     based on the given function and record_id, eliminating duplicates.
    #     Includes timing and call count.
    #
    #     Parameters:
    #         left_vertices (list): Vertices from the left child.
    #         current_vertices (list): Current vertices.
    #         func (list or tuple): Linear function represented as coefficients followed by a constant term.
    #         cache (dict, optional): Cache to store and retrieve precomputed results.
    #
    #     Returns:
    #         list: Right vertices.
    #     """
    #     start_time = time.time()
    #
    #     # Determine right vertices
    #     right_vertices = []
    #
    #     # Combine left and current vertices and remove duplicates while preserving order
    #     combined_vertices = []
    #     seen = set()
    #     for vertex in new_vertices:
    #         vertex_tuple = tuple(vertex)
    #         if vertex_tuple not in seen:
    #             seen.add(vertex_tuple)
    #             right_vertices.append(vertex)
    #
    #     for vertex in left_vertices + current_vertices:
    #         vertex_tuple = tuple(vertex)
    #         if vertex_tuple not in seen:
    #             seen.add(vertex_tuple)
    #             combined_vertices.append(vertex)
    #
    #
    #     # Separate coefficients and constant
    #     coefficients = np.array(func[:-1])
    #     constant = func[-1]
    #
    #
    #     for vertex in combined_vertices:
    #         value = np.dot(vertex, coefficients) - constant
    #         if value >= 0:
    #             right_vertices.append(vertex)
    #
    #
    #     # Update timing and call count
    #     elapsed_time = time.time() - start_time
    #     cls.total_time_get_right_vertices += elapsed_time
    #     cls.total_get_right_vertices_calls += 1
    #
    #     return right_vertices

    ## Jan 19, 2025
    @classmethod
    def get_combination_ids_for_function(cls, m, target_function_num):
        """
        Get all IDs of combinations involving the target function.

        Args:
            m (int): Total number of functions.
            target_function_num (int): Target function number (1-based index).

        Returns:
            list: IDs of combinations where the target function appears.
        """
        # Ensure valid input
        if target_function_num < 1 or target_function_num > m:
            return []

        # Calculate the IDs where the target function appears
        result_ids = []
        id_counter = 1
        for i in range(1, m + 1):
            if i == target_function_num:
                # Add IDs for all pairs (target_function, f_j) where j > i
                result_ids.extend(range(id_counter, id_counter + m - i))
            id_counter += m - i
        return result_ids

    @classmethod
    def get_combination_id_for_pair(cls, m, function1_id, function2_id):
        """
        Get the combination ID for a specific pair of functions.

        Args:
            m (int): Total number of functions.
            function1_id (int): ID of the first function (1-based index).
            function2_id (int): ID of the second function (1-based index).

        Returns:
            int: Combination ID for the pair, or None if the pair is invalid.
        """
        # Ensure valid input
        if function1_id < 1 or function2_id < 1 or function1_id > m or function2_id > m:
            return None
        if function1_id == function2_id:
            return None  # Invalid pair: a function cannot pair with itself

        # Ensure function1_id is smaller than function2_id for consistency
        if function1_id > function2_id:
            function1_id, function2_id = function2_id, function1_id

        # Calculate the combination ID
        id_counter = 1
        for i in range(1, function1_id):
            id_counter += m - i  # Skip combinations for earlier functions

        # Add the offset for the second function in the range
        combination_id = id_counter + (function2_id - function1_id - 1)
        return combination_id

    @classmethod
    def check_if_redundant(cls, func, vertices, cache=None) -> tuple[bool | Any, bool, bool, int]:
        """
        Checks if the function evaluates to both >0 and <0 across the vertex set,
        with timer and cache logic included. Also counts the number of zero values.

        Parameters:
            func (list or tuple): A linear function represented as coefficients followed by a constant term.
            vertices (list or numpy.ndarray): A list or array of vertices, where each vertex is an n-dimensional point.
            cache (dict, optional): A dictionary for caching results.

        Returns:
            tuple:
                - bool: True if the function evaluates to both positive and negative values, False otherwise.
                - bool: True if all values are non-negative.
                - bool: True if all values are non-positive.
                - int: Count of zero values in the computed results.
        """

        start_time = time.time()

        # Separate coefficients and constant
        coefficients = np.array(func[:-3], dtype=float)
        constant = float(func[-3])

        # Convert vertices to a numpy array if it's a list
        vertices = np.asarray(vertices, dtype=float)

        # Vectorized evaluation
        values = vertices.dot(coefficients) + constant


        # Compute min, max, and zero count
        val_min = values.min()
        val_max = values.max()
        # zero_count = np.count_nonzero(values == 0.0)
        zero_count = np.count_nonzero(np.abs(values) < 1e-6)
        # zero_count = np.sum(np.abs(values) < 1e-6)  # Faster than count_nonzero

        # print(values[np.abs(values) < 1e-6])
        # print(zero_count)

        # Determine results
        result = (val_min < 0.0) and (val_max > 0.0)
        all_positive = np.all(values >= 0)
        all_negative = np.all(values <= 0)

        elapsed_time = time.time() - start_time
        cls.total_time_check_function += elapsed_time
        cls.total_feasibility_check_calls += 1

        return result, all_positive, all_negative, zero_count


    @classmethod
    def find_non_redundant_constraints(cls, test_constraints, vertices):
        # print('-' * 50)
        # print(f'original len: {len(test_constraints)}')
        # Convert constraints and vertices to NumPy arrays
        A = np.array(test_constraints, dtype=np.float64)  # (m x d+1)
        V = np.array(vertices, dtype=np.float64)  # (n x d)

        # Split coefficients and constants
        coefficients = A[:, :-1]  # All but last column (m x d)
        constants = A[:, -1]  # Last column (m,)

        # Matrix multiplication to compute function values
        values = coefficients @ V.T + constants[:, np.newaxis]  # (m x n)

        # Count zero values for each constraint
        zero_counts = np.sum(np.abs(values) < 1e-6, axis=1)

        # Keep constraints where zero_count > 1
        non_redundant_mask = zero_counts > 1
        non_redundant_constraints = A[non_redundant_mask]
        # print(f'non-redundant len: {len(non_redundant_constraints)}')

        return non_redundant_constraints.tolist()


    @classmethod
    def find_non_redundant_constraints_ids(cls, constraint_ids, init_constraints, vertices):
        """
        Find non-redundant constraints from both constraint_ids and init_constraints.

        Args:
            constraint_ids (list): List of constraint record IDs.
            init_constraints (list of tuples): List of initial constraints.
            vertices (list): List of vertices.
            SQLiteReader (object): Database reader object to fetch constraints.

        Returns:
            list: [
                [non-redundant constraint IDs],
                [indexes of non-redundant init_constraints]
            ]
        """

        # Retrieve constraints from database based on IDs
        constraints = []
        resolved_ids = []
        for record_id in constraint_ids:
            if record_id < 0:
                record = SQLiteReader.get_record_by_id(-record_id)
                record = tuple(-coeff for coeff in record[:-3]) + (
                -record[-3],)  # Negate coefficients, keep constant
            else:
                record = SQLiteReader.get_record_by_id(record_id)
                record = tuple(record[:-3]) + (record[-3],)  # Keep coefficients, negate constant

            constraints.append(record)
            resolved_ids.append(record_id)

        # Merge constraints from both sources
        all_constraints = constraints + init_constraints  # Combine
        num_constraints = len(constraints)  # Track split point

        # Convert to NumPy arrays
        A = np.array(all_constraints, dtype=np.float64)  # (m x d+1)
        V = np.array(vertices, dtype=np.float64)  # (n x d)

        # Split coefficients and constants
        coefficients = A[:, :-1]  # All but last column (m x d)
        constants = A[:, -1]  # Last column (m,)

        # Compute function values
        values = coefficients @ V.T + constants[:, np.newaxis]  # (m x n)

        # Count zero values
        zero_counts = np.sum(np.abs(values) < 1e-6, axis=1)

        # Get non-redundant constraints
        non_redundant_mask = zero_counts > 1

        # Extract indices
        non_redundant_indices = np.where(non_redundant_mask)[0]

        # Separate non-redundant constraints into two lists
        non_redundant_constraint_ids = []
        non_redundant_init_indexes = []

        for idx in non_redundant_indices:
            if idx < num_constraints:  # From constraint_ids
                non_redundant_constraint_ids.append(resolved_ids[idx])
            else:  # From init_constraints
                non_redundant_init_indexes.append(idx - num_constraints)

        return [non_redundant_constraint_ids, non_redundant_init_indexes]


    @classmethod
    def get_right_vertices(cls, new_intersection, current_vertices, left_children_vertices):
        # print(' ' * 50)
        # print("Current vertices: ", current_vertices)
        # print("Left children vertices: ", left_children_vertices)

        start_time = time.time()

        """
        Find the right vertices based on the new intersection.

        Args:
            new_intersection (tuple): A linear equation representing the intersection.
            current_vertices (list of lists): List of current vertices.
            left_children_vertices (list of lists): List of left children vertices.

        Returns:
            list: Right vertices satisfying the given conditions.
        """

        # Convert input lists to NumPy arrays for fast computation
        current_vertices = np.array(current_vertices, dtype=np.float64)
        left_children_vertices = np.array(left_children_vertices, dtype=np.float64)

        # Convert sets for easy comparison
        current_set = set(map(tuple, current_vertices))
        left_set = set(map(tuple, left_children_vertices))

        # Condition 1: In current but not in left
        unique_vertices = np.array([v for v in current_set if v not in left_set], dtype=np.float64)

        # Condition 2: In current vertices, make new_intersection zero
        dimension = current_vertices.shape[1]  # Get number of variables in vertices
        coefficients = np.array(new_intersection[:dimension], dtype=np.float64)  # Match dimension
        constant = np.float64(new_intersection[-3])

        # Compute values for the intersection equation
        values = current_vertices @ coefficients + constant
        zero_vertices = current_vertices[np.abs(values) < 1e-6]

        # Condition 3: In left but not in current
        left_unique_vertices = np.array([v for v in left_set if v not in current_set], dtype=np.float64)

        # Stack only non-empty arrays
        non_empty_arrays = [arr for arr in [zero_vertices, unique_vertices, left_unique_vertices] if arr.size > 0]

        if not non_empty_arrays:  # If all arrays are empty, return an empty list
            return []

        right_vertices = list({tuple(v) for v in np.vstack(non_empty_arrays)})

        # Update timing and call count
        elapsed_time = time.time() - start_time
        cls.total_time_get_right_vertices += elapsed_time
        cls.total_get_right_vertices_calls += 1
        cls.total_time_compute_vertices += elapsed_time

        # print("Right vertices: ", right_vertices)
        return right_vertices  # Convert back to list

    @classmethod
    def get_right_vertices_simple(cls, current_vertices, left_children_vertices):
        start_time = time.time()
        # Convert each vertex (which is a list) to a tuple so we can use set operations.
        current_set = set(tuple(v) for v in current_vertices)
        left_set = set(tuple(v) for v in left_children_vertices)

        # Determine the dimension from one of the vertices (if available)
        if current_vertices:
            d = len(current_vertices[0])
        else:
            return []

        # Condition 1: Include the zero vector (of appropriate dimension) if present in current.
        zero_vertex = tuple([0.0] * d)
        result_set = set()
        if zero_vertex in current_set:
            result_set.add(zero_vertex)

        # Condition 2: Include vertices in left but not in current.
        result_set |= (left_set - current_set)

        # Condition 3: Include vertices in current but not in left.
        result_set |= (current_set - left_set)

        # Update timing and call count
        elapsed_time = time.time() - start_time
        cls.total_time_get_right_vertices += elapsed_time
        cls.total_get_right_vertices_calls += 1

        # Return the result as a list of lists.
        return [list(vertex) for vertex in result_set]

    @classmethod
    def evaluate_function(cls, func, vertices):
        """Evaluate the function at all given vertices with minimal overhead."""
        if not vertices:  # Avoid unnecessary array conversion
            return np.array([], dtype=np.float64)

        # Use views for speed
        coefficients = np.asarray(func[:-3], dtype=np.float64)
        constant = func[-3]  # No conversion needed

        # Convert to NumPy array only once (if needed)
        vertices = np.asarray(vertices, dtype=np.float64)

        # Avoid explicit dot product when possible (NumPy optimization)
        return np.einsum('ij,j->i', vertices, coefficients) + constant

    @classmethod
    def check_function_location(cls, func, left, both, right):

        start_time = time.time()
        all_vertices = right + both + left

        values = cls.evaluate_function(func, all_vertices)  # Compute function values



        # Get indices for each region
        right_count = len(right)
        both_count = len(both)

        values_right_both = values[:right_count + both_count]  # First part (right + both)
        values_left_both = values[-(both_count + len(left)):]  # Last part (left + both)

        if values_right_both.size > 0:
            if np.min(values_right_both) >= 0 or np.max(values_right_both) <= 0:
                return "left"

        if values_left_both.size > 0:
            if np.min(values_left_both) >= 0 or np.max(values_left_both) <= 0:
                return "right"

        elapsed_time = time.time() - start_time

        cls.total_time_check_function += elapsed_time
        cls.total_feasibility_check_calls += 1

        return "both"


    @classmethod
    def check_sign_change_single_split(cls, func, vertices, cache=None):
        if len(vertices) == 0:
            return False, np.array([]), np.array([])

        # Extract coefficients and constant
        coefficients = np.asarray(func[:-3], dtype=np.float64)
        constant = np.float64(func[-3])

        # Convert vertices to NumPy array if necessary
        vertices = np.asarray(vertices, dtype=np.float64)

        # Vectorized evaluation
        values = vertices @ coefficients + constant

        # Boolean masks
        greater_mask = values >= 0
        less_mask = values < 0

        # Filter vertices using NumPy's fast indexing
        # Convert to Python lists before returning
        greater_vertices = vertices[greater_mask].tolist()
        less_vertices = vertices[less_mask].tolist()

        return greater_vertices, less_vertices
