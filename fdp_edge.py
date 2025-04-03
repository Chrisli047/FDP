import itertools
import time

import numpy as np
import numba as nb

class TimerNew:
    total_time_find_edges_high_dim = 0
    total_time_split_edges_by_equality_generic = 0
    total_time_deduplicate_intersection_points = 0
    total_time_compute_intersection_on_edge = 0
    total_time_update_intersections_with_equality = 0
    total_time_compute_new_edges = 0
    total_time_split_edges_by_equality_generic_check_only = 0

    total_time = 0

    domain_counter = 0

    total_time_compute_new_edges_calls = 0

    @classmethod
    def print_times(cls):
        for attr, value in cls.__dict__.items():
            if not attr.startswith("__") and not callable(value):
                print(f"{attr} = {value}")


def find_edges_high_dim(constraints, vertices, tol=1e-9):
    start_time = time.time()
    """
    Finds all edges of a d-dimensional simple polytope defined by linear equality constraints and vertices.

    Parameters:
        constraints: list of tuples (a1, a2, ..., ad, c) representing the constraint
                     a1*x1 + a2*x2 + ... + ad*xd + c = 0.
        vertices: list of vertices, each given as a list [x1, x2, ..., xd].
        tol: tolerance for checking if a constraint is active at a vertex.

    Returns:
        A list of tuples (v1, v2, common_active_constraints) where v1 and v2 are endpoints
        of the edge (as d-dimensional points) and common_active_constraints is a tuple
        of converted constraint indices (of length d-1) that are active along the entire edge.
        The conversion is done by mapping each index i to -(i+1).
    """
    # Dimension of the space.
    d = len(vertices[0])

    # Step 1: Determine the active constraints for each vertex.
    # For a simple polytope in d-dimensions, each vertex should have exactly d active constraints.
    active_for_vertex = []
    for vertex in vertices:
        active = []
        for i, constr in enumerate(constraints):
            # Split constraint into coefficients and constant.
            coeffs = constr[:-1]
            c = constr[-1]
            # Evaluate a1*x1 + a2*x2 + ... + ad*xd + c.
            value = sum(a * x for a, x in zip(coeffs, vertex)) + c
            if abs(value) < tol:
                active.append(i)  # Record the index of the constraint.
        active.sort()
        active_for_vertex.append(active)

    # Optionally, check that each vertex has exactly d active constraints.
    for idx, active in enumerate(active_for_vertex):
        if len(active) != d:
            raise ValueError(
                f"Vertex at index {idx} has {len(active)} active constraints, expected {d} (polytope may not be simple)."
            )

    # Step 2: For each vertex, generate each (d-1)-subset of its active constraints.
    # Two vertices sharing the same (d-1)-subset are adjacent.
    subset_dict = {}
    for v_idx, active in enumerate(active_for_vertex):
        # Each vertex has d active constraints; drop one at a time.
        for drop in active:
            # Form the key as the sorted tuple of active constraints excluding the one dropped.
            key = tuple(x for x in active if x != drop)
            if key not in subset_dict:
                subset_dict[key] = []
            subset_dict[key].append(v_idx)

    # Step 3: Identify edges from subsets that appear exactly twice.

    # edges = []
    edges = {}
    for common_key, v_indices in subset_dict.items():
        if len(v_indices) == 2:
            v1_idx, v2_idx = v_indices
            v1 = tuple(vertices[v1_idx])
            v2 = tuple(vertices[v2_idx])
            # Convert common_key indices: 0 -> -1, 1 -> -2, etc.
            converted_key = tuple(-(i + 1) for i in common_key)
            # edges.append((v1, v2, converted_key))
            # print((v1, v2))
            edges[(v1, v2)] = converted_key

    TimerNew.total_time += time.time() - start_time
    return edges

def compute_new_edges_new(vertex_dict, new_id, d):
    start_time = time.time()
    new_edge = {}

    # # Get the dimension from any vertex key. All keys are assumed to have the same dimension.
    # d = len(next(iter(vertex_dict)))
    required_overlap_len = d - 2

    # Iterate over all unique pairs of vertices
    for key1, key2 in itertools.combinations(vertex_dict.keys(), 2):
        # Convert value tuples to sets for intersection operation.
        set1 = set(vertex_dict[key1])
        set2 = set(vertex_dict[key2])
        overlap = set1.intersection(set2)

        if len(overlap) == required_overlap_len:
            # Sort the overlapping values (for a consistent order) and add new_id as the last element.
            overlap_sorted = tuple(sorted(overlap))
            new_edge[(key1, key2)] = overlap_sorted + (new_id,)

    total_time_compute_new_edges = time.time() - start_time
    TimerNew.total_time_compute_new_edges += total_time_compute_new_edges
    TimerNew.total_time += total_time_compute_new_edges

    return new_edge


def edge_relation(edge, coeffs, c):
    v0, v1 = edge
    # Manual dot product for f0 and f1 in one loop.
    f0 = c
    f1 = c
    for a, x0, x1 in zip(coeffs, v0, v1):
        f0 += a * x0
        f1 += a * x1

    # If both endpoints lie on the same side (or exactly on the hyperplane):
    if f0 >= 0 and f1 >= 0:
        return "larger", None, None
    if f0 <= 0 and f1 <= 0:
        return "less", None, None

    start_time = time.time()
    # Compute interpolation parameter t (f0 and f1 have opposite signs)
    t = f0 / (f0 - f1)

    # Compute the intersection point via linear interpolation.
    d = len(v0)
    intersection_coords = [0] * d
    for i in range(d):
        intersection_coords[i] = v0[i] + t * (v1[i] - v0[i])
    intersection = tuple(intersection_coords)

    # Identify which vertex is on the "larger" side.
    if f0 > 0:
        larger_vertex, less_vertex = v0, v1
    else:
        larger_vertex, less_vertex = v1, v0

    edge_larger = (larger_vertex, intersection)
    edge_less = (less_vertex, intersection)

    elapsed = time.time() - start_time
    TimerNew.total_time_compute_intersection_on_edge += elapsed

    return "intersect", (edge_larger, edge_less), intersection


def split_edges_by_equality_generic_new(edges, equality, tol=1e-9):
    start_time = time.time()

    coeffs = equality[:-1]
    c = equality[-1]

    edges_larger = {}
    edges_smaller = {}
    intersection_points = {}

    # Iterate using items() to avoid extra dict lookups.
    for edge, val in edges.items():
        relation, new_edges, intersection_point = edge_relation(edge, coeffs, c)
        if relation == "larger":
            edges_larger[edge] = val
        elif relation == "less":
            edges_smaller[edge] = val
        else:
            # For intersecting edges, new_edges is a pair: (edge_larger, edge_less)
            edge_larger, edge_less = new_edges
            edges_larger[edge_larger] = val
            edges_smaller[edge_less] = val
            intersection_points[intersection_point] = val

    elapsed = time.time() - start_time
    TimerNew.total_time_split_edges_by_equality_generic += elapsed
    TimerNew.total_time += elapsed

    return edges_larger, edges_smaller, intersection_points
