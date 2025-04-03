import argparse
import sys

from sqlite_utils import get_all_ids, SQLiteReader
from function_utils import generate_constraints, FunctionProfiler
import time  # Import time for measuring execution
import sqlite3


from fdp_check import VITree, build_individual_tree
from fdp_edge import TimerNew


def log_output(file_path):
    """Redirects print statements to a log file while keeping stdout functional."""
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a")  # Append mode

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(file_path)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process VI tree with given m and n.")
    parser.add_argument("m", type=int, help="Number of functions (m)")
    parser.add_argument("n", type=int, help="Dimension of functions (n)")
    parser.add_argument("r", type=int, help="Number of records to process")
    parser.add_argument("--db", type=str, default="test_intersections.db", help="Database file (default: intersections.db)")
    parser.add_argument("--var_min", type=float, default=0, help="Minimum value for variables (default: 0)")
    parser.add_argument("--var_max", type=float, default=100, help="Maximum value for variables (default: 10)")
    args = parser.parse_args()

    m = args.m
    n = args.n
    db_name = args.db
    var_min = args.var_min
    var_max = args.var_max

    num_of_records_to_process = args.r

    # Create a log file for this m, n combination
    # log_filename = f"data_{m}_{n}"
    # log_output(log_filename)  # Redirect print output to file

    # Dynamically construct the table name
    table_name = f"intersections_m{m}_n{n}"

    # Open a single connection to the database
    conn = sqlite3.connect(db_name)

    SQLiteReader.read_all_from_sqlite(m, n, num_of_records_to_process)
    # SQLiteReader.read_all_from_sqlite(m, n)
    # print(SQLiteReader.records)

    print(f"Read all records from the database.")

    global processing_stack

    # initial_records = list(range(1, m + 1))[0: num_of_records_to_process]
    # initial_records = random.sample(initial_records, 5)
    # initial_records = [5, 3, 1, 7, 4, 6, 2]
    # print(f"Initial records: {initial_records}")

    initial_records = list(range(1, m + 1))
    pivot_element = initial_records[0]


    # intersection_to_process = SQLiteReader.find_matching_records(initial_records)
    intersection_to_process = list(range(1, num_of_records_to_process))
    # print(f"Intersection to process: {intersection_to_process}")

    # print(f"Intersection to process: {intersection_to_process}")

    # Generate constraints using n (as dimensionality), var_min, and var_max
    constraints = generate_constraints(n, var_min, var_max)

    SQLiteReader.init_constraint = constraints

    # Compute vertices for the initial domain
    vertices = FunctionProfiler.compute_vertices(constraints)

    # print(f'constraints: {constraints}')
    # print(f'vertices: {vertices}')

    # Create a VI Tree
    vi_tree = VITree()

    vi_tree.create_root(0, pivot_element)
    # exit(0)
    # print(vi_tree.root.vertex_pairs)
    print("len of vertex pairs:", len(vi_tree.root.vertex_pairs))

    # exit(0)

    VITree.processing_stack.append(vi_tree.root)

    start_time = time.time()
    while len(VITree.processing_stack) > 0:
        current_node = VITree.processing_stack.pop()
        build_individual_tree(vi_tree.root, intersection_to_process, n)
        # build_individual_tree(current_node, [2, 3 ,4], n)
        # VITree.tree_sort(current_node, n)


    # print("Number of records processed:", num_of_records_to_process)
    print("Total time taken to build the VI Tree:", time.time() - start_time)

    # Set a counter for the number of intersection partitions the domain

    print("Root larger intersection:", vi_tree.root.larger_intersection)
    print("Root smaller intersection:", vi_tree.root.smaller_intersection)
    print("\nVI Tree Structure (Layer by Layer with Records):")
    # vi_tree.print_tree_by_layer_v1(m, n, db_name, conn)

    # vi_tree.visualize_tree()

    # Print the height of the tree
    print(f"Height of the VI Tree: {vi_tree.get_height()}")

    # Print the number of leaf nodes
    print(f"Number of leaf nodes in the VI Tree: {vi_tree.get_leaf_count()}")
    print(f"Number of nodes in the VI Tree: {vi_tree.get_total_node_count()}")

    print("Total time in compute_vertices:", FunctionProfiler.total_time_compute_vertices)
    print("Total time in compute_vertices in mp mode:", FunctionProfiler.total_time_compute_vertices_mp)
    print("Total time in check_function:", FunctionProfiler.total_time_check_function)
    print("Total time in read_from_sqlite:", FunctionProfiler.total_time_read_from_sqlite)
    print("Total time of compute right vertices:", FunctionProfiler.total_time_get_right_vertices)

    print("Total feasibility checking calls:", FunctionProfiler.total_feasibility_check_calls)
    print("Total compute vertices calls:", FunctionProfiler.total_compute_vertices_calls)
    print("Total compute right vertices calls:", FunctionProfiler.total_get_right_vertices_calls)

    print("Total call in feasibility checking: ", FunctionProfiler.total_feasibility_check_calls)

    # Close the database connection
    conn.close()
    print("Database connection closed.")
    print(" " * 10)
    #
    # plot_linear_equations(records_to_draw)

    # print(f"Total time taken to build the VI Tree: {TimerNew.total_time}")
    TimerNew.print_times()

    print("Method total time:",
          FunctionProfiler.total_time_check_function + TimerNew.total_time_compute_intersection_on_edge + TimerNew.total_time_compute_new_edges)

