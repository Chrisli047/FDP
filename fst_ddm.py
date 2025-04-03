import copy
import os
import sys
import time
import matplotlib.pyplot as plt
import networkx as nx

from executor_manager import ExecutorManager
from function_utils import FunctionProfiler, merge_constraints
from sqlite_utils import read_from_sqlite, SQLiteReader
from concurrent.futures import ProcessPoolExecutor

init_constraints = []  # Global variable to store initial constraints

class TreeNode:
    def __init__(self, intersection_id, constraints=None, vertices=None, pivot_record=0):
        self.intersection_id = intersection_id  # ID of the intersection (record_id)
        self.constraints = constraints if constraints is not None else []  # Constraints for this node, defaults to []
        self.vertices = vertices if vertices is not None else []  # Associated vertices, defaults to []
        # self.rounded_vertices = []  # Vertices rounded to the nearest interval
        self.left_children = None  # Left child
        self.right_children = None  # Right child
        # self.skip_flag = False  # Flag to indicate if this node should be skipped
        self.not_enough_vertices = False

        # for record the intersection with the larger and smaller intersection
        self.larger_intersection = []
        self.smaller_intersection = []
        self.pivot_record = pivot_record

        self.uuid = None  # UUID for storing vertices in the database
        #
        # def save_vertices(self):
        #     """Save this node’s vertices using the TreeStorage class."""
        #     TreeStorage.save_vertices(self)
        #
        # def load_vertices(self):
        #     """Load this node’s vertices using the TreeStorage class."""
        #     TreeStorage.load_vertices(self)


def round_vertices(data):
    return [[round(value, 3) for value in sublist] for sublist in data]

def build_individual_tree(tree_node, intersections_to_process, n, vi_tree):

    smaller_intersection = []
    larger_intersection = []

    # Insert records into the VI Tree with progress tracking
    for record_id in intersections_to_process:
        print(f"Processing record {record_id}...")
        VITree.insert(tree_node, record_id, n)

    tree_node.larger_intersection = larger_intersection
    tree_node.smaller_intersection = smaller_intersection



class VITree:
    processing_stack = []

    executor = ProcessPoolExecutor()  # Global process pool for parallel execution

    def __init__(self):
        self.root = None  # Initialize the tree with no root

    def create_root(self, record_id, constraints, vertices, n, pivot_record):
        global init_constraints

        new_node = TreeNode(record_id, None, vertices)
        # Set the root if the tree is empty
        self.root = new_node

        # Update the global variable and node properties
        init_constraints = constraints
        print(f"Initial constraints for record {record_id}: {init_constraints}")

        # Explicitly set root node properties
        self.root.intersection_id = record_id
        self.root.vertices = vertices if vertices is not None else []
        self.root.pivot_record = pivot_record


    @classmethod
    def insert(cls, current_root_node, record_id, n=None):
        global init_constraints

        # Use a stack to manage nodes for non-recursive traversal
        # stack = [current_root_node.left_children, current_root_node.right_children]
        stack = [current_root_node]

        while stack:
            # print(len(cache))
            current = stack.pop()
            # Get the record from the database

            insert_record = SQLiteReader.get_record_by_id(record_id)
            # print(f"Processing record {record_id}: {insert_record}")

            partition, all_positive, all_negative = FunctionProfiler.check_sign_change_single(insert_record, current.vertices)

            if all_positive:
                if insert_record[-2] == current.pivot_record:
                    current.smaller_intersection.append(insert_record[-1])
                else:
                    current.larger_intersection.append(insert_record[-2])
            if all_negative:
                if insert_record[-2] == current.pivot_record:
                    current.larger_intersection.append(insert_record[-1])
                else:
                    current.smaller_intersection.append(insert_record[-2])

            if not partition:
                continue  # Skip to the next iteration if not satisfied


            if current.left_children is None and current.right_children is None:

                left_merged_constraints = merge_constraints(current.constraints + [-record_id], init_constraints)
                # print(f"Left merged constraints: {left_merged_constraints} \n")
                # right_merged_constraints = merge_constraints(current.constraints + [record_id], init_constraints)


                left_children_vertices = FunctionProfiler.compute_vertices(left_merged_constraints)

                if len(left_children_vertices) <= n:
                    continue

                start_time = time.time()
                right_children_vertices = FunctionProfiler.get_right_vertices(insert_record, current.vertices, left_children_vertices)

                current.left_children = TreeNode(
                    -record_id,
                    constraints=[-record_id] + current.constraints
                )

                # expirement_data.counter += 1

                current.right_children = TreeNode(
                    record_id,
                    constraints=[record_id] + current.constraints
                )

                current.left_children.vertices = left_children_vertices
                current.right_children.vertices = right_children_vertices

                current.left_children.pivot_record = current.pivot_record
                current.right_children.pivot_record = current.pivot_record

                continue

            stack.append(current.left_children)
            stack.append(current.right_children)

    @classmethod
    def insert_mp(cls, current_root_node, record_id, n=None):
        """
        Insert a node into the VI tree using a non-recursive method.
        Instead of computing immediately, we collect independent tasks and process them in parallel at the end.
        """
        global init_constraints

        stack = [current_root_node]

        task_list = []  # Store tasks for deferred execution

        while stack:
            current = stack.pop()

            # Get the record from the database
            insert_record = SQLiteReader.get_record_by_id(record_id)

            # Check sign change
            partition, all_positive, all_negative = FunctionProfiler.check_sign_change_single(insert_record, current.vertices)

            if all_positive:
                if insert_record[-2] == current.pivot_record:
                    current.smaller_intersection.append(insert_record[-1])
                else:
                    current.larger_intersection.append(insert_record[-2])
            if all_negative:
                if insert_record[-2] == current.pivot_record:
                    current.larger_intersection.append(insert_record[-1])
                else:
                    current.smaller_intersection.append(insert_record[-2])

            if not partition:
                continue  # Skip if not satisfied

            # If this is a leaf node, we defer the computation instead of processing immediately
            if current.left_children is None and current.right_children is None:
                left_constraints = current.constraints + [-record_id]
                right_constraints = current.constraints + [record_id]

                left_merged_constraints = merge_constraints(left_constraints, init_constraints)
                right_merged_constraints = merge_constraints(right_constraints, init_constraints)

                # task_list.append((current, left_merged_constraints, right_merged_constraints))  # ✅ Pass precomputed constraints

                task_list.append(
                    (insert_record, current, current.vertices, left_merged_constraints, right_merged_constraints))

                continue

            stack.append(current.left_children)
            stack.append(current.right_children)

        if len(task_list) == 0:
            return

        # vertex_computation_start_time = time.time()

        # # 🔥 Parallelize the deferred vertex computations
        # with ProcessPoolExecutor() as executor:
        #     results = executor.map(cls.process_insert_task, task_list, chunksize=500)
        #
        # # 🔥 Parallelize vertex computations with the **persistent executor**
        # results = list(cls.executor.map(cls.process_insert_task, task_list, chunksize=5))

        executor = ExecutorManager.get_executor(max_workers=24)  # ✅ Get the shared executor


        vertex_computation_start_time = time.time()
        results = list(executor.map(cls.process_insert_task, task_list, chunksize= len(task_list) // 24 if len(task_list) // 24 >= 1 else len(task_list)))

        # print(f"Time taken to compute vertices: {time.time() - vertex_computation_start_time}")

        FunctionProfiler.total_time_compute_vertices_mp += time.time() - vertex_computation_start_time

        # Assign results to nodes
        # for (current, left_merged_constraints, right_merged_constraints), (left_vertices, right_vertices) in zip(
        #         task_list, results):

        for (insert_record, current, current_vertices, left_merged_constraints, right_merged_constraints), (left_vertices, right_vertices) in zip(
                task_list, results):


            if len(left_vertices) <= n or len(right_vertices) <= n:
                continue

            if [current.vertices].count(left_vertices) > 0 or [current.vertices].count(right_vertices) > 0:
                continue

            current.left_children = TreeNode(
                -record_id,
                constraints=[-record_id] + current.constraints
            )

            current.right_children = TreeNode(
                record_id,
                constraints=[record_id] + current.constraints
            )

            current.left_children.vertices = left_vertices
            current.right_children.vertices = right_vertices

            current.left_children.pivot_record = current.pivot_record
            current.right_children.pivot_record = current.pivot_record

        # FunctionProfiler.total_time_compute_vertices_mp += time.time() - vertex_computation_start_time


    # @staticmethod
    # def process_insert_task(task):
    #     """
    #     Process a single deferred insert task.
    #     This is executed in parallel for multiple tasks.
    #     """
    #
    #     current, left_merged_constraints, right_merged_constraints = task  # ✅ Now we only compute vertices
    #
    #     left_vertices = FunctionProfiler.compute_vertices(left_merged_constraints)
    #     right_vertices = FunctionProfiler.compute_vertices(right_merged_constraints)
    #
    #     return left_vertices, right_vertices

    @staticmethod
    def process_insert_task(task):
        """
        Process a single deferred insert task.
        This is executed in parallel for multiple tasks.
        """

        insert_record, current, current_vertices, left_merged_constraints, right_merged_constraints = task  # ✅ Now we only compute vertices
        # print(insert_record, current_vertices, left_merged_constraints, right_merged_constraints)

        start_time = time.time()

        left_vertices = FunctionProfiler.compute_vertices(left_merged_constraints)

        # print(f'time taken to compute left vertices: {time.time() - start_time}')

        start_time = time.time()
        # right_vertices = FunctionProfiler.get_right_vertices(insert_record, current_vertices, left_vertices)
        # right_vertices = FunctionProfiler.get_right_vertices_simple(current_vertices, left_vertices)
        get_vertex_time = time.time() - start_time
        # print(f'time taken to compute get right vertices: {time.time() - start_time}')

        start_time = time.time()
        right_vertices = FunctionProfiler.compute_vertices(right_merged_constraints)
        compute_vertex_time = time.time() - start_time

        # if get_vertex_time > compute_vertex_time:
        #     print(len(left_vertices))
        #     print(f"Get vertex time: {get_vertex_time}")
        #     print(f"Compute vertex time: {compute_vertex_time}")

        # print(f'time taken to compute right vertices: {time.time() - start_time}')


        return left_vertices, right_vertices


    @classmethod
    def tree_sort(cls, current_root_node, n):

        pivot_one = current_root_node.pivot_record
        # Use a queue to implement level-order traversal, along with layer tracking
        queue = [(current_root_node, 0, None)]  # Each element is a tuple (node, layer, parent)
        current_layer = 0
        layer_output = []

        while queue:
            current, layer, parent = queue.pop(0)  # Dequeue the front node

            # Check if we've moved to a new layer
            if layer > current_layer:
                current_layer = layer

            # Fetch record from the database
            record_id = abs(current.intersection_id)  # Use positive ID for fetching
            record = SQLiteReader.get_record_by_id(record_id)

            # Append parent's larger and smaller intersections to the current node
            if parent:
                current.larger_intersection = (
                    parent.larger_intersection + current.larger_intersection if parent.larger_intersection else current.larger_intersection
                )
                current.smaller_intersection = (
                    parent.smaller_intersection + current.smaller_intersection if parent.smaller_intersection else current.smaller_intersection
                )

                if current.intersection_id < 0:
                    if record[-2] == pivot_one:
                        current.larger_intersection.append(record[-1])
                    else:
                        current.smaller_intersection.append(record[-2])

                if current.intersection_id > 0:
                    if record[-2] == pivot_one:
                        current.smaller_intersection.append(record[-1])
                    else:
                        current.larger_intersection.append(record[-2])


            # Update the intersections for children based on the current record
            if current.left_children:
                queue.append((current.left_children, layer + 1, current))

            if current.right_children:
                queue.append((current.right_children, layer + 1, current))

            continue

            if current.left_children is None and current.left_children is None:


                if not (len(current.smaller_intersection) > 1 or len(current.larger_intersection) > 1):
                    # print(len(current.smaller_intersection))
                    # print(len(current.larger_intersection))
                    continue
                #
                left_node = TreeNode(
                    current.intersection_id,
                    constraints = copy.deepcopy(current.constraints),
                    vertices = copy.deepcopy(current.vertices),
                    pivot_record = pivot_one
                )

                right_node = TreeNode(
                    current.intersection_id,
                    constraints = copy.deepcopy(current.constraints),
                    vertices = copy.deepcopy(current.vertices),
                    pivot_record = pivot_one
                )

                current.left_children = left_node
                current.right_children = right_node
                #
                left_records = current.smaller_intersection
                left_intersections_to_process = SQLiteReader.find_matching_records(left_records) if left_records else []
                right_records = current.larger_intersection
                right_intersections_to_process = SQLiteReader.find_matching_records(right_records) if right_records else []

                # print(f"Left records: {left_records}")
                # print(f"Left: {left_intersections_to_process}")
                # print(f"Right records: {right_records}")
                # print(f"Right: {right_intersections_to_process}")

                current.left_children.pivot_record = left_records[0] if left_intersections_to_process else pivot_one
                current.right_children.pivot_record = right_records[0] if right_intersections_to_process else pivot_one

                build_individual_tree(current.left_children, left_intersections_to_process, n)
                VITree.tree_sort(current.left_children, n)

                build_individual_tree(current.right_children, right_intersections_to_process, n)
                VITree.tree_sort(current.right_children, n)


    def print_tree_by_layer_v1(self, m, n, db_name, conn):
        """
        Print the VI Tree layer by layer, showing each node's ID, vertices, and database record.
        Handles negative IDs by converting them to positive when fetching records.
        Parameters:
            m (int): Number of functions.
            n (int): Dimension of functions.
            db_name (str): Database file name.
            conn: SQLite database connection.
        """
        if self.root is None:
            print("The tree is empty.")
            return

        # Use a queue to implement level-order traversal, along with layer tracking
        queue = [(self.root, 0)]  # Each element is a tuple (node, layer)
        current_layer = 0
        layer_output = []

        while queue:
            current, layer = queue.pop(0)  # Dequeue the front node

            # Check if we've moved to a new layer
            if layer > current_layer:
                # Print all nodes in the previous layer
                print(f"Layer {current_layer}:")
                for node_output in layer_output:
                    print(node_output)
                print()  # Blank line between layers
                layer_output = []  # Reset the layer output
                current_layer = layer

            # Fetch record from the database
            record_id = abs(current.intersection_id)  # Use positive ID for fetching
            record = SQLiteReader.get_record_by_id(record_id)

            if current.left_children or current.right_children:
                node_type = "Internal Node"
            else:
                node_type = "Leaf Node"

            # Add the current node's details to the layer output
            layer_output.append(
                f"Node ID: {current.intersection_id}, Vertices: {current.vertices}, Record: {record}"
                # f"constraints: {current.constraints}, Vertices: {current.vertices}, Record: {record}, "
                f" Larger: {current.larger_intersection}, Pivot: {current.pivot_record}, Smaller: {current.smaller_intersection}"
                f"Type: {node_type}"
            )

            # Enqueue the left and right children if they exist, with incremented layer
            if current.left_children:
                queue.append((current.left_children, layer + 1))
            if current.right_children:
                queue.append((current.right_children, layer + 1))

        # Print the last layer
        print(f"Layer {current_layer}:")
        for node_output in layer_output:
            print(node_output)

    def get_height(self):
        if self.root is None:
            return 0
        # Iterative approach using a queue (BFS)
        from collections import deque
        queue = deque([(self.root, 1)])
        max_depth = 0
        while queue:
            node, depth = queue.popleft()
            if node is not None:
                max_depth = max(max_depth, depth)
                queue.append((node.left_children, depth + 1))
                queue.append((node.right_children, depth + 1))
        return max_depth

    def get_leaf_count(self):
        if self.root is None:
            return 0
        # Iterative approach using a stack (DFS)
        stack = [self.root]
        leaf_count = 0
        while stack:
            node = stack.pop()
            if node is not None:
                # Check if it is a leaf and not flagged as not_enough_vertices
                if node.left_children is None and node.right_children is None and not node.not_enough_vertices:
                    leaf_count += 1
                else:
                    stack.append(node.left_children)
                    stack.append(node.right_children)
        return leaf_count

    def get_total_node_count(self):
        if self.root is None:
            return 0
        # Iterative approach using a stack (DFS)
        stack = [self.root]
        total_count = 0
        while stack:
            node = stack.pop()
            if node is not None:
                # Increment the total node count for every node visited
                total_count += 1
                stack.append(node.left_children)
                stack.append(node.right_children)
        return total_count


    def print_tree_by_layer(self, m, n, db_name, conn):
        """
        Print the VI Tree layer by layer, showing each node's ID, vertices, and database record.
        Handles negative IDs by converting them to positive when fetching records.
        Parameters:
            m (int): Number of functions.
            n (int): Dimension of functions.
            db_name (str): Database file name.
            conn: SQLite database connection.
        """
        if self.root is None:
            print("The tree is empty.")
            return

        # Use a queue to implement level-order traversal, along with layer tracking
        queue = [(self.root, 0, None)]  # Each element is a tuple (node, layer, parent)
        current_layer = 0
        layer_output = []

        while queue:
            current, layer, parent = queue.pop(0)  # Dequeue the front node

            # Check if we've moved to a new layer
            if layer > current_layer:
                # Print all nodes in the previous layer
                print(f"Layer {current_layer}:")
                for node_output in layer_output:
                    print(node_output)
                print()  # Blank line between layers
                layer_output = []  # Reset the layer output
                current_layer = layer

            # Fetch record from the database
            record_id = abs(current.intersection_id)  # Use positive ID for fetching
            record = read_from_sqlite(m, n, db_name=db_name, record_id=record_id, conn=conn)

            # Append parent's larger and smaller intersections to the current node
            if parent:
                current.larger_intersection = (
                    parent.larger_intersection + current.larger_intersection if parent.larger_intersection else current.larger_intersection
                )
                current.smaller_intersection = (
                    parent.smaller_intersection + current.smaller_intersection if parent.smaller_intersection else current.smaller_intersection
                )

                if current.intersection_id < 0:
                    current.larger_intersection.append(record[-1])

                if current.intersection_id > 0:
                    current.smaller_intersection.append(record[-1])


            if current.left_children or current.right_children:
                node_type = "Internal Node"
            else:
                node_type = "Leaf Node"

            # Add the current node's details to the layer output
            # layer_output.append(
            #     f"Node ID: {current.intersection_id}, Vertices: {current.vertices}, Record: {record}, "
            #     f"Larger records: {current.larger_intersection}, Smaller records: {current.smaller_intersection}, "
            #     f"Type: {node_type}"
            # )

            layer_output.append(
                # f"Node ID: {current.intersection_id}, Vertices: {current.vertices}, Record: {record}, "
                f"Larger records: {current.larger_intersection}, The pivot record = {current.pivot_record}, Smaller records: {current.smaller_intersection}, "
                f"Type: {node_type}"
            )


            #
            # # Add the current node's details to the layer output
            # layer_output.append(
            #     f"Node ID: {current.intersection_id}, Vertices: {current.vertices}, Record: {record}, "
            #     f"Larger Intersection: {current.larger_intersection}, Smaller Intersection: {current.smaller_intersection}"
            # )

            # Update the intersections for children based on the current record
            if current.left_children:
                queue.append((current.left_children, layer + 1, current))

            if current.right_children:
                queue.append((current.right_children, layer + 1, current))

        # Print the last layer
        if layer_output:
            print(f"Layer {current_layer}:")
            for node_output in layer_output:
                print(node_output)
