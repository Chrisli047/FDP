import copy
import time
import numpy as np

from executor_manager import ExecutorManager
from function_utils import FunctionProfiler
from sqlite_utils import read_from_sqlite, SQLiteReader
from concurrent.futures import ProcessPoolExecutor

from fdp_edge import find_edges_high_dim, split_edges_by_equality_generic_new, compute_new_edges_new


class TreeNode:
    def __init__(self, intersection_id=0, pivot_record=0, vertex=None, vertex_pairs=None):
        self.intersection_id = intersection_id  # ID of the intersection (record_id)
        self.vertex = vertex
        self.vertex_pairs = vertex_pairs if vertex_pairs else {}

        self.left_children = None  # Left child
        self.right_children = None  # Right child

        # for record the intersection with the larger and smaller intersection
        self.larger_intersection = []
        self.smaller_intersection = []
        self.pivot_record = pivot_record


def build_individual_tree(tree_node, intersections_to_process, n):
    for record_id in intersections_to_process:
        print(f"Processing record {record_id}...")
        VITree.insert(tree_node, record_id, n)


class VITree:
    processing_stack = []

    executor = ProcessPoolExecutor()  # Global process pool for parallel execution

    def __init__(self):
        self.root = None  # Initialize the tree with no root

    def create_root(self, record_id, pivot_record):

        vertices = FunctionProfiler.compute_vertices(SQLiteReader.init_constraint)
        print(f"Computing vertices for record {record_id}...")
        # print(vertices)
        vertices = [p for p in vertices if any(coord != 0.0 for coord in p)]
        # print(f"Filtered points: {vertices}")

        new_node = TreeNode(record_id, pivot_record, vertices)
        # Set the root if the tree is empty
        self.root = new_node

        self.root.intersection_id = record_id
        self.root.pivot_record = pivot_record
        self.root.vertex = np.asarray(vertices, dtype=float)
        self.root.vertex_pairs = find_edges_high_dim(SQLiteReader.init_constraint, vertices)
        print(f"self.root.vertex_pairs: {self.root.vertex_pairs}")
        print(len(self.root.vertex_pairs))


    @classmethod
    def insert(cls, current_root_node, record_id, n=None):

        stack = [current_root_node]

        while stack:
            current = stack.pop()

            insert_record = SQLiteReader.get_record_by_id(record_id)
            # print(current.vertex)
            partition, _, _ = FunctionProfiler.check_sign_change_single(insert_record, list(current.vertex))

            if not partition:
                continue  # Skip to the next iteration if not satisfied

            if current.left_children is None and current.right_children is None:

                edges_larger, edges_smaller, intersection_points = split_edges_by_equality_generic_new(current.vertex_pairs, insert_record[:-2])
                new_edge = compute_new_edges_new(intersection_points, record_id, n)

                # print("New edge: ", new_edge)

                left_pairs = edges_larger | new_edge
                left_vertices = {vertex for edge in left_pairs.keys() for vertex in edge}
                # left_vertices = get_unique_vertices(left_pairs)

                right_pairs = edges_smaller | new_edge
                right_vertices = {vertex for edge in right_pairs.keys() for vertex in edge}


                current.left_children = TreeNode(
                    -record_id,
                    vertex=left_vertices,
                    vertex_pairs=left_pairs
                )

                current.right_children = TreeNode(
                    record_id,
                    vertex=right_vertices,
                    vertex_pairs=right_pairs
                )


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

        stack = [current_root_node]

        task_list = []  # Store tasks for deferred execution

        while stack:

            # current = stack.pop()
            #
            # # Get the record from the database
            # insert_record = SQLiteReader.get_record_by_id(record_id)
            #
            # # Check sign change
            # partition, all_positive, all_negative = FunctionProfiler.check_sign_change_single(insert_record, current.vertex)
            #
            # if all_positive:
            #     if insert_record[-2] == current.pivot_record:
            #         current.smaller_intersection.append(insert_record[-1])
            #     else:
            #         current.larger_intersection.append(insert_record[-2])
            # if all_negative:
            #     if insert_record[-2] == current.pivot_record:
            #         current.larger_intersection.append(insert_record[-1])
            #     else:
            #         current.smaller_intersection.append(insert_record[-2])
            #
            # if not partition:
            #     continue  # Skip if not satisfied

            # If this is a leaf node, we defer the computation instead of processing immediately
            if current.left_children is None and current.right_children is None:
                vertex_pairs = current.vertex_pairs
                record_to_insert = insert_record[0:-2]

                task_list.append(
                    (current, vertex_pairs, record_to_insert, record_id))

                continue

            stack.append(current.left_children)
            stack.append(current.right_children)

        if len(task_list) == 0:
            return

        executor = ExecutorManager.get_executor(max_workers=24)  # ✅ Get the shared executor


        vertex_computation_start_time = time.time()
        results = list(executor.map(cls.process_insert_task, task_list, chunksize= len(task_list) // 24 if len(task_list) // 24 >= 1 else len(task_list)))

        # print(f"Time taken to compute vertices: {time.time() - vertex_computation_start_time}")

        FunctionProfiler.total_time_compute_vertices_mp += time.time() - vertex_computation_start_time

        # Assign results to nodes
        # for (current, left_merged_constraints, right_merged_constraints), (left_vertices, right_vertices) in zip(
        #         task_list, results):

        for (current, vertex_pairs, record_to_insert, record_id), (left_pairs, right_pairs, left_vertices, right_vertices) in zip(
                task_list, results):

            current.left_children = TreeNode(
                -record_id,
                vertex=left_vertices,
                vertex_pairs=left_pairs
            )

            current.right_children = TreeNode(
                record_id,
                vertex=right_vertices,
                vertex_pairs=right_pairs
            )

            current.left_children.pivot_record = current.pivot_record
            current.right_children.pivot_record = current.pivot_record

            current.vertex_pairs = None

        # FunctionProfiler.total_time_compute_vertices_mp += time.time() - vertex_computation_start_time


    @staticmethod
    def process_insert_task(task):
        """
        Process a single deferred insert task.
        This is executed in parallel for multiple tasks.
        """

        current, vertex_pairs, record_to_insert, record_id = task  # ✅ Now we only compute vertices
        # print(insert_record, current_vertices, left_merged_constraints, right_merged_constraints)

        edge_groups = split_edges_by_equality_generic(current.vertex_pairs, record_to_insert)

        intersection_edges, more_larger, more_smaller = update_intersections_with_equality(edge_groups["intersect"],
                                                                                           record_to_insert,
                                                                                           record_id)
        # print(f"Intersection edges: {intersection_edges}")

        new_edge = compute_new_edges(edge_groups["intersection_point"], intersection_edges, record_id)
        # print(f"New edge: {new_edge}")

        left_pairs = edge_groups["larger"] + more_larger + new_edge
        # print(f"Left pairs: {left_pairs}")
        left_vertices = get_unique_vertices(left_pairs)

        right_pairs = edge_groups["smaller"] + more_smaller + new_edge
        right_vertices = get_unique_vertices(right_pairs)


        return left_pairs, right_pairs, left_vertices, right_vertices


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

            # continue

            if current.left_children is None and current.left_children is None:

                if not (len(current.smaller_intersection) > 1 or len(current.larger_intersection) > 1):
                    # print(len(current.smaller_intersection))
                    # print(len(current.larger_intersection))
                    continue

                left_node = TreeNode(
                    current.intersection_id,
                    vertex = copy.deepcopy(current.vertex),
                    vertex_pairs = copy.deepcopy(current.vertex_pairs),
                    pivot_record = pivot_one
                )

                right_node = TreeNode(
                    current.intersection_id,
                    vertex = copy.deepcopy(current.vertex),
                    vertex_pairs = copy.deepcopy(current.vertex_pairs),
                    pivot_record = pivot_one
                )


                # continue

                current.left_children = left_node
                current.right_children = right_node
                #
                left_records = current.smaller_intersection
                left_intersections_to_process = SQLiteReader.find_matching_records(left_records) if left_records else []
                # print("left_intersections_to_process: ", left_intersections_to_process)
                right_records = current.larger_intersection
                right_intersections_to_process = SQLiteReader.find_matching_records(right_records) if right_records else []
                # print("right_intersections_to_process: ", right_intersections_to_process)

                # print(f"Left records: {left_records}")
                # print(f"Left: {left_intersections_to_process}")
                # print(f"Right records: {right_records}")
                # print(f"Right: {right_intersections_to_process}")

                current.left_children.pivot_record = left_records[0] if left_intersections_to_process else pivot_one
                current.right_children.pivot_record = right_records[0] if right_intersections_to_process else pivot_one

                # print("Left pivot: ", current.left_children.pivot_record)
                # print("Right pivot: ", current.right_children.pivot_record)

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
                # f"Node ID: {current.intersection_id}, Vertices: {current.vertices}, Record: {record}"
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
                if node.left_children is None and node.right_children is None:
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
