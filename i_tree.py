from function_utils import merge_constraints

from simplex import SimplexWrapper
from sqlite_utils import read_from_sqlite

init_constraints = []  # Global variable to store initial constraints

class TreeNode:
    def __init__(self, intersection_id, constraints=None, vertices=None):
        self.intersection_id = intersection_id  # ID of the intersection (record_id)
        self.constraints = constraints if constraints is not None else []  # Constraints for this node, defaults to []
        self.left_children = None  # Left child
        self.right_children = None  # Right child
        self.skip_flag = False  # Flag to indicate if this node should be skipped


class ITree:
    def __init__(self):
        self.root = None  # Initialize the tree with no root

    def insert(self, record_id, constraints, vertices=None, m=None, n=None, db_name=None, conn=None):
        global init_constraints

        new_node = TreeNode(record_id, None, vertices)

        if self.root is None:
            # Set the root if the tree is empty
            self.root = new_node

            # Update the global variable and node properties
            init_constraints = constraints
            print(f"Initial constraints for record {record_id}: {init_constraints}")

            # Explicitly set root node properties
            self.root.intersection_id = record_id
            self.root.vertices = vertices if vertices is not None else []

            # Initialize left and right children with constraints
            self.root.left_children = TreeNode(-record_id, [-record_id])
            self.root.right_children = TreeNode(record_id, [record_id])

            return

        # Use a stack to manage nodes for non-recursive traversal
        stack = [self.root.left_children, self.root.right_children]

        while stack:
            current = stack.pop()
            # Get the record from the database
            insert_record = read_from_sqlite(m=m, n=n, db_name=db_name, record_id=record_id, conn=conn)[0:-2]
            merged_constraints = merge_constraints(current.constraints, init_constraints)

            if SimplexWrapper.check_constraints_feasibility(merged_constraints, insert_record, 0, 1):
                if current.left_children is None and current.right_children is None:
                    current.left_children = TreeNode(
                        -record_id,
                        constraints=[-record_id] + current.constraints
                    )
                    current.right_children = TreeNode(
                        record_id,
                        constraints=[record_id] + current.constraints
                    )
                    continue
                else:
                    # Add left and right children to the stack for further traversal
                    if current.left_children is not None:
                        stack.append(current.left_children)
                    if current.right_children is not None:
                        stack.append(current.right_children)
            else:
                continue


    def print_tree_by_layer(self, m, n, db_name, conn):
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
            record = read_from_sqlite(m, n, db_name=db_name, record_id=record_id, conn=conn)

            # Add the current node's details to the layer output
            layer_output.append(
                f"Node ID: {current.intersection_id}, Record: {record}"
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
        """
        Calculate the height of the VI Tree.
        Returns:
            int: Height of the tree (max depth of any node).
        """
        def _height(node):
            if node is None:
                return 0
            left_height = _height(node.left_children)
            right_height = _height(node.right_children)
            return max(left_height, right_height) + 1

        return _height(self.root)

    def get_leaf_count(self):
        """
        Count the number of leaf nodes in the VI Tree.
        Returns:
            int: Number of leaf nodes in the tree.
        """
        def _leaf_count(node):
            if node is None:
                return 0
            if node.left_children is None and node.right_children is None:
                return 1
            return _leaf_count(node.left_children) + _leaf_count(node.right_children)

        return _leaf_count(self.root)