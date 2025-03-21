import re
import os
from graphviz import Digraph
from datetime import datetime
from slim_gsgp.utils.utils import create_result_directory


def preprocess_tree_structure(node):
    if isinstance(node, tuple):
        return tuple([preprocess_tree_structure(child) for child in node])
    elif isinstance(node, list):
        return [preprocess_tree_structure(child) for child in node]
    elif callable(node):
        return f"func:{node.__name__}" if hasattr(node, '__name__') else "func"
    else:
        return node


def parse_tree_string(tree_str):
    """Parse a tree string into a nested structure."""
    # Remove newlines and extra whitespace
    tree_str = re.sub(r'\s+', ' ', tree_str).strip()

    def parse_recursive(s, pos=0):
        # Skip whitespace
        while pos < len(s) and s[pos].isspace():
            pos += 1

        if pos >= len(s):
            return None, pos

        # If it's a function call
        if '(' in s[pos:]:
            # Extract function name
            func_end = s.find('(', pos)
            func_name = s[pos:func_end].strip()
            pos = func_end + 1

            children = []
            # Parse arguments
            while pos < len(s) and s[pos] != ')':
                child, pos = parse_recursive(s, pos)
                if child:
                    children.append(child)

                # Skip whitespace and potential comma
                while pos < len(s) and (s[pos].isspace() or s[pos] == ','):
                    pos += 1

            # Skip closing parenthesis
            if pos < len(s) and s[pos] == ')':
                pos += 1

            return {'type': 'function', 'name': func_name, 'children': children}, pos
        else:
            # It's a terminal or constant
            end = pos
            while end < len(s) and s[end] not in '(), \n\t':
                end += 1
            return {'type': 'terminal', 'name': s[pos:end].strip()}, end

    tree, _ = parse_recursive(tree_str)
    return tree


def create_graphviz_tree(tree, filename='gp_tree', format='png'):
    """Create a Graphviz visualization of the tree."""
    dot = Digraph(comment='GP Tree Visualization')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')

    node_count = [0]  # Use list for mutable counter

    def add_nodes(node, parent_id=None):
        if not node:
            return

        current_id = f"node{node_count[0]}"
        node_count[0] += 1

        if node['type'] == 'function':
            # Use different color for functions
            dot.node(current_id, node['name'], fillcolor='lightgreen')

            for child in node['children']:
                child_id = add_nodes(child, current_id)
                if child_id:
                    dot.edge(current_id, child_id)
        else:
            # Use different color for terminals
            dot.node(current_id, node['name'], fillcolor='lightyellow')

        if parent_id:
            return current_id
        return None

    # Parse the tree if it's a string
    if isinstance(tree, str):
        tree = parse_tree_string(tree)

    add_nodes(tree)

    # Render the graph
    dot.render(filename, format=format, cleanup=True)
    return dot


def visualize_gp_tree(tree_structure, filename='gp_tree', format='png', dataset=None, algorithm=None):
    """Create visualization directly from tree structure instead of parsing strings.

    Parameters:
    -----------
    tree_structure : tuple or list
        The tree structure to visualize
    filename : str
        Base filename for the output visualization file
    format : str
        Output format ('png', 'svg', etc.)
    dataset : str, optional
        Dataset name for organizing visualizations
    algorithm : str, optional
        Algorithm type for organizing visualizations

    Returns:
    --------
    str
        Path to the generated visualization file
    """

    processed_tree = preprocess_tree_structure(tree_structure)

    # Create a visualization
    dot = Digraph(comment='GP Tree Visualization')
    dot.attr('node', shape='box', style='filled')

    node_count = [0]

    # Modified add_nodes function for visualize_gp_tree in tree_visualizer.py
    def add_nodes(node, parent_id=None):
        if node is None:
            return None

        current_id = f"node{node_count[0]}"
        node_count[0] += 1

        if isinstance(node, tuple):  # Function node
            function_name = node[0]
            dot.node(current_id, function_name, fillcolor='lightgreen')

            # Add child nodes
            for child in node[1:]:
                child_id = add_nodes(child, current_id)
                if child_id:
                    dot.edge(current_id, child_id)
        elif isinstance(node, list):  # Collection of trees
            dot.node(current_id, 'collection', fillcolor='lightpink')

            for child in node:
                child_id = add_nodes(child, current_id)
                if child_id:
                    dot.edge(current_id, child_id)
        elif callable(node):  # Function object
            # Handle function objects by using their name without the <locals> part
            if hasattr(node, '__name__'):
                func_name = node.__name__
            else:
                func_name = "function"
            dot.node(current_id, func_name, fillcolor='orange')
        else:  # Terminal node
            # Sanitize the string representation to remove any HTML-like elements
            node_str = str(node).replace('<', '&lt;').replace('>', '&gt;')
            dot.node(current_id, node_str, fillcolor='lightyellow')

        return current_id

    add_nodes(tree_structure)

    # If dataset and algorithm are provided, use the standard result directory
    if dataset and algorithm:
        # Get project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

        # Create result directory
        vis_dir = create_result_directory(
            root_dir=root_dir,
            dataset=dataset,
            algorithm=algorithm,
            result_type="visualizations"
        )

        # Use the provided directory for the output
        filepath = os.path.join(vis_dir, filename)
    else:
        # Use the filename as provided (relative path)
        filepath = filename

    # Render the graph
    dot.render(filepath, format=format, cleanup=True)
    return f"{filepath}.{format}"


def extract_tree_structure(tree):
    """
    Extract tree structure from different types of tree objects.

    Parameters:
    -----------
    tree : object
        The tree object to extract structure from

    Returns:
    --------
    object
        The extracted tree structure, or None if extraction failed
    """
    try:
        # Try different approaches to get the tree structure
        if hasattr(tree, 'get_tree_structure'):
            return tree.get_tree_structure()
        elif hasattr(tree, 'repr_'):
            return tree.repr_
        elif hasattr(tree, 'structure'):
            return tree.structure
        elif hasattr(tree, 'collection'):
            return [t.structure for t in tree.collection]
        else:
            # Last resort: just return the tree itself
            print(f"Warning: Using direct tree object of type {type(tree)} for visualization")
            return tree
    except Exception as e:
        print(f"Error extracting tree structure: {str(e)}")
        return None


def visualize_classification_model(model, base_filename, format='png', dataset=None, algorithm=None, strategy=None):
    """
    Visualize trees in a classification model (both binary and multiclass).

    Parameters:
    -----------
    model : BinaryClassifier or MulticlassClassifier
        The trained classification model
    base_filename : str
        Base filename for the output visualization files
    format : str
        Output format ('png', 'svg', etc.)
    dataset : str, optional
        Dataset name for organizing visualizations
    algorithm : str, optional
        Algorithm type (gp, gsgp, slim) for organizing visualizations
    strategy : str, optional
        Classification strategy (ovr, ovo) for organizing visualizations

    Returns:
    --------
    list
        Paths to the generated visualization files
    """

    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    visualization_paths = []

    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    # Set default values if not provided
    if not dataset:
        dataset = "unknown_dataset"
    if not algorithm:
        algorithm = "unknown_algorithm"

    # Determine strategy from model if not provided
    if not strategy and hasattr(model, 'strategy'):
        strategy = model.strategy.lower()

    # Create the directory structure
    vis_dir = create_result_directory(
        root_dir=root_dir,
        dataset=dataset,
        algorithm=algorithm,
        result_type="visualizations",
        strategy=strategy
    )

    # Determine if it's a binary or multiclass model
    is_multiclass = hasattr(model, 'n_classes') and model.n_classes > 2

    # Add timestamp to the base filename to make it unique
    unique_base_filename = f"{base_filename}_{timestamp}"

    if is_multiclass:
        # For multiclass models, we need to extract individual trees
        if hasattr(model.model, 'trees'):
            trees = model.model.trees
        else:
            # For more complex models like OVO
            print("Model structure doesn't have direct tree access, trying alternate methods")
            trees = [model.model]  # Fall back to treating the whole model as one tree

        # Handle different multiclass strategies (OVR or OVO)
        if hasattr(model, 'strategy') and model.strategy.lower() == 'ovr':
            # One tree per class
            for i, tree in enumerate(trees):
                # Get class label
                class_label = model.class_labels[i] if model.class_labels else f"class_{i}"
                class_filename = os.path.join(vis_dir, f"{unique_base_filename}_{class_label}")

                # Extract and visualize tree
                tree_structure = extract_tree_structure(tree)
                if tree_structure:
                    path = visualize_gp_tree(tree_structure, class_filename, format)
                    visualization_paths.append(path)
                    print(f"Saved visualization for {class_label} to {path}")
                else:
                    print(f"Could not extract structure for {class_label}")

        elif hasattr(model, 'strategy') and model.strategy.lower() == 'ovo':
            # For OVO, check if the wrapper has pair_models
            if hasattr(trees[0], 'pair_models'):
                for (class1, class2), pair_model in trees[0].pair_models:
                    # Create filename for this pair
                    class1_name = model.class_labels[class1] if model.class_labels else f"class_{class1}"
                    class2_name = model.class_labels[class2] if model.class_labels else f"class_{class2}"
                    pair_filename = os.path.join(vis_dir, f"{unique_base_filename}_{class1_name}_vs_{class2_name}")

                    # Extract and visualize tree
                    tree_structure = extract_tree_structure(pair_model)
                    if tree_structure:
                        path = visualize_gp_tree(tree_structure, pair_filename, format)
                        visualization_paths.append(path)
                        print(f"Saved visualization for {class1_name} vs {class2_name} to {path}")
                    else:
                        print(f"Could not extract structure for {class1_name} vs {class2_name}")
            else:
                print("OVO strategy model doesn't have the expected structure")
                # Fall back to treating each tree individually
                for i, tree in enumerate(trees):
                    tree_structure = extract_tree_structure(tree)
                    if tree_structure:
                        tree_filename = os.path.join(vis_dir, f"{unique_base_filename}_tree_{i}")
                        path = visualize_gp_tree(tree_structure, tree_filename, format)
                        visualization_paths.append(path)
        else:
            # Direct or unknown strategy, visualize all trees separately
            for i, tree in enumerate(trees):
                tree_structure = extract_tree_structure(tree)
                if tree_structure:
                    tree_filename = os.path.join(vis_dir, f"{unique_base_filename}_tree_{i}")
                    path = visualize_gp_tree(tree_structure, tree_filename, format)
                    visualization_paths.append(path)
    else:
        # Binary model - only one tree to visualize
        tree_structure = extract_tree_structure(model.model)
        if tree_structure:
            full_filename = os.path.join(vis_dir, unique_base_filename)
            path = visualize_gp_tree(tree_structure, full_filename, format)
            visualization_paths.append(path)
            print(f"Saved binary classifier visualization to {path}")
        else:
            print("Could not extract structure for binary classifier")

    return visualization_paths


# Example usage:
if __name__ == "__main__":
    example_tree = """
    multiply(
      add(
        x1
        x2
      )
      subtract(
        x3
        x4
      )
    )
    """

    visualize_gp_tree(example_tree, 'example_tree')
    print("Tree visualization saved as example_tree.png")
