import re
from graphviz import Digraph


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


def visualize_gp_tree(tree_str, filename='gp_tree', format='png'):
    """
    Create a visual representation of a GP tree from its string representation.

    Parameters:
    -----------
    tree_str : str
        String representation of the GP tree
    filename : str
        Output filename (without extension)
    format : str
        Output format ('png', 'svg', 'pdf', etc.)

    Returns:
    --------
    str
        Path to the generated visualization file
    """
    tree = parse_tree_string(tree_str)
    dot = create_graphviz_tree(tree, filename, format)
    return f"{filename}.{format}"


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