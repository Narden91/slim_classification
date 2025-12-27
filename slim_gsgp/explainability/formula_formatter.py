# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Formula Formatter for SLIM Models.

Converts SLIM Individual tree structures into human-readable mathematical formulas.
"""

from typing import Any, Dict, List, Optional, Tuple, Union


class FormulaFormatter:
    """
    Formats SLIM Individual tree structures into readable mathematical formulas.
    
    Supports multiple output formats:
    - text: Human-readable plain text
    - latex: LaTeX-formatted mathematical notation
    
    Parameters
    ----------
    slim_version : str, optional
        The SLIM version string (e.g., "SLIM+SIG2") to determine operator and function types.
    """
    
    # Mapping of internal function names to display names
    FUNCTION_DISPLAY_NAMES: Dict[str, str] = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "protected_div": "/",
        "mean_": "mean",
        "sqrt": "sqrt",
        "log": "log",
        "exp": "exp",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
    }
    
    # LaTeX equivalents
    LATEX_FUNCTION_NAMES: Dict[str, str] = {
        "add": "+",
        "sub": "-",
        "mul": r"\cdot",
        "div": r"\div",
        "protected_div": r"\div",
        "mean_": r"\text{mean}",
        "sqrt": r"\sqrt",
        "log": r"\log",
        "exp": r"\exp",
        "sin": r"\sin",
        "cos": r"\cos",
        "tan": r"\tan",
    }
    
    def __init__(self, slim_version: Optional[str] = None) -> None:
        """
        Initialize the FormulaFormatter.
        
        Parameters
        ----------
        slim_version : str, optional
            The SLIM version string to determine formatting options.
        """
        self.slim_version = slim_version
        self._operator = self._get_operator_from_version(slim_version)
        self._uses_sigmoid = self._get_sigmoid_from_version(slim_version)
    
    def _get_operator_from_version(self, version: Optional[str]) -> str:
        """Determine the aggregation operator from SLIM version."""
        if version is None:
            return "+"
        return "*" if "*" in version else "+"
    
    def _get_sigmoid_from_version(self, version: Optional[str]) -> bool:
        """Determine if sigmoid is used from SLIM version."""
        if version is None:
            return False
        return "SIG" in version.upper()
    
    def _format_node(self, node: Any, format_type: str = "text") -> str:
        """
        Format a single tree node into a string.
        
        Parameters
        ----------
        node : Any
            The tree node to format (can be tuple, callable, or terminal).
        format_type : str
            Output format: "text" or "latex".
            
        Returns
        -------
        str
            Formatted string representation of the node.
        """
        if node is None:
            return ""
        
        # Handle tuple (function node with children)
        if isinstance(node, tuple):
            return self._format_function_node(node, format_type)
        
        # Handle callable (function reference)
        if callable(node):
            func_name = getattr(node, '__name__', 'f')
            return self._get_function_display_name(func_name, format_type)
        
        # Handle terminal (variable or constant)
        return self._format_terminal(node, format_type)
    
    def _format_function_node(self, node: Tuple, format_type: str) -> str:
        """Format a function node (tuple with function and arguments)."""
        if len(node) == 0:
            return ""
        
        func_name = node[0]
        
        # Get function display name
        if callable(func_name):
            func_name = getattr(func_name, '__name__', 'f')
        
        display_name = self._get_function_display_name(func_name, format_type)
        
        # Format children
        children = [self._format_node(child, format_type) for child in node[1:]]
        
        # Handle binary operators specially for infix notation
        if func_name in ("add", "sub", "mul", "div", "protected_div") and len(children) == 2:
            return self._format_binary_op(display_name, children[0], children[1], format_type)
        
        # Default function notation: func(arg1, arg2, ...)
        if format_type == "latex":
            return f"{display_name}({', '.join(children)})"
        return f"{func_name}({', '.join(children)})"
    
    def _format_binary_op(self, op: str, left: str, right: str, format_type: str) -> str:
        """Format a binary operation in infix notation."""
        if format_type == "latex":
            if op in ("+", "-"):
                return f"({left} {op} {right})"
            elif op == r"\cdot":
                return f"({left} {op} {right})"
            elif op == r"\div":
                return f"\\frac{{{left}}}{{{right}}}"
        else:
            return f"({left} {op} {right})"
        return f"({left} {op} {right})"
    
    def _format_terminal(self, terminal: Any, format_type: str) -> str:
        """Format a terminal node (variable or constant)."""
        terminal_str = str(terminal)
        
        # Handle variable names (e.g., x_0, x_1)
        if terminal_str.startswith("x_") or terminal_str.startswith("x"):
            if format_type == "latex":
                # Convert x_0 to x_{0} for LaTeX subscript
                if "_" in terminal_str:
                    parts = terminal_str.split("_")
                    return f"{parts[0]}_{{{parts[1]}}}"
                return terminal_str
            return terminal_str
        
        # Handle numeric constants
        try:
            val = float(terminal_str)
            if format_type == "latex":
                return f"{val:.4g}"
            return f"{val:.4g}"
        except (ValueError, TypeError):
            pass
        
        return terminal_str
    
    def _get_function_display_name(self, func_name: str, format_type: str) -> str:
        """Get the display name for a function based on format type."""
        if format_type == "latex":
            return self.LATEX_FUNCTION_NAMES.get(func_name, func_name)
        return self.FUNCTION_DISPLAY_NAMES.get(func_name, func_name)
    
    def format_block(self, block: Any, block_index: int, format_type: str = "text") -> str:
        """
        Format a single SLIM block (tree) into a string.
        
        Parameters
        ----------
        block : Any
            The block structure to format.
        block_index : int
            The index of this block (0 for base tree).
        format_type : str
            Output format: "text" or "latex".
            
        Returns
        -------
        str
            Formatted string representation of the block.
        """
        # Check if it's a Tree object with structure attribute
        if hasattr(block, 'structure'):
            structure = block.structure
        else:
            structure = block
        
        # Handle base GP tree (tuple)
        if isinstance(structure, tuple):
            return self._format_node(structure, format_type)
        
        # Handle SLIM mutation block (list with [operator, tree(s), ms])
        if isinstance(structure, list):
            return self._format_mutation_block(structure, format_type)
        
        return str(structure)
    
    def _format_mutation_block(self, structure: List, format_type: str) -> str:
        """Format a SLIM mutation block."""
        if len(structure) < 3:
            return str(structure)
        
        operator_func = structure[0]
        ms = structure[-1]  # mutation step is last element
        
        # Get operator name
        op_name = getattr(operator_func, '__name__', 'f') if callable(operator_func) else str(operator_func)
        
        # Format mutation step
        ms_str = f"{float(ms):.4g}" if isinstance(ms, (int, float)) else str(ms)
        
        # One tree mutation: [operator, tree, ms]
        if len(structure) == 3:
            tree = structure[1]
            tree_str = self._format_node(tree.structure if hasattr(tree, 'structure') else tree, format_type)
            
            if self._uses_sigmoid:
                if format_type == "latex":
                    return f"{ms_str} \\cdot \\sigma({tree_str})"
                return f"{ms_str} * sigmoid({tree_str})"
            else:
                if format_type == "latex":
                    return f"{ms_str} \\cdot f({tree_str})"
                return f"{ms_str} * f({tree_str})"
        
        # Two tree mutation: [operator, tree1, tree2, ms]
        elif len(structure) == 4:
            tree1 = structure[1]
            tree2 = structure[2]
            tree1_str = self._format_node(tree1.structure if hasattr(tree1, 'structure') else tree1, format_type)
            tree2_str = self._format_node(tree2.structure if hasattr(tree2, 'structure') else tree2, format_type)
            
            if format_type == "latex":
                return f"{ms_str} \\cdot \\sigma({tree1_str} - {tree2_str})"
            return f"{ms_str} * sigmoid({tree1_str} - {tree2_str})"
        
        return str(structure)
    
    def format_individual(self, individual: Any, format_type: str = "text") -> str:
        """
        Format a complete SLIM Individual into a mathematical formula.
        
        Parameters
        ----------
        individual : Individual
            The SLIM Individual to format.
        format_type : str
            Output format: "text" or "latex".
            
        Returns
        -------
        str
            Complete formula representation of the individual.
        """
        if not hasattr(individual, 'collection'):
            raise ValueError("Individual does not have a collection attribute. "
                           "Ensure reconstruct=True was used during evolution.")
        
        # Update version if available
        if hasattr(individual, 'version'):
            self.slim_version = individual.version
            self._operator = self._get_operator_from_version(individual.version)
            self._uses_sigmoid = self._get_sigmoid_from_version(individual.version)
        
        # Format each block
        blocks = []
        for i, block in enumerate(individual.collection):
            block_str = self.format_block(block, i, format_type)
            blocks.append(block_str)
        
        # Join blocks with operator
        if format_type == "latex":
            op = " + " if self._operator == "+" else r" \cdot "
        else:
            op = f" {self._operator} "
        
        formula = op.join(blocks)
        
        # Wrap in equation for latex
        if format_type == "latex":
            return f"y = {formula}"
        
        return f"y = {formula}"
    
    def to_text(self, individual: Any) -> str:
        """
        Convert Individual to plain text formula.
        
        Parameters
        ----------
        individual : Individual
            The SLIM Individual to convert.
            
        Returns
        -------
        str
            Plain text formula.
        """
        return self.format_individual(individual, "text")
    
    def to_latex(self, individual: Any) -> str:
        """
        Convert Individual to LaTeX formula.
        
        Parameters
        ----------
        individual : Individual
            The SLIM Individual to convert.
            
        Returns
        -------
        str
            LaTeX-formatted formula.
        """
        return self.format_individual(individual, "latex")
    
    def get_summary(self, individual: Any) -> Dict[str, Any]:
        """
        Get a summary of the Individual's structure.
        
        Parameters
        ----------
        individual : Individual
            The SLIM Individual to summarize.
            
        Returns
        -------
        dict
            Summary containing blocks count, nodes count, depth, etc.
        """
        summary = {
            "n_blocks": len(individual.collection) if hasattr(individual, 'collection') else 0,
            "n_nodes": getattr(individual, 'nodes_count', 0),
            "depth": getattr(individual, 'depth', 0),
            "version": getattr(individual, 'version', self.slim_version),
            "operator": self._operator,
            "uses_sigmoid": self._uses_sigmoid,
        }
        return summary
