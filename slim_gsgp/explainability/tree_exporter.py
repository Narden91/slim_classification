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
Tree Exporter for SLIM Models.

Exports SLIM Individual trees to various formats:
- Interactive visualization (HTML) using Plotly
- Static visualization (SVG, PDF) using Plotly + Kaleido
- Formula files (text)

Cross-platform: Works on Windows and Linux without external binaries.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from slim_gsgp.explainability.formula_formatter import FormulaFormatter


# Modern color palette - vibrant professional colors
COLORS = {
    'root': '#2E86C1',       # Strong Blue
    'block': '#E74C3C',      # Red
    'operator': '#27AE60',   # Green
    'mutation': '#F39C12',   # Orange
    'terminal': '#8E44AD',   # Purple
    'constant': '#16A085',   # Teal
    'ms': '#BDC3C7',         # Gray
    'edge': '#34495E',       # Dark blue-gray
    'background': '#F8F9F9', # Off-white
}


class TreeExporter:
    """
    Exports SLIM Individual trees to files.
    
    Supports:
    - Visual formats: html (interactive), svg, pdf
    - Text formats: text (.txt)
    - Combined: all (exports all formats)
    
    Parameters
    ----------
    slim_version : str, optional
        The SLIM version string for formula formatting.
    """
    
    VISUAL_FORMATS = {"html", "svg", "pdf"}
    TEXT_FORMATS = {"text"}
    ALL_FORMATS = VISUAL_FORMATS | TEXT_FORMATS
    
    def __init__(self, slim_version: Optional[str] = None) -> None:
        """
        Initialize the TreeExporter.
        
        Parameters
        ----------
        slim_version : str, optional
            The SLIM version string.
        """
        self.slim_version = slim_version
        self.formatter = FormulaFormatter(slim_version)
        self._visualization_available = self._check_visualization_deps()
    
    def _check_visualization_deps(self) -> bool:
        """Check if visualization dependencies are available."""
        try:
            import plotly
            return True
        except ImportError:
            return False
    
    def _check_static_export_deps(self) -> bool:
        """Check if static export (SVG/PDF) dependencies are available."""
        try:
            import kaleido
            return True
        except ImportError:
            return False
    
    def _ensure_directory(self, path: str) -> None:
        """Ensure the output directory exists."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    def _build_tree_data(self, individual: Any) -> Tuple[List[Dict], List[Tuple[int, int]]]:
        """
        Build tree data structure from SLIM Individual.
        
        Returns
        -------
        tuple
            (nodes, edges) where nodes is list of node dicts and edges is list of (parent_idx, child_idx)
        """
        nodes: List[Dict] = []
        edges: List[Tuple[int, int]] = []
        
        def add_node(label: str, node_type: str, tooltip: str = "") -> int:
            """Add a node and return its index."""
            idx = len(nodes)
            nodes.append({
                'label': label,
                'type': node_type,
                'tooltip': tooltip or label,
            })
            return idx
        
        def process_structure(structure: Any, parent_idx: Optional[int] = None) -> int:
            """Recursively process tree structure."""
            
            if structure is None:
                idx = add_node("None", "constant")
                if parent_idx is not None:
                    edges.append((parent_idx, idx))
                return idx
            
            # Handle tuple (function node)
            if isinstance(structure, tuple):
                if len(structure) == 0:
                    idx = add_node("()", "constant")
                    if parent_idx is not None:
                        edges.append((parent_idx, idx))
                    return idx
                
                func = structure[0]
                if callable(func):
                    func_name = getattr(func, '__name__', 'f')
                else:
                    func_name = str(func)
                
                idx = add_node(func_name, "operator", f"Function: {func_name}")
                if parent_idx is not None:
                    edges.append((parent_idx, idx))
                
                for child in structure[1:]:
                    process_structure(child, idx)
                
                return idx
            
            # Handle list (mutation block structure)
            elif isinstance(structure, list):
                if len(structure) >= 3:
                    # Mutation block: [operator, tree(s), ms]
                    op_func = structure[0]
                    op_name = getattr(op_func, '__name__', 'mutation') if callable(op_func) else str(op_func)
                    
                    idx = add_node(op_name, "mutation", f"Mutation: {op_name}")
                    if parent_idx is not None:
                        edges.append((parent_idx, idx))
                    
                    # Add child trees (skip operator and ms)
                    for child in structure[1:-1]:
                        if hasattr(child, 'structure'):
                            process_structure(child.structure, idx)
                        else:
                            process_structure(child, idx)
                    
                    # Add mutation step value
                    ms_val = structure[-1]
                    ms_str = f"{float(ms_val):.4f}" if isinstance(ms_val, (int, float)) else str(ms_val)
                    ms_idx = add_node(f"ms={ms_str}", "ms", f"Mutation Step: {ms_str}")
                    edges.append((idx, ms_idx))
                else:
                    idx = add_node(str(structure)[:15], "constant")
                    if parent_idx is not None:
                        edges.append((parent_idx, idx))
                
                return idx
            
            # Handle Tree object
            elif hasattr(structure, 'structure'):
                return process_structure(structure.structure, parent_idx)
            
            # Handle callable (function reference)
            elif callable(structure):
                func_name = getattr(structure, '__name__', 'f')
                idx = add_node(func_name, "operator", f"Function: {func_name}")
                if parent_idx is not None:
                    edges.append((parent_idx, idx))
                return idx
            
            # Handle terminal
            else:
                terminal_str = str(structure)
                # Classify as variable or constant
                if terminal_str.startswith('x') and terminal_str[1:].isdigit():
                    node_type = "terminal"
                    tooltip = f"Variable: {terminal_str}"
                elif 'constant' in terminal_str.lower():
                    node_type = "constant"
                    tooltip = f"Constant: {terminal_str}"
                else:
                    node_type = "terminal"
                    tooltip = str(structure)
                
                # Truncate long labels
                if len(terminal_str) > 12:
                    terminal_str = terminal_str[:9] + "..."
                
                idx = add_node(terminal_str, node_type, tooltip)
                if parent_idx is not None:
                    edges.append((parent_idx, idx))
                return idx
        
        # Get version info
        version = getattr(individual, 'version', self.slim_version) or "SLIM"
        operator = "+" if version is None or "+" in version else "×"
        
        # Create root node
        root_idx = add_node(f"SLIM ({operator})", "root", f"SLIM Model\nVersion: {version}")
        
        # Add each block
        for i, block in enumerate(individual.collection):
            block_formula = "Formula unavailable"
            try:
                import textwrap
                raw_formula = self.formatter.format_block(block, i, "text")
                wrapped_formula = "<br>".join(textwrap.wrap(raw_formula, width=50))
                block_formula = f"<b>Formula:</b><br>{wrapped_formula}"
            except Exception:
                pass
                
            tooltip = f"<b>Semantic Block {i+1}</b><br><br>{block_formula}"
            block_idx = add_node(f"Block {i+1}", "block", tooltip)
            edges.append((root_idx, block_idx))
            
            if hasattr(block, 'structure'):
                process_structure(block.structure, block_idx)
        
        return nodes, edges
    
    def _compute_tree_layout(self, 
                             nodes: List[Dict], 
                             edges: List[Tuple[int, int]]) -> Tuple[List[float], List[float]]:
        """
        Compute Reingold-Tilford style tree layout.
        
        Returns
        -------
        tuple
            (x_positions, y_positions) for each node
        """
        n = len(nodes)
        if n == 0:
            return [], []
        
        # Build adjacency list
        children: Dict[int, List[int]] = {i: [] for i in range(n)}
        has_parent = set()
        for parent, child in edges:
            children[parent].append(child)
            has_parent.add(child)
        
        # Find root (node with no parent)
        roots = [i for i in range(n) if i not in has_parent]
        root = roots[0] if roots else 0
        
        # Compute levels using BFS
        levels: Dict[int, int] = {root: 0}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in children[node]:
                if child not in levels:
                    levels[child] = levels[node] + 1
                    queue.append(child)
        
        # Handle disconnected nodes
        for i in range(n):
            if i not in levels:
                levels[i] = 0
        
        # Group by level
        level_nodes: Dict[int, List[int]] = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        max_level = max(level_nodes.keys()) if level_nodes else 0
        
        # Compute subtree widths for better spacing
        subtree_width: Dict[int, float] = {}
        
        def compute_width(node: int) -> float:
            if not children[node]:
                subtree_width[node] = 1.0
                return 1.0
            width = sum(compute_width(c) for c in children[node])
            width = max(width, 1.0)
            subtree_width[node] = width
            return width
        
        compute_width(root)
        
        # Assign x positions using subtree widths
        x_pos: Dict[int, float] = {}
        
        def assign_x(node: int, left: float, right: float) -> None:
            x_pos[node] = (left + right) / 2
            if children[node]:
                total_width = sum(subtree_width.get(c, 1.0) for c in children[node])
                current_left = left
                for child in children[node]:
                    child_width = subtree_width.get(child, 1.0)
                    child_right = current_left + (right - left) * (child_width / total_width)
                    assign_x(child, current_left, child_right)
                    current_left = child_right
        
        assign_x(root, 0, 1)
        
        # Handle disconnected nodes
        for i in range(n):
            if i not in x_pos:
                x_pos[i] = 0.5
        
        # Convert to lists
        x = [x_pos[i] for i in range(n)]
        y = [-(levels.get(i, 0)) for i in range(n)]  # Negative so tree grows downward
        
        return x, y
    
    def export_visualization(self,
                            individual: Any,
                            output_path: str,
                            format: str = "html") -> Optional[str]:
        """
        Export tree visualization using Plotly.
        
        Parameters
        ----------
        individual : Individual
            The SLIM Individual to visualize.
        output_path : str
            Output file path (without extension).
        format : str
            Output format: html, svg, or pdf.
            
        Returns
        -------
        str or None
            Path to the generated file, or None if failed.
        """
        if not self._visualization_available:
            print("  Note: plotly not installed. Skipping visualization.")
            print("        Install with: pip install plotly kaleido")
            return None
        
        if format in {"svg", "pdf"} and not self._check_static_export_deps():
            print(f"  Note: kaleido not installed. Skipping {format} export.")
            print("        Install with: pip install kaleido")
            return None
        
        if not hasattr(individual, 'collection'):
            print("  Warning: Individual has no collection. Cannot visualize.")
            return None
        
        try:
            import plotly.graph_objects as go
            
            # Build tree data
            nodes, edges = self._build_tree_data(individual)
            
            if not nodes:
                print("  Warning: Empty tree. Cannot visualize.")
                return None
            
            # Compute layout
            x_pos, y_pos = self._compute_tree_layout(nodes, edges)
            
            # Scale positions for better visibility
            x_scaled = [x * 10 for x in x_pos]
            y_scaled = [y * 1.5 for y in y_pos]
            
            # Pre-calculate node sizes so edges can offset correctly
            node_sizes = []
            for i, n in enumerate(nodes):
                text_len = len(n['label'])
                node_type = n['type']
                # Base size + extra size per character
                if node_type in {'root', 'block'}:
                    node_sizes.append(max(60, 20 + text_len * 6))
                else:
                    node_sizes.append(max(45, 15 + text_len * 5))
                    
            # Create edge annotations (fancy directed arrows)
            edge_annotations = []
            for parent, child in edges:
                # standoff radius is approx half the size + padding
                target_standoff = (node_sizes[child] / 2) + 2
                source_standoff = (node_sizes[parent] / 2) + 2
                    
                edge_annotations.append(
                    dict(
                        ax=x_scaled[parent], ay=y_scaled[parent],
                        x=x_scaled[child], y=y_scaled[child],
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=1.5,
                        arrowcolor=COLORS['edge'],
                        standoff=target_standoff,
                        startstandoff=source_standoff
                    )
                )
            
            # Create node traces (one per type for legend)
            node_traces = []
            node_types = set(n['type'] for n in nodes)
            
            type_labels = {
                'root': 'SLIM Model',
                'block': 'Semantic Block',
                'operator': 'Operator',
                'mutation': 'Mutation Op',
                'terminal': 'Variable',
                'constant': 'Constant',
                'ms': 'Mutation Step',
            }
            
            symbol_map = {
                'root': 'square',
                'block': 'square',
                'operator': 'circle',
                'mutation': 'square',
                'terminal': 'circle',
                'constant': 'circle',
                'ms': 'circle-open'
            }
            
            for node_type in node_types:
                indices = [i for i, n in enumerate(nodes) if n['type'] == node_type]
                
                # Retrieve pre-calculated sizes
                sizes = [node_sizes[i] for i in indices]
                
                text_color = 'white' if node_type != 'ms' else 'black'
                
                node_trace = go.Scatter(
                    x=[x_scaled[i] for i in indices],
                    y=[y_scaled[i] for i in indices],
                    mode='markers+text',
                    marker=dict(
                        size=sizes,
                        color=COLORS.get(node_type, '#888888'),
                        line=dict(width=2, color='#2C3E50'),
                        symbol=symbol_map.get(node_type, 'circle'),
                        opacity=0.95
                    ),
                    text=[f"<b>{nodes[i]['label']}</b>" for i in indices],
                    textposition='middle center',
                    textfont=dict(
                        size=10,
                        color=text_color,
                        family='Arial Black, Arial, sans-serif',
                    ),
                    hovertext=[nodes[i]['tooltip'] for i in indices],
                    hoverinfo='text',
                    name=type_labels.get(node_type, node_type.capitalize()),
                    showlegend=True,
                )
                node_traces.append(node_trace)
            
            # Create figure
            version = getattr(individual, 'version', self.slim_version) or "SLIM"
            
            # Add general annotations
            general_annotations = [
                dict(
                    text=f"Blocks: {len(individual.collection)} | Nodes: {len(nodes)}",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.05,
                    showarrow=False,
                    font=dict(size=12, color='#666'),
                )
            ]
            
            fig = go.Figure(
                data=node_traces,
                layout=go.Layout(
                    title=dict(
                        text=f'<b>SLIM Tree Structure</b><br><sup>{version}</sup>',
                        font=dict(size=20, family='Arial'),
                        x=0.5,
                    ),
                    showlegend=True,
                    legend=dict(
                        x=1.02,
                        y=1,
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='#ccc',
                        borderwidth=1,
                    ),
                    hovermode='closest',
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        showline=False,
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        showline=False,
                    ),
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor='white',
                    margin=dict(l=40, r=150, t=80, b=40),
                    annotations=general_annotations + edge_annotations,
                )
            )
            
            # Ensure directory exists
            self._ensure_directory(output_path)
            
            # Save based on format
            final_path = f"{output_path}.{format}"
            
            if format == "html":
                fig.write_html(final_path, include_plotlyjs='cdn')
            else:
                # Use kaleido for static exports
                fig.write_image(final_path, width=1200, height=800, scale=2)
            
            return final_path
            
        except Exception as e:
            print(f"  Warning: Failed to create visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def export_formula_text(self,
                           individual: Any,
                           output_path: str,
                           include_metadata: bool = True) -> Optional[str]:
        """
        Export formula as plain text file.
        
        Parameters
        ----------
        individual : Individual
            The SLIM Individual to export.
        output_path : str
            Output file path.
        include_metadata : bool
            Whether to include metadata header.
            
        Returns
        -------
        str or None
            Path to the generated file, or None if failed.
        """
        try:
            # Update formatter version
            if hasattr(individual, 'version'):
                self.formatter.slim_version = individual.version
                self.formatter._operator = self.formatter._get_operator_from_version(individual.version)
                self.formatter._uses_sigmoid = self.formatter._get_sigmoid_from_version(individual.version)
            
            formula = self.formatter.to_text(individual)
            summary = self.formatter.get_summary(individual)
            
            self._ensure_directory(output_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if include_metadata:
                    f.write("=" * 60 + "\n")
                    f.write("SLIM Model Formula Export\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Version: {summary.get('version', 'N/A')}\n")
                    f.write(f"Operator: {summary.get('operator', 'N/A')}\n")
                    f.write(f"Blocks: {summary.get('n_blocks', 'N/A')}\n")
                    f.write(f"Nodes: {summary.get('n_nodes', 'N/A')}\n")
                    f.write(f"Depth: {summary.get('depth', 'N/A')}\n")
                    f.write("\n" + "-" * 60 + "\n\n")
                
                f.write("Formula:\n")
                f.write(formula + "\n\n")
                
                if include_metadata and hasattr(individual, 'collection'):
                    f.write("-" * 60 + "\n")
                    f.write("Block Details:\n")
                    f.write("-" * 60 + "\n\n")
                    
                    for i, block in enumerate(individual.collection):
                        f.write(f"Block {i+1}:\n")
                        block_formula = self.formatter.format_block(block, i, "text")
                        f.write(f"  {block_formula}\n\n")
            
            return output_path
            
        except Exception as e:
            print(f"  Warning: Failed to export text formula: {str(e)}")
            return None
    
    def export(self,
               individual: Any,
               output_dir: str,
               format: str = "all",
               filename: str = "final_tree",
               seed: Optional[int] = None,
               verbose: bool = True) -> Dict[str, Optional[str]]:
        """
        Export the SLIM Individual in the specified format(s).
        
        Parameters
        ----------
        individual : Individual
            The SLIM Individual to export.
        output_dir : str
            Output directory.
        format : str
            Output format: html, svg, pdf, text, or all.
        filename : str
            Base filename for exports.
        seed : int, optional
            Seed to include in filename.
        verbose : bool
            Whether to print progress messages.
            
        Returns
        -------
        dict
            Dictionary mapping format names to file paths (or None if failed).
        """
        results: Dict[str, Optional[str]] = {}
        
        # Add seed to filename if provided
        if seed is not None:
            filename = f"{filename}_seed{seed}"
        
        # Determine which formats to export
        if format.lower() == "all":
            formats_to_export = self.ALL_FORMATS.copy()
        else:
            formats_to_export = {format.lower()}
        
        # Check if visual formats are requested and dependencies are available
        visual_formats_requested = formats_to_export & self.VISUAL_FORMATS
        if visual_formats_requested and not self._visualization_available:
            if verbose:
                print("  Note: plotly not installed. Skipping visual exports.")
                print("        Install with: pip install plotly kaleido")
            formats_to_export = formats_to_export - self.VISUAL_FORMATS
        
        # Export visual formats
        for fmt in formats_to_export & self.VISUAL_FORMATS:
            output_path = os.path.join(output_dir, filename)
            result = self.export_visualization(individual, output_path, fmt)
            results[fmt] = result
            if verbose and result:
                print(f"  Tree visualization saved to: {result}")
        
        # Export text format
        if "text" in formats_to_export:
            output_path = os.path.join(output_dir, f"{filename}.txt")
            result = self.export_formula_text(individual, output_path)
            results["text"] = result
            if verbose and result:
                print(f"  Formula (text) saved to: {result}")
        
        return results
    
    def print_formula(self, individual: Any) -> None:
        """
        Print the formula to console.
        
        Parameters
        ----------
        individual : Individual
            The SLIM Individual to print.
        """
        # Update formatter version
        if hasattr(individual, 'version'):
            self.formatter.slim_version = individual.version
            self.formatter._operator = self.formatter._get_operator_from_version(individual.version)
            self.formatter._uses_sigmoid = self.formatter._get_sigmoid_from_version(individual.version)
        
        summary = self.formatter.get_summary(individual)
        formula = self.formatter.to_text(individual)
        
        print("\n" + "=" * 60)
        print("SLIM Final Model")
        print("=" * 60)
        print(f"Version: {summary.get('version', 'N/A')}")
        print(f"Blocks: {summary.get('n_blocks', 'N/A')}")
        print(f"Nodes: {summary.get('n_nodes', 'N/A')}")
        print(f"Depth: {summary.get('depth', 'N/A')}")
        print("-" * 60)
        print(f"Formula:\n{formula}")
        print("=" * 60 + "\n")
