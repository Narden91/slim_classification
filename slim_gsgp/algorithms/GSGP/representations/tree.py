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
Tree Class for Genetic Programming using PyTorch.
"""

import torch
from slim_gsgp.algorithms.GP.representations.tree_utils import flatten, tree_depth
from slim_gsgp.algorithms.GSGP.representations.tree_utils import apply_tree, nested_depth_calculator, nested_nodes_calculator
from slim_gsgp.algorithms.GSGP.operators.crossover_operators import geometric_crossover


class Tree:
    """
    Tree class implementation for representing tree structures in GSGP.

    Attributes
    ----------
    structure : tuple or str
        The tree structure, either as a tuple or a list of pointers.
    FUNCTIONS : dict
        Dictionary of allowed functions in the tree.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    depth : int
        The maximum depth of the tree structure.
    nodes : int
        The total number of nodes in the tree.
    train_semantics : torch.Tensor
        The training semantics associated with the tree.
    test_semantics : torch.Tensor
        The testing semantics associated with the tree.
    fitness : float or None
        The fitness value of the tree. Defaults to None.
    test_fitness : float or None
        The fitness value of the tree during testing. Defaults to None.
    """
    FUNCTIONS = None
    TERMINALS = None
    CONSTANTS = None

    def __init__(self, structure, train_semantics, test_semantics, reconstruct):
        """
        Initialize the Tree object with its structure and semantics.

        Parameters
        ----------
        structure : tuple or str
            The tree structure, either as a tuple or a list of pointers.
        train_semantics : torch.Tensor
            The training semantics associated with the tree.
        test_semantics : torch.Tensor
            The testing semantics associated with the tree.
        reconstruct : bool
            Indicates if the tree's structure should be stored for later reconstruction.

        """
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS

        if structure is not None and reconstruct:
            self.structure = (
                structure  # either repr_ from gp(tuple) or list of pointers
            )
        self.train_semantics = train_semantics
        self.test_semantics = test_semantics

        # if the tree is a base (gp) tree
        if isinstance(structure, tuple):
            self.depth = tree_depth(Tree.FUNCTIONS)(structure)
            self.nodes = len(list(flatten(structure)))

        # if it's not a base tree, calculate the depth via the nested depth function
        elif reconstruct:
            self.depth = nested_depth_calculator(
                self.structure[0],
                [tree.depth for tree in self.structure[1:] if isinstance(tree, Tree)],
            )
            self.nodes = nested_nodes_calculator(
                self.structure[0],
                [tree.nodes for tree in self.structure[1:] if isinstance(tree, Tree)],
            )

        self.fitness = None
        self.test_fitness = None

    def calculate_semantics(self, inputs, testing=False, logistic=False):
        """
        Calculate the semantics for the tree.

        Semantics are stored as an attribute in their respective objects.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for calculating semantics.
        testing : bool, optional
            Indicates if the calculation is for testing semantics. Defaults to `False`.
        logistic : bool, optional
            Indicates if a logistic (Sigmoid) function should be applied. Defaults to `False`.

        Returns
        -------
        None
        """
        # if testing
        if testing and self.test_semantics is None:
            # if the structure is a base (gp) tree, call apply_tree in order to obtain the semantics
            if isinstance(self.structure, tuple):
                self.test_semantics = (
                    torch.sigmoid(apply_tree(self, inputs))
                    if logistic
                    else apply_tree(self, inputs)
                )
            else:
                # otherwise, the semantics are computed by calling the operator (crossover or mutation)
                # with the remaindin structure of the individual
                self.test_semantics = self.structure[0](
                    *self.structure[1:], testing=True
                )
        # if training
        elif self.train_semantics is None:
            # if the structure is a base (gp) tree, call apply_tree in order to obtain the semantics
            if isinstance(self.structure, tuple):
                self.train_semantics = (
                    torch.sigmoid(apply_tree(self, inputs))
                    if logistic
                    else apply_tree(self, inputs)
                )
            else:
                # otherwise, the semantics are computed by calling the operator (crossover or mutation)
                # with the remaining structure of the individual
                self.train_semantics = self.structure[0](
                    *self.structure[1:], testing=False
                )

    def evaluate(self, ffunction, y, testing=False, X=None):
        """
        Evaluate the tree using a fitness function.

        During the evolution process, stores the fitness as an attribute. If evaluating with new data, fitness is
        returned as a float.

        Parameters
        ----------
        ffunction : callable
            Fitness function to evaluate the individual.
        y : torch.Tensor
            Expected output (target) values as a torch tensor.
        testing : bool, optional
            Indicates if the evaluation is for testing semantics. Defaults to `False`.
        X : torch.Tensor, optional
            Input data used for calculation. Optional inside the evolution process as only the semantics are needed,
            but necessary outside of it.

        Returns
        -------
        None or float
            Returns nothing if no new data is provided, as the training and testing fitness is stored as an attribute.
        float
            The fitness value of the tree when evaluated with new data.
        """
        # if data is provided
        if X is not None:
            # obtaining the semantics of the individual either by calling apply_tree (if it's a base (gp) tree) or
            # by calling the operator with the remaining structure of the individual
            semantics = apply_tree(self, X) if isinstance(self.structure, tuple) \
                else self.structure[0](*self.structure[1:], testing=False)
            return ffunction(y, semantics)
        # if data is not provided
        else:
            if testing:
                self.test_fitness = ffunction(y, self.test_semantics)
            else:
                self.fitness = ffunction(y, self.train_semantics)

    def predict(self, data):
        """
        Predict the output for the given input data using the model's structure.

        Uses recursive logic to call itself on the structure of the tree until arriving at a basic tuple structure, and
        then applies the necessary operations to arrive at the final result for the whole tree.

        Parameters
        ----------
        data : torch.Tensor
            The input data to predict.

        Returns
        -------
        torch.Tensor
            The predicted output for the input data.

        Notes
        -----
        The prediction process depends on the structure of the model:

        - If `self.structure` is a tuple, the `apply_tree` function is used for prediction.
        - If `self.structure` is a list, the first element is assumed to be a function that
          combines the predictions of multiple base trees (contained in the list) along with
          additional parameters (floats) extracted from the list. The base trees are instances
          of the `Tree` class, and their individual predictions are passed to the combining
          function along with any extracted parameters.

        The combining function is called with the predictions of the base trees and the
        extracted parameters, along with `testing` set to False and `new_data` set to True.
        """

        # seeing if the tree has the structure attribute, if not reconstruct was set to false during learning
        if not hasattr(self, "structure"):
            raise Exception("If reconstruct was set to False, .predict() is not available.")

        # Evaluate iteratively to avoid recursion-limit failures on deep evolved trees.
        # GSGP individuals are DAG-like compositions of previous trees; this computes
        # node outputs in post-order without recursive Python calls.
        memo = {}
        processing = set()
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()
            node_id = id(node)

            if node_id in memo:
                continue

            if not hasattr(node, "structure"):
                raise Exception("If reconstruct was set to False, .predict() is not available.")

            if expanded:
                structure = node.structure
                if isinstance(structure, tuple):
                    memo[node_id] = apply_tree(node, data)
                else:
                    op = structure[0]
                    ms_values = [ms for ms in structure[1:] if isinstance(ms, float)]
                    base_trees = [tree for tree in structure if isinstance(tree, Tree)]
                    base_outputs = [memo[id(tree)] for tree in base_trees]

                    if op == geometric_crossover:
                        memo[node_id] = op(
                            *base_outputs[:-1],
                            torch.sigmoid(base_outputs[-1]),
                            testing=False,
                            new_data=True,
                        )
                    else:
                        memo[node_id] = op(
                            *[
                                torch.sigmoid(output) if i != 0 else output
                                for i, output in enumerate(base_outputs)
                            ],
                            *ms_values,
                            testing=False,
                            new_data=True,
                        )

                processing.discard(node_id)
                continue

            if node_id in processing:
                raise RuntimeError("Cycle detected in GSGP tree structure during prediction.")

            processing.add(node_id)
            stack.append((node, True))

            structure = node.structure
            if isinstance(structure, tuple):
                continue

            child_trees = [tree for tree in structure if isinstance(tree, Tree)]
            for child in reversed(child_trees):
                child_id = id(child)
                if child_id not in memo:
                    stack.append((child, False))

        return memo[id(self)]



