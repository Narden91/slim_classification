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
This script sets up the configuration dictionaries for the execution of the GSGP algorithm
"""
import torch
from slim_gsgp.initializers.initializers import rhh, grow, full
from slim_gsgp.selection.selection_algorithms import tournament_selection_min, tournament_selection_max

from slim_gsgp.algorithms.GSGP.operators.crossover_operators import geometric_crossover
from slim_gsgp.algorithms.GSGP.operators.mutators import standard_geometric_mutation
from slim_gsgp.evaluators.fitness_functions import *
from slim_gsgp.utils.utils import (get_best_min, get_best_max,
                                   protected_div)

# Device configuration - auto-detect GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    """Get the configured device for tensor operations."""
    return DEVICE

def set_device(device):
    """Set the device for tensor operations."""
    global DEVICE
    if isinstance(device, str):
        DEVICE = torch.device(device)
    else:
        DEVICE = device

# Define functions and constants
FUNCTIONS = {
    'add': {'function': torch.add, 'arity': 2},
    'subtract': {'function': torch.sub, 'arity': 2},
    'multiply': {'function': torch.mul, 'arity': 2},
    'divide': {'function': protected_div, 'arity': 2}
}

# Constants with device support
def _create_constant(value):
    """Create a constant tensor on the configured device."""
    def constant_fn(_):
        return torch.tensor(value, device=DEVICE)
    return constant_fn

CONSTANTS = {
    'constant_2': _create_constant(2.0),
    'constant_3': _create_constant(3.0),
    'constant_4': _create_constant(4.0),
    'constant_5': _create_constant(5.0),
    'constant__1': _create_constant(-1.0)
}

# Set parameters
settings_dict = {"p_test": 0.2}

# GSGP solve parameters
gsgp_solve_parameters = {
    "run_info": None,
    "reconstruct": False,
    "n_jobs": 1,
    "n_iter": 1000,
    "elitism": True,
    "n_elites": 1,
    "log": 1,
    "verbose": 1,
    "ffunction": "rmse",
    "test_elite": True
}

# GSGP parameters
gsgp_parameters = {
    "selector": tournament_selection_min(2),
    "crossover": geometric_crossover,
    "mutator": standard_geometric_mutation,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min,
    "pop_size": 100,
    "p_xo": 0.0,
    "seed": 74,
    "initializer": "rhh"
}

gsgp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0.2,
    "init_depth": 8
}

fitness_function_options = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "mae_int": mae_int,
    "signed_errors": signed_errors,
    "binary_cross_entropy": binary_cross_entropy
}

initializer_options = {
    "rhh": rhh,
    "grow": grow,
    "full": full
}