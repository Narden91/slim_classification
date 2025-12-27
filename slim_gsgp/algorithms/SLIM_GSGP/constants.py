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
Constants for SLIM_GSGP algorithm.
"""

# Semantics clamping bounds to prevent numerical overflow
SEMANTICS_CLAMP_MIN: float = -1e12
SEMANTICS_CLAMP_MAX: float = 1e12

# Default algorithm parameters
DEFAULT_POPULATION_SIZE: int = 100
DEFAULT_MAX_DEPTH: int = 17
DEFAULT_N_ELITES: int = 1
DEFAULT_N_ITERATIONS: int = 20

# Mutation probabilities
DEFAULT_MUTATION_PROB: float = 1.0
DEFAULT_CROSSOVER_PROB: float = 0.0
DEFAULT_INFLATE_PROB: float = 0.3
DEFAULT_DEFLATE_PROB: float = 0.7

# Operator types
OPERATOR_SUM: str = "sum"
OPERATOR_PROD: str = "prod"
