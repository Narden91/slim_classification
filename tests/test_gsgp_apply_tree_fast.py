import torch

from slim_gsgp.algorithms.GP.representations.tree_utils import _execute_tree as gp_execute_tree
from slim_gsgp.algorithms.GSGP.representations.tree import Tree as GSGP_Tree
from slim_gsgp.algorithms.GSGP.representations.tree_utils import apply_tree


def _init_minimal_primitives(device: torch.device):
    # Minimal primitive set matching the codebase conventions.
    functions = {
        "add": {"function": torch.add, "arity": 2},
        "subtract": {"function": torch.sub, "arity": 2},
        "multiply": {"function": torch.mul, "arity": 2},
    }
    terminals = {"x0": 0, "x1": 1}

    # constant returns a tensor on the right device
    def _c1(_):
        return torch.tensor(1.0, device=device)

    constants = {"c1": _c1}

    GSGP_Tree.FUNCTIONS = functions
    GSGP_Tree.TERMINALS = terminals
    GSGP_Tree.CONSTANTS = constants

    return functions, terminals, constants


def test_gsgp_apply_tree_matches_gp_execute_tree_cpu():
    device = torch.device("cpu")
    functions, terminals, constants = _init_minimal_primitives(device)

    # (add (multiply x0 c1) (subtract x1 c1))
    structure = ("add", ("multiply", "x0", "c1"), ("subtract", "x1", "c1"))

    X = torch.tensor([[2.0, 5.0], [3.0, 7.0]], device=device)
    tree = GSGP_Tree(structure, train_semantics=None, test_semantics=None, reconstruct=True)

    out_apply = apply_tree(tree, X)
    out_exec = gp_execute_tree(structure, X, functions, terminals, constants)

    assert torch.allclose(out_apply, out_exec)


def test_gsgp_apply_tree_matches_gp_execute_tree_cuda_if_available():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    functions, terminals, constants = _init_minimal_primitives(device)

    structure = ("add", ("multiply", "x0", "c1"), ("subtract", "x1", "c1"))

    X = torch.tensor([[2.0, 5.0], [3.0, 7.0]], device=device)
    tree = GSGP_Tree(structure, train_semantics=None, test_semantics=None, reconstruct=True)

    out_apply = apply_tree(tree, X)
    out_exec = gp_execute_tree(structure, X, functions, terminals, constants)

    assert out_apply.is_cuda
    assert out_exec.is_cuda
    assert torch.allclose(out_apply, out_exec)
