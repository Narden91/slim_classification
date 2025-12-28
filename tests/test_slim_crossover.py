import torch
import tempfile
from pathlib import Path

from slim_gsgp.algorithms.SLIM_GSGP.operators.crossover_operators import (
    one_point_block_crossover,
    uniform_block_crossover,
)
from slim_gsgp.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from slim_gsgp.algorithms.SLIM_GSGP.operators.mutators import inflate_mutation, deflate_mutation
from slim_gsgp.config.slim_config import FUNCTIONS, CONSTANTS
from slim_gsgp.config.slim_config import set_device
from slim_gsgp.datasets.data_loader import load_breast_cancer
from slim_gsgp.evaluators.fitness_functions import rmse
from slim_gsgp.initializers.initializers import rhh
from slim_gsgp.selection.selection_algorithms import tournament_selection_min
from slim_gsgp.utils.utils import get_terminals, generate_random_uniform, get_best_min
from slim_gsgp.main_slim import slim


def _make_tiny_problem(seed: int = 0):
    # Ensure tests are deterministic and do not depend on GPU availability.
    set_device("cpu")
    X, y = load_breast_cancer(X_y=True)
    # Keep it small/fast
    X = X[:80]
    y = y[:80]
    return X, y


def _make_optimizer(*, crossover, seed: int, reconstruct: bool):
    X, y = _make_tiny_problem(seed)

    pi_init = {
        "FUNCTIONS": FUNCTIONS,
        "TERMINALS": get_terminals(X),
        "CONSTANTS": CONSTANTS,
        "p_c": 0.2,
        "init_depth": 2,
        "init_pop_size": 6,
    }

    ms = generate_random_uniform(0.1, 0.5)

    inflate = inflate_mutation(
        FUNCTIONS=pi_init["FUNCTIONS"],
        TERMINALS=pi_init["TERMINALS"],
        CONSTANTS=pi_init["CONSTANTS"],
        two_trees=True,
        operator="sum",
        sig=True,
    )

    optimizer = SLIM_GSGP(
        pi_init=pi_init,
        initializer=rhh,
        selector=tournament_selection_min(2),
        inflate_mutator=inflate,
        deflate_mutator=deflate_mutation,
        ms=ms,
        crossover=crossover,
        find_elit_func=get_best_min,
        p_xo=1.0,
        p_m=0.0,
        p_inflate=0.5,
        p_deflate=0.5,
        pop_size=6,
        seed=seed,
        operator="sum",
        copy_parent=True,
        two_trees=True,
        settings_dict=None,
    )

    return optimizer, X, y, reconstruct


def test_one_point_block_crossover_direct_properties():
    optimizer, X, y, reconstruct = _make_optimizer(crossover=one_point_block_crossover, seed=1, reconstruct=True)
    optimizer.solve(
        X_train=X,
        X_test=None,
        y_train=y,
        y_test=None,
        curr_dataset="bc",
        run_info=["SLIM+SIG2", "test", "bc"],
        n_iter=0,
        elitism=False,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        ffunction=rmse,
        max_depth=20,
        n_elites=1,
        reconstruct=True,
        n_jobs=1,
    )

    # pick two distinct individuals from initial population
    p1 = optimizer.population.population[0]
    p2 = optimizer.population.population[1]
    child = one_point_block_crossover(p1, p2, reconstruct=True)

    assert child.size >= 1
    assert child.train_semantics.shape[0] == child.size
    assert child.train_semantics.shape[1] == X.shape[0]
    assert child.nodes_count >= 1
    assert child.depth >= 1


def test_uniform_block_crossover_direct_properties():
    optimizer, X, y, reconstruct = _make_optimizer(crossover=uniform_block_crossover, seed=2, reconstruct=True)
    optimizer.solve(
        X_train=X,
        X_test=None,
        y_train=y,
        y_test=None,
        curr_dataset="bc",
        run_info=["SLIM+SIG2", "test", "bc"],
        n_iter=0,
        elitism=False,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        ffunction=rmse,
        max_depth=20,
        n_elites=1,
        reconstruct=True,
        n_jobs=1,
    )

    p1 = optimizer.population.population[0]
    p2 = optimizer.population.population[1]
    child = uniform_block_crossover(p1, p2, reconstruct=True)

    assert child.size == max(p1.size, p2.size)
    assert child.train_semantics.shape[0] == child.size
    assert child.train_semantics.shape[1] == X.shape[0]


def test_solve_runs_with_one_point_crossover_reconstruct_true():
    optimizer, X, y, _ = _make_optimizer(crossover=one_point_block_crossover, seed=3, reconstruct=True)
    optimizer.solve(
        X_train=X,
        X_test=None,
        y_train=y,
        y_test=None,
        curr_dataset="bc",
        run_info=["SLIM+SIG2", "test", "bc"],
        n_iter=2,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        ffunction=rmse,
        max_depth=20,
        n_elites=1,
        reconstruct=True,
        n_jobs=1,
    )

    assert optimizer.elite is not None
    assert len(optimizer.population.population) == optimizer.pop_size


def test_solve_runs_with_uniform_crossover_reconstruct_false():
    optimizer, X, y, _ = _make_optimizer(crossover=uniform_block_crossover, seed=4, reconstruct=False)
    optimizer.solve(
        X_train=X,
        X_test=None,
        y_train=y,
        y_test=None,
        curr_dataset="bc",
        run_info=["SLIM+SIG2", "test", "bc"],
        n_iter=2,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        ffunction=rmse,
        max_depth=20,
        n_elites=1,
        reconstruct=False,
        n_jobs=1,
    )

    assert optimizer.elite is not None
    assert len(optimizer.population.population) == optimizer.pop_size


def test_main_slim_accepts_crossover_argument():
    # Ensure tests do not depend on GPU availability.
    set_device("cpu")
    X, y = _make_tiny_problem(seed=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = str(Path(tmpdir) / "slim.csv")

        elite = slim(
            X_train=X,
            y_train=y,
            X_test=None,
            y_test=None,
            dataset_name="bc",
            slim_version="SLIM+SIG2",
            pop_size=6,
            n_iter=2,
            p_xo=1.0,
            crossover_operator="uniform",
            log_path=log_path,
            log_level=0,
            verbose=0,
            test_elite=False,
            max_depth=20,
            reconstruct=True,
            n_jobs=1,
            seed=123,
        )

        assert elite is not None
