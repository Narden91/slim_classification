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
"""Tests for the experiment registry helpers."""

from types import SimpleNamespace

import pytest

from slim_gsgp.utils.experiment_registry import (
    ExperimentRegistry,
    experiment_run_from_config,
)


def _sample_namespace(seed: int = 42) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="eeg",
        algorithm="slim",
        slim_version="SLIM+ABS",
        seed=seed,
        pop_size=50,
        n_iter=10,
        max_depth=None,
        p_inflate=0.5,
        fitness_function="binary_rmse",
        use_sigmoid=True,
        sigmoid_scale=1.0,
        save_visualization=False,
        verbose=False,
    )


def test_experiment_run_from_config_deterministic():
    config = _sample_namespace()
    run_a = experiment_run_from_config(config)
    run_b = experiment_run_from_config(_sample_namespace())
    assert run_a.run_id == run_b.run_id


def test_registry_persists_status(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ExperimentRegistry(registry_path)
    run = experiment_run_from_config(_sample_namespace())
    stored = registry.register(run)
    registry.mark_started(stored.run_id)
    registry.mark_completed(stored.run_id, metrics_path="results.csv", duration=12.5)

    reloaded = ExperimentRegistry(registry_path)
    persisted = reloaded.get(stored.run_id)
    assert persisted is not None
    assert persisted.status == "completed"
    assert persisted.metrics_path == "results.csv"
    assert persisted.duration_seconds == pytest.approx(12.5)


def test_registry_reset_and_fail(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ExperimentRegistry(registry_path)
    run = experiment_run_from_config(_sample_namespace())
    stored = registry.register(run)
    registry.mark_started(stored.run_id)
    registry.mark_failed(stored.run_id, "boom")

    failed = registry.get(stored.run_id)
    assert failed is not None and failed.status == "failed"

    registry.reset(stored.run_id)
    pending = registry.get(stored.run_id)
    assert pending is not None and pending.status == "pending"