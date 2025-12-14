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
"""Registry helpers to coordinate experiment-level checkpointing."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

__all__ = [
    "ExperimentRun",
    "ExperimentRegistry",
    "ExperimentStatusError",
    "experiment_run_from_config",
]

ISO_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_now() -> str:
    """Return the current UTC timestamp formatted as ISO-8601."""
    return datetime.now(timezone.utc).strftime(ISO_TS_FORMAT)


class ExperimentStatusError(RuntimeError):
    """Raised when an invalid status transition is requested."""


@dataclass
class ExperimentRun:
    """Structured metadata describing a single experiment execution."""

    dataset: str
    algorithm: str
    seed: int
    slim_version: Optional[str] = None
    pop_size: Optional[int] = None
    n_iter: Optional[int] = None
    max_depth: Optional[int] = None
    p_inflate: Optional[float] = None
    fitness_function: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics_path: Optional[str] = None
    log_path: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.run_id is None:
            self.run_id = self._compute_run_id()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentRun":
        return cls(**payload)

    # ------------------------------------------------------------------
    # Derived data
    # ------------------------------------------------------------------
    def config_signature(self) -> Dict[str, Any]:
        """Return the subset of fields that uniquely describe a run configuration."""
        signature_fields = {
            "dataset": self.dataset,
            "algorithm": self.algorithm,
            "slim_version": self.slim_version,
            "seed": self.seed,
            "pop_size": self.pop_size,
            "n_iter": self.n_iter,
            "max_depth": self.max_depth,
            "p_inflate": self.p_inflate,
            "fitness_function": self.fitness_function,
        }
        if self.extra_params:
            signature_fields.update(self.extra_params)
        return signature_fields

    def _compute_run_id(self) -> str:
        signature = self.config_signature()
        encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha1(encoded, usedforsecurity=False).hexdigest()

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------
    def mark_started(self) -> None:
        self.status = "running"
        self.started_at = _utc_now()
        self.completed_at = None
        self.error_message = None

    def mark_completed(self, metrics_path: Optional[str] = None, duration: Optional[float] = None) -> None:
        self.status = "completed"
        self.completed_at = _utc_now()
        if metrics_path is not None:
            self.metrics_path = metrics_path
        if duration is not None:
            self.duration_seconds = duration

    def mark_failed(self, error_message: str) -> None:
        self.status = "failed"
        self.completed_at = _utc_now()
        self.error_message = error_message


class _FileLock:
    """Minimal cross-platform file lock using the standard library."""

    def __init__(self, lock_path: Path):
        self._lock_path = lock_path
        self._handle = None

    def acquire(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._lock_path, "a+")
        if os.name == "nt":  # pragma: no cover
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def release(self) -> None:
        if self._handle is None:
            return
        if os.name == "nt":  # pragma: no cover
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "_FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class ExperimentRegistry:
    """JSON-backed registry with optimistic cross-process locking."""

    schema_version = 1

    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _FileLock(self.path.with_suffix(self.path.suffix + ".lock"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(self, run: ExperimentRun) -> ExperimentRun:
        """Ensure a run exists in the registry and return the persisted record."""
        with self._transaction() as runs:
            mapping = {item.run_id: item for item in runs}
            if run.run_id not in mapping:
                runs.append(run)
                mapping[run.run_id] = run
            return mapping[run.run_id]

    def upsert(self, run: ExperimentRun) -> ExperimentRun:
        with self._transaction() as runs:
            mapping = {item.run_id: item for item in runs}
            mapping[run.run_id] = run
            return mapping[run.run_id]

    def get(self, run_id: str) -> Optional[ExperimentRun]:
        runs = self._read_runs()
        return next((run for run in runs if run.run_id == run_id), None)

    def list(self, status: Optional[str] = None) -> List[ExperimentRun]:
        runs = self._read_runs()
        if status is None:
            return runs
        return [run for run in runs if run.status == status]

    def mark_started(self, run_id: str, log_path: Optional[str] = None) -> ExperimentRun:
        def mutate(run: ExperimentRun) -> None:
            if run.status not in {"pending", "failed"}:
                raise ExperimentStatusError(
                    f"Cannot start run {run_id}: status={run.status!r}"
                )
            run.mark_started()
            if log_path is not None:
                run.log_path = log_path

        return self._mutate_run(run_id, mutate)

    def mark_completed(self, run_id: str, metrics_path: Optional[str] = None, duration: Optional[float] = None) -> ExperimentRun:
        def mutate(run: ExperimentRun) -> None:
            if run.status != "running":
                raise ExperimentStatusError(
                    f"Cannot complete run {run_id}: status={run.status!r}"
                )
            run.mark_completed(metrics_path=metrics_path, duration=duration)

        return self._mutate_run(run_id, mutate)

    def mark_failed(self, run_id: str, error_message: str) -> ExperimentRun:
        def mutate(run: ExperimentRun) -> None:
            if run.status not in {"running", "pending"}:
                raise ExperimentStatusError(
                    f"Cannot fail run {run_id}: status={run.status!r}"
                )
            run.mark_failed(error_message)

        return self._mutate_run(run_id, mutate)

    def reset(self, run_id: str) -> ExperimentRun:
        def mutate(run: ExperimentRun) -> None:
            run.status = "pending"
            run.started_at = None
            run.completed_at = None
            run.error_message = None
            run.metrics_path = None

        return self._mutate_run(run_id, mutate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _transaction(self) -> Iterable[List[ExperimentRun]]:
        with self._lock:
            runs = self._read_runs()
            yield runs
            self._write_runs(runs)

    def _mutate_run(self, run_id: str, mutator) -> ExperimentRun:
        with self._transaction() as runs:
            for run in runs:
                if run.run_id == run_id:
                    mutator(run)
                    return run
            raise KeyError(f"Run id {run_id} not found in registry")

    def _read_runs(self) -> List[ExperimentRun]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        version = payload.get("version", 0)
        if version != self.schema_version:
            raise RuntimeError(
                f"Unsupported experiment registry schema: {version}, expected {self.schema_version}"
            )
        return [ExperimentRun.from_dict(item) for item in payload.get("runs", [])]

    def _write_runs(self, runs: List[ExperimentRun]) -> None:
        payload = {
            "version": self.schema_version,
            "updated_at": _utc_now(),
            "runs": [run.to_dict() for run in runs],
        }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, sort_keys=False)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, Mapping):
        return config.get(key, default)
    return default


def experiment_run_from_config(config: Any) -> ExperimentRun:
    dataset = _get_config_value(config, "dataset")
    algorithm = _get_config_value(config, "algorithm")
    seed = _get_config_value(config, "seed")
    if dataset is None or algorithm is None or seed is None:
        raise ValueError("Config must define dataset, algorithm, and seed fields")

    extra_params = {
        "use_sigmoid": _get_config_value(config, "use_sigmoid", True),
        "sigmoid_scale": _get_config_value(config, "sigmoid_scale", 1.0),
        "fitness_function": _get_config_value(config, "fitness_function", "binary_rmse"),
        "save_visualization": _get_config_value(config, "save_visualization", False),
        "verbose": _get_config_value(config, "verbose", False),
    }

    return ExperimentRun(
        dataset=dataset,
        algorithm=algorithm,
        seed=seed,
        slim_version=_get_config_value(config, "slim_version") if algorithm == "slim" else None,
        pop_size=_get_config_value(config, "pop_size"),
        n_iter=_get_config_value(config, "n_iter"),
        max_depth=_get_config_value(config, "max_depth"),
        p_inflate=_get_config_value(config, "p_inflate"),
        fitness_function=_get_config_value(config, "fitness_function"),
        extra_params=extra_params,
    )
