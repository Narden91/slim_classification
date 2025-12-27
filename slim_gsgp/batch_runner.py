#!/usr/bin/env python3
"""Utility to execute batches of experiments with checkpointing support."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional dependency – only needed for YAML configs
    import yaml  # type: ignore
except Exception:  # pragma: no cover - YAML is optional
    yaml = None

from slim_gsgp.example_binary_classification import (
    create_default_experiment_config,
    run_experiment,
)
from slim_gsgp.utils.experiment_registry import (
    ExperimentRegistry,
    ExperimentRun,
    experiment_run_from_config,
)

GRID_KEYS = {"datasets", "algorithms", "seeds", "slim_versions"}
RESERVED_KEYS = GRID_KEYS | {"runs", "grid", "defaults"}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute multiple experiments with resume support")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON or YAML file describing the experiment grid",
    )
    parser.add_argument(
        "--registry-path",
        default="results/experiment_registry.json",
        help="Location of the experiment registry JSON file",
    )
    parser.add_argument(
        "--resume/--no-resume",
        dest="resume",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Skip experiments already marked as completed",
    )
    parser.add_argument(
        "--reset-running",
        action="store_true",
        help="Reset runs that are marked as running to pending before execution",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list the registry entries produced from the config and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the batch and print the order without executing the experiments",
    )
    parser.add_argument(
        "--stop-on-error/--keep-going",
        dest="stop_on_error",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Abort on first failure instead of continuing with remaining runs",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsn"}:
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to use YAML configuration files")
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config extension: {suffix}")


def _extract_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = create_default_experiment_config()
    defaults.update(cfg.get("defaults", {}))
    for key, value in cfg.items():
        if key in RESERVED_KEYS:
            continue
        defaults[key] = value
    return defaults


def _extract_grid(cfg: Dict[str, Any]) -> Dict[str, Sequence[Any]]:
    if "grid" in cfg:
        grid = cfg["grid"] or {}
    else:
        grid = {key: cfg.get(key) for key in GRID_KEYS if key in cfg}
    return {k: v for k, v in grid.items() if v is not None}


def build_run_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    defaults = _extract_defaults(cfg)
    run_specs: List[Dict[str, Any]] = []

    if "runs" in cfg:
        for entry in cfg["runs"]:
            spec = dict(defaults)
            spec.update(entry)
            run_specs.append(spec)
        return run_specs

    grid = _extract_grid(cfg)
    datasets = grid.get("datasets", [defaults["dataset"]])
    algorithms = grid.get("algorithms", [defaults["algorithm"]])
    seeds = grid.get("seeds", [defaults["seed"]])
    slim_versions = grid.get("slim_versions", [defaults.get("slim_version")])

    for dataset in datasets:
        for algorithm in algorithms:
            versions = slim_versions if algorithm == "slim" else [None]
            if algorithm == "slim" and not any(versions):
                raise ValueError("SLIM runs require at least one slim_version entry")
            for version in versions:
                for seed in seeds:
                    spec = dict(defaults)
                    spec.update(
                        {
                            "dataset": dataset,
                            "algorithm": algorithm,
                            "seed": seed,
                            "slim_version": version,
                        }
                    )
                    run_specs.append(spec)
    return run_specs


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _sanitize_max_depth(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return value


def _config_to_namespace(config: Dict[str, Any]) -> SimpleNamespace:
    config = dict(config)
    config["max_depth"] = _sanitize_max_depth(config.get("max_depth"))
    return SimpleNamespace(**config)


def _print_plan(planned: List[Tuple[ExperimentRun, SimpleNamespace]]) -> None:
    print("Planned experiments:")
    for idx, (run, ns) in enumerate(planned, start=1):
        slim_info = f" {ns.slim_version}" if run.slim_version else ""
        print(
            f"[{idx}] {run.dataset} | {run.algorithm}{slim_info} | seed={run.seed} | pop={run.pop_size} | iters={run.n_iter}"
        )


def execute_runs(
    planned: List[Tuple[ExperimentRun, SimpleNamespace]],
    registry: ExperimentRegistry,
    resume: bool,
    reset_running: bool,
    dry_run: bool,
    stop_on_error: bool,
) -> None:
    if dry_run:
        _print_plan(planned)
        return

    for run, ns in planned:
        stored = registry.register(run)
        if stored.status == "completed" and resume:
            print(f"[SKIP] {stored.run_id} already completed")
            continue
        if stored.status == "running":
            if reset_running:
                stored = registry.reset(run.run_id)
            else:
                print(f"[SKIP] {stored.run_id} still marked as running (use --reset-running to retry)")
                continue

        print(
            f"[RUN ] dataset={stored.dataset} algorithm={stored.algorithm} seed={stored.seed}"
        )
        try:
            registry.mark_started(stored.run_id)
            metrics, training_time, metrics_path, _, _ = run_experiment(ns)
            registry.mark_completed(stored.run_id, metrics_path=metrics_path, duration=training_time)
            print(
                f"[DONE] run_id={stored.run_id[:10]}... metrics={metrics_path} duration={training_time:.2f}s"
            )
        except Exception as exc:  # pragma: no cover - defensive
            registry.mark_failed(stored.run_id, str(exc))
            print(f"[FAIL] run_id={stored.run_id[:10]}... {exc}")
            if stop_on_error:
                raise


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config_data = load_config(config_path)
    run_configs = build_run_configs(config_data)

    namespaces = [_config_to_namespace(cfg) for cfg in run_configs]
    planned = [(experiment_run_from_config(ns), ns) for ns in namespaces]

    registry = ExperimentRegistry(Path(args.registry_path))

    if args.list:
        _print_plan(planned)
        existing = registry.list()
        print(f"\nRegistry currently tracks {len(existing)} run(s).")
        status_counts: Dict[str, int] = {}
        for run in existing:
            status_counts[run.status] = status_counts.get(run.status, 0) + 1
        for status, count in sorted(status_counts.items()):
            print(f"  - {status}: {count}")
        return 0

    execute_runs(
        planned=planned,
        registry=registry,
        resume=args.resume,
        reset_running=args.reset_running,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
