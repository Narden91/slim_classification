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
import argparse

import torch

from slim_gsgp.main_slim import slim  # import the slim_gsgp library
from slim_gsgp.datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from slim_gsgp.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim_gsgp.utils.utils import train_test_split  # import the train-test split function


def _resolve_device(requested_device: str) -> str:
    requested_device = (requested_device or "auto").strip().lower()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Warning: --device=cuda requested but CUDA is not available; falling back to CPU.")
        return "cpu"
    return requested_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SLIM-GSGP regression on the PPB dataset")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--slim-version", type=str, default="SLIM+SIG2")
    parser.add_argument("--p-inflate", type=float, default=0.5)

    # Profiling (Phase 0)
    parser.add_argument("--profile", action="store_true", help="Print SLIM stage timing breakdown")
    parser.add_argument("--profile-cuda-sync", action="store_true", help="Synchronize CUDA for accurate timings")
    parser.set_defaults(profile_cuda_sync=True)
    parser.add_argument("--torch-profile", action="store_true", help="Enable torch.profiler summary + optional trace")
    parser.add_argument("--torch-profile-steps", type=int, default=2, help="Number of steps to record in torch.profiler")
    parser.add_argument("--torch-profile-trace-dir", type=str, default=None, help="Directory to write chrome trace JSON")
    args = parser.parse_args()

    resolved_device = _resolve_device(args.device)

    # Keep global config modules in sync with the requested device.
    # These modules generate constants using their internal DEVICE.
    try:
        from slim_gsgp.config.gp_config import set_device as _gp_set_device
        from slim_gsgp.config.gsgp_config import set_device as _gsgp_set_device
        from slim_gsgp.config.slim_config import set_device as _slim_set_device

        _gp_set_device(resolved_device)
        _gsgp_set_device(resolved_device)
        _slim_set_device(resolved_device)
    except Exception:
        pass

    device_obj = torch.device(resolved_device)

    # Load the PPB dataset
    X, y = load_ppb(X_y=True)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4, seed=args.seed)

    # Split the test set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5, seed=args.seed)

    # Move tensors to the selected device to prevent CPU/CUDA mismatch.
    X_train = X_train.to(device_obj)
    y_train = y_train.to(device_obj)
    X_val = X_val.to(device_obj)
    y_val = y_val.to(device_obj)
    X_test = X_test.to(device_obj)
    y_test = y_test.to(device_obj)

    print("Running regression with SLIM on ppb")
    print(f"  Device: {args.device} (resolved: {resolved_device})")
    print(f"  Slim version: {args.slim_version}")
    print(f"  Population size: {args.pop_size}")
    print(f"  Iterations: {args.n_iter}")
    print(f"  Seed: {args.seed}")
    print(f"  Max depth: {args.max_depth}")
    print(f"  P-inflate: {args.p_inflate}")
    print()

    # Apply the SLIM GSGP algorithm
    final_tree = slim(
        X_train=X_train,
        y_train=y_train,
        X_test=X_val,
        y_test=y_val,
        dataset_name="ppb",
        slim_version=args.slim_version,
        pop_size=args.pop_size,
        n_iter=args.n_iter,
        ms_lower=0,
        ms_upper=1,
        p_inflate=args.p_inflate,
        reconstruct=True,
        seed=args.seed,
        max_depth=args.max_depth,
        n_jobs=1,
        profile=args.profile,
        profile_cuda_sync=args.profile_cuda_sync,
        torch_profile=args.torch_profile,
        torch_profile_steps=args.torch_profile_steps,
        torch_profile_trace_dir=args.torch_profile_trace_dir,
    )

    # Show the best individual structure at the last generation
    final_tree.print_tree_representation()

    # Get the prediction of the best individual on the test set
    predictions = final_tree.predict(X_test)

    # Compute and print the RMSE on the test set
    score = rmse(y_true=y_test, y_pred=predictions)
    print(float(score.detach().cpu().item()))


if __name__ == "__main__":
    main()
