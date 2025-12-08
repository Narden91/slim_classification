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
Device utilities for SLIM-GSGP.

This module provides utilities for managing PyTorch device selection
(CPU/GPU) and tensor device placement for optimized performance.
"""

import torch
from typing import Optional, Union

# Global device configuration
_default_device: Optional[torch.device] = None


def get_device(prefer_cuda: bool = True, device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the best available device for tensor operations.
    
    Parameters
    ----------
    prefer_cuda : bool
        If True, prefer CUDA if available. Default is True.
    device : str or torch.device, optional
        Explicit device to use. Overrides prefer_cuda if provided.
        Can be "auto", "cuda", "cpu", or a torch.device object.
    
    Returns
    -------
    torch.device
        The selected device for tensor operations.
    
    Examples
    --------
    >>> device = get_device()  # Auto-detect best device
    >>> device = get_device(device="cuda")  # Force CUDA
    >>> device = get_device(device="cpu")  # Force CPU
    """
    global _default_device
    
    # If explicit device is provided
    if device is not None:
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            pass  # Fall through to auto-detection
        elif device == "cuda":
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                return torch.device('cpu')
        elif device == "cpu":
            return torch.device('cpu')
        else:
            return torch.device(device)
    
    # Auto-detect best device
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_default_device(device: Union[str, torch.device]) -> None:
    """
    Set the default device for all SLIM-GSGP operations.
    
    Parameters
    ----------
    device : str or torch.device
        The device to use as default. Can be "auto", "cuda", "cpu".
    """
    global _default_device
    _default_device = get_device(device=device)


def get_default_device() -> torch.device:
    """
    Get the current default device.
    
    Returns
    -------
    torch.device
        The current default device. If not set, returns auto-detected device.
    """
    global _default_device
    if _default_device is None:
        _default_device = get_device()
    return _default_device


def to_device(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Move tensor to the specified device efficiently.
    
    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to move.
    device : torch.device, optional
        Target device. If None, uses the default device.
    
    Returns
    -------
    torch.Tensor
        The tensor on the target device.
    """
    if device is None:
        device = get_default_device()
    
    if tensor.device == device:
        return tensor
    return tensor.to(device)


def ensure_tensor_on_device(data, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Ensure data is a tensor on the specified device.
    
    Parameters
    ----------
    data : array-like or torch.Tensor
        Input data to convert/move.
    device : torch.device, optional
        Target device. If None, uses the default device.
    dtype : torch.dtype
        Data type for the tensor. Default is float32.
    
    Returns
    -------
    torch.Tensor
        The data as a tensor on the target device.
    """
    if device is None:
        device = get_default_device()
    
    if isinstance(data, torch.Tensor):
        if data.device == device and data.dtype == dtype:
            return data
        return data.to(device=device, dtype=dtype)
    
    return torch.tensor(data, dtype=dtype, device=device)


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns
    -------
    dict
        Dictionary containing device information including:
        - cuda_available: bool
        - cuda_device_count: int
        - cuda_device_names: list of str (if CUDA available)
        - default_device: str
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'default_device': str(get_default_device()),
    }
    
    if info['cuda_available']:
        info['cuda_device_names'] = [
            torch.cuda.get_device_name(i) for i in range(info['cuda_device_count'])
        ]
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated()
        info['cuda_memory_reserved'] = torch.cuda.memory_reserved()
    
    return info


def clear_cuda_cache() -> None:
    """
    Clear CUDA memory cache if CUDA is available.
    
    Useful for freeing up GPU memory between experiments.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
