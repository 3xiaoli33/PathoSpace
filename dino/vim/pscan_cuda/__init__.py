"""
CUDA-accelerated parallel scan module for 2D Mamba.

This module provides optimized CUDA kernels for selective scan operations
used in the 2D Mamba architecture, significantly speeding up training.

Functions:
    fwd: Forward pass of selective scan
    bwd: Backward pass of selective scan
"""

# Import the compiled CUDA extension
try:
    from . import pscan as pscan_cuda
    # Make the module directly accessible
    __all__ = ['pscan_cuda']
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import pscan CUDA module: {e}. "
                  "The model will fall back to slower PyTorch implementation.")
    pscan_cuda = None
    __all__ = []
