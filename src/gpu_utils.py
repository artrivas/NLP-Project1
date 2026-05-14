"""GPU inspection helpers for training scripts."""

from __future__ import annotations

import logging

import torch


logger = logging.getLogger(__name__)


def get_gpu_name() -> str:
    """Return the current CUDA GPU name, or a CPU-only message."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    return torch.cuda.get_device_name(torch.cuda.current_device())


def get_cuda_memory_summary() -> str:
    """Return a CUDA memory summary when CUDA is available."""
    if not torch.cuda.is_available():
        return "CUDA not available; no CUDA memory summary."
    return torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=True)


def empty_cuda_cache() -> None:
    """Clear the CUDA allocator cache when CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_info() -> None:
    """Log CUDA availability, GPU name, and current memory state."""
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("GPU name: %s", get_gpu_name())
    if torch.cuda.is_available():
        logger.info("Initial CUDA memory summary:\n%s", get_cuda_memory_summary())
