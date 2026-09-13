# -*- coding: utf-8 -*-
"""
PyTorch Accelerator Compatibility Shim.
Diffusers >= 0.31 unconditionally queries `torch.xpu.empty_cache`, `torch.xpu.device_count`, etc.
On PyTorch 2.2 / CUDA builds without Intel XPU support, accessing `torch.xpu` triggers:
    AttributeError: module 'torch' has no attribute 'xpu'

This module safely shims `torch.xpu` with benign stubs before Diffusers or Accelerate are imported.
"""

import sys

try:
    import torch

    if not hasattr(torch, "xpu"):
        class _DummyXpu:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def device_count():
                return 0

            @staticmethod
            def empty_cache():
                pass

            @staticmethod
            def manual_seed(seed=0):
                pass

            @staticmethod
            def reset_peak_memory_stats(device=None):
                pass

            @staticmethod
            def reset_max_memory_allocated(device=None):
                pass

            @staticmethod
            def max_memory_allocated(device=None):
                return 0

            @staticmethod
            def synchronize(device=None):
                pass

            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        torch.xpu = _DummyXpu()
except ImportError:
    pass
