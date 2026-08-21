"""Compatibility shim for older ResShift torchvision imports."""

from __future__ import annotations

import sys
import types


def install() -> None:
    """Expose torchvision.transforms.functional_tensor.rgb_to_grayscale if absent."""
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    from torchvision.transforms import functional as functional

    shim = types.ModuleType(module_name)
    shim.rgb_to_grayscale = functional.rgb_to_grayscale
    shim.__all__ = ["rgb_to_grayscale"]
    sys.modules[module_name] = shim


install()
