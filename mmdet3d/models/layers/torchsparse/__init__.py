# Copyright (c) OpenMMLab. All rights reserved.
# PlantSegStudio/mmdet3d/models/layers/torchsparse/__init__.py
from .torchsparse_wrapper import register_torchsparse

try:
    import torchsparse  # noqa
except ImportError:
    print("Torchspare is NOT available")
    IS_TORCHSPARSE_AVAILABLE = False
else:
    IS_TORCHSPARSE_AVAILABLE = register_torchsparse()

__all__ = ['IS_TORCHSPARSE_AVAILABLE']
