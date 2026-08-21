# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This file keeps the original ``third_parts.sam2`` import paths working while
# also allowing the package to be imported directly as ``sam2``.  Both names
# must resolve to the same package object so Hydra is initialized only once.

import sys
import types
from pathlib import Path

from hydra import initialize_config_module


_THIS_MODULE = sys.modules[__name__]
_SOURCE_ROOT = Path(__file__).resolve().parent

if __name__ == "sam2":
    third_parts = sys.modules.get("third_parts")
    if third_parts is None:
        third_parts = types.ModuleType("third_parts")
        third_parts.__path__ = [str(_SOURCE_ROOT.parent / "third_parts")]
        sys.modules["third_parts"] = third_parts
    sys.modules.setdefault("third_parts.sam2", _THIS_MODULE)
else:
    sys.modules.setdefault("sam2", _THIS_MODULE)


initialize_config_module("third_parts.sam2.sam2_configs", version_base="1.2")
