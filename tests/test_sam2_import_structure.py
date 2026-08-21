"""Regression tests for both supported SAM2 import namespaces."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


def test_third_parts_sam2_imports_and_composes_config():
    result = _run_python(
        "from third_parts.sam2.modeling.sam2_base import SAM2Base; "
        "from hydra import compose; "
        "cfg=compose(config_name='sam2_hiera_l.yaml'); "
        "assert SAM2Base is not None; "
        "assert cfg.model._target_ == 'third_parts.sam2.modeling.sam2_base.SAM2Base'; "
        "assert cfg.model.image_size == 1024"
    )
    assert result.returncode == 0, result.stderr


def test_sam2_build_import_and_composes_config():
    result = _run_python(
        "from sam2.build_sam import build_sam2; "
        "from hydra import compose; "
        "cfg=compose(config_name='sam2_hiera_l.yaml'); "
        "assert build_sam2 is not None; "
        "assert cfg.model.image_size == 1024"
    )
    assert result.returncode == 0, result.stderr
