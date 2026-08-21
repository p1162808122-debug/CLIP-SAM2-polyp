import numpy as np
import torch

from visualize_explainability import (
    boundary_statistics,
    gradient_to_heatmap,
    resize_mask,
)


def test_gradient_to_heatmap_reduces_channels_and_normalizes():
    gradient = torch.tensor([[[[0.0, 1.0]], [[0.0, 2.0]], [[0.0, 3.0]]]])
    heatmap = gradient_to_heatmap(gradient)
    assert heatmap.shape == (1, 2)
    assert np.isfinite(heatmap).all()
    assert heatmap[0, 1] > heatmap[0, 0]


def test_boundary_statistics_separates_boundary_from_background():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    heatmap = np.zeros((5, 5), dtype=np.float32)
    heatmap[1:4, 1:4] = 1.0
    result = boundary_statistics(heatmap, mask)
    assert result["boundary_mean"] > result["background_mean"]


def test_resize_mask_aligns_original_target_to_model_resolution():
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 3:6] = 1
    resized = resize_mask(mask, size=4)
    assert resized.shape == (4, 4)
    assert resized.dtype == np.bool_
    assert int(resized.sum()) > 0
