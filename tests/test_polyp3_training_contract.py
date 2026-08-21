import sys

import pytest
import torch

import MyTest
import MyTrain
from models.polyp5 import Polyp5Model
from utils.losses import (
    boundary_similarity_loss,
    compute_polyp5_loss_components,
    soft_patch_similarity_loss,
    structure_loss,
)


def test_polyp7_training_defaults_and_variant(monkeypatch):
    monkeypatch.setattr(MyTrain, "__file__", "/tmp/SAM2-polyp7/MyTrain.py")
    monkeypatch.setattr(sys, "argv", ["MyTrain.py"])

    args = MyTrain.parse_args()

    assert MyTrain.project_variant() == "polyp7"
    assert args.lora_rank is None
    assert args.sam_lora_rank is None
    assert args.clip_lora_rank is None
    assert args.text_lora_rank is None
    assert args.similarity_loss_weight == pytest.approx(0.5)
    assert args.boundary_loss_weight == pytest.approx(0.25)
    assert args.boundary_threshold == pytest.approx(0.1)
    assert args.boundary_temperature == pytest.approx(0.05)
    assert args.boundary_radius == 8
    assert args.trainsize == 352


def test_polyp7_loss_contains_three_weighted_components():
    masks = torch.zeros(1, 1, 4, 4)
    masks[:, :, 1:3, 1:3] = 1.0
    output = {
        "high_res_masks": torch.zeros(1, 1, 4, 4),
        "similarity_logits": torch.zeros(1, 2, 2, 2),
        "similarity_delta": torch.zeros(1, 1, 2, 2, requires_grad=True),
    }

    components = compute_polyp5_loss_components(output, masks, similarity_loss_weight=0.5, boundary_loss_weight=0.25)
    expected_boundary = boundary_similarity_loss(output["similarity_delta"], masks)[0]
    expected = structure_loss(output["high_res_masks"], masks) + 0.5 * soft_patch_similarity_loss(
        output["similarity_logits"], masks
    ) + 0.25 * expected_boundary
    assert components["total_loss"].item() == pytest.approx(expected.item())


def test_polyp7_text_lora_is_always_enabled():
    assert MyTest.infer_text_lora(None, requested=None) is True
    assert MyTest.infer_text_lora(None, requested=False) is True
    assert Polyp5Model.SUPPORTED_LORA_RANKS == (128,)
