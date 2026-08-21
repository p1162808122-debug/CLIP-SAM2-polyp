import inspect
import sys

import pytest
import torch


def test_polyp7_model_defaults_to_rank128_with_independent_clip_ranks(monkeypatch):
    from models.polyp5 import Polyp5Model, polyp5_rank_label, resolve_polyp5_ranks
    import MyTrain
    import MyTest

    assert resolve_polyp5_ranks() == (128, 128, 64)
    assert resolve_polyp5_ranks(sam_lora_rank=256, clip_lora_rank=128, text_lora_rank=64) == (256, 128, 64)
    assert polyp5_rank_label(256, 128, 64) == "sam256_clip128_text64"
    parameters = inspect.signature(Polyp5Model.__init__).parameters
    assert {"sam_lora_rank", "clip_lora_rank", "text_lora_rank"}.issubset(parameters)

    monkeypatch.setattr(sys, "argv", ["MyTrain.py"])
    args = MyTrain.parse_args()
    assert args.boundary_loss_weight == pytest.approx(0.25)
    assert args.boundary_threshold == pytest.approx(0.1)
    assert args.boundary_temperature == pytest.approx(0.05)
    assert args.boundary_radius == 8

    monkeypatch.setattr(sys, "argv", ["MyTest.py"])
    assert MyTest.parse_args().testsize == 352


def test_morphological_boundary_is_area_pooled_to_22x22():
    from utils.losses import morphological_boundary_target

    mask = torch.zeros(1, 1, 352, 352)
    mask[:, :, 128:224, 128:224] = 1.0
    boundary = morphological_boundary_target(mask, output_size=(22, 22), radius=8)

    assert boundary.shape == (1, 1, 22, 22)
    assert float(boundary.min()) >= 0.0
    assert float(boundary.max()) <= 1.0
    assert float(boundary.max()) > 0.0
    assert float(boundary[:, :, 11, 11].max()) == pytest.approx(0.0)


def test_boundary_logits_are_high_only_for_near_zero_delta_and_have_gradient():
    from utils.losses import boundary_similarity_logits

    delta = torch.tensor([[[[-0.0, 0.05, 0.1, 0.3]]]], requires_grad=True)
    logits = boundary_similarity_logits(delta, threshold=0.1, temperature=0.05)

    assert logits.shape == delta.shape
    assert logits[..., 0].item() > logits[..., 3].item()
    assert logits[..., 1].item() > logits[..., 3].item()
    logits.sum().backward()
    assert delta.grad is not None
    assert float(delta.grad.abs().sum()) > 0.0


def test_polyp7_loss_has_three_components_and_backpropagates():
    from utils.losses import compute_polyp5_loss_components

    masks = torch.zeros(1, 1, 352, 352)
    masks[:, :, 128:224, 128:224] = 1.0
    output = {
        "high_res_masks": torch.zeros(1, 1, 352, 352, requires_grad=True),
        "similarity_logits": torch.zeros(1, 2, 22, 22, requires_grad=True),
        "similarity_delta": torch.zeros(1, 1, 22, 22, requires_grad=True),
    }
    losses = compute_polyp5_loss_components(
        output,
        masks,
        similarity_loss_weight=0.5,
        boundary_loss_weight=0.25,
        boundary_threshold=0.1,
        boundary_temperature=0.05,
        boundary_radius=8,
    )

    assert set(("mask_loss", "similarity_loss", "boundary_loss", "total_loss")).issubset(losses)
    expected = losses["mask_loss"] + 0.5 * losses["similarity_loss"] + 0.25 * losses["boundary_loss"]
    assert torch.allclose(losses["total_loss"], expected)
    losses["total_loss"].backward()
    assert output["high_res_masks"].grad is not None
    assert output["similarity_logits"].grad is not None
    assert output["similarity_delta"].grad is not None
