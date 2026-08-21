import torch

from models.gated_fusion import GatedMultiScaleFusion


def test_three_scale_fusion_shapes_and_gate_ranges():
    fusion = GatedMultiScaleFusion()
    output = fusion(
        torch.randn(2, 256, 22, 22),
        torch.randn(2, 256, 22, 22),
        torch.randn(2, 64, 44, 44),
        torch.randn(2, 32, 88, 88),
    )

    assert output["fused22"].shape == (2, 256, 22, 22)
    assert output["fused44"].shape == (2, 64, 44, 44)
    assert output["fused88"].shape == (2, 32, 88, 88)
    for name in (
        "gate_sam22",
        "gate_clip22",
        "gate_sam44",
        "gate_clip44",
        "gate_sam88",
        "gate_clip88",
    ):
        assert torch.all((output[name] >= 0) & (output[name] <= 1))


def test_three_scale_fusion_backpropagates_to_both_sources():
    sam22 = torch.randn(1, 256, 22, 22, requires_grad=True)
    clip22 = torch.randn(1, 256, 22, 22, requires_grad=True)
    sam44 = torch.randn(1, 64, 44, 44, requires_grad=True)
    sam88 = torch.randn(1, 32, 88, 88, requires_grad=True)
    output = GatedMultiScaleFusion()(sam22, clip22, sam44, sam88)
    fused_loss = sum(
        value.square().mean()
        for key, value in output.items()
        if key.startswith("fused")
    )
    fused_loss.backward()

    assert sam22.grad is not None
    assert clip22.grad is not None
    assert sam44.grad is not None
    assert sam88.grad is not None
