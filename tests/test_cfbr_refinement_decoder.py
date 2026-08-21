import torch

from models.cfbr_decoder import CFBRRefinementDecoder


def test_three_cfbr_stages_refine_mask_through_22_44_88_scales():
    decoder = CFBRRefinementDecoder()
    initial_mask_logits = torch.randn(2, 1, 88, 88)
    combined_image_feature = torch.randn(2, 256, 22, 22)
    high_res_features = [
        torch.randn(2, 32, 88, 88),
        torch.randn(2, 64, 44, 44),
    ]

    outputs = decoder(
        initial_mask_logits,
        combined_image_feature,
        high_res_features,
    )

    assert decoder.num_cfbr_stages == 3
    assert outputs["refined_22_logits"].shape == (2, 1, 22, 22)
    assert outputs["refined_44_logits"].shape == (2, 1, 44, 44)
    assert outputs["refined_88_logits"].shape == (2, 1, 88, 88)
    assert outputs["refined_high_res_masks"].shape == (2, 1, 352, 352)


def test_three_cfbr_stages_backpropagate_to_initial_mask_and_features():
    decoder = CFBRRefinementDecoder()
    initial_mask_logits = torch.randn(1, 1, 88, 88, requires_grad=True)
    combined_image_feature = torch.randn(1, 256, 22, 22, requires_grad=True)
    high_res_features = [
        torch.randn(1, 32, 88, 88, requires_grad=True),
        torch.randn(1, 64, 44, 44, requires_grad=True),
    ]

    outputs = decoder(initial_mask_logits, combined_image_feature, high_res_features)
    outputs["refined_high_res_masks"].mean().backward()

    assert initial_mask_logits.grad is not None
    assert combined_image_feature.grad is not None
    assert high_res_features[0].grad is not None
    assert high_res_features[1].grad is not None
