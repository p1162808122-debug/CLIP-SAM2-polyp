import torch
from torch import nn


def test_polyp4_prompt_adapter_projects_512_patch_tokens_to_256_dense_prompt():
    from models.polyp4 import ClipDensePromptAdapter

    adapter = ClipDensePromptAdapter()
    output = adapter(torch.randn(2, 484, 512))
    assert output.shape == (2, 256, 22, 22)


def test_polyp4_similarity_logits_stay_in_512_dimension():
    from models.polyp4 import patch_text_similarity

    patches = torch.randn(2, 484, 512, requires_grad=True)
    texts = torch.randn(2, 512, requires_grad=True)
    logits = patch_text_similarity(patches, texts)
    assert logits.shape == (2, 2, 22, 22)
    logits.mean().backward()
    assert patches.grad is not None
    assert texts.grad is not None


def test_dense_static_sam2_uses_image_encoder_positional_encoding_without_prompt_encoder():
    from models.static_sam2 import StaticSAM2DensePrompt

    class BombPromptEncoder(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("SAM2 PromptEncoder must not be called")

    class Decoder(nn.Module):
        def forward(self, **kwargs):
            assert kwargs["sparse_prompt_embeddings"].shape == (2, 0, 256)
            assert kwargs["dense_prompt_embeddings"].shape == (2, 256, 22, 22)
            assert kwargs["image_pe"].shape == (1, 256, 22, 22)
            low = kwargs["image_embeddings"].new_zeros(2, 1, 88, 88)
            iou = kwargs["image_embeddings"].new_zeros(2, 1)
            tokens = kwargs["image_embeddings"].new_zeros(2, 1, 256)
            obj = kwargs["image_embeddings"].new_zeros(2, 1)
            return low, iou, tokens, obj

    class FakeSAM2(nn.Module):
        image_size = 352
        num_feature_levels = 3
        use_high_res_features_in_sam = True
        directly_add_no_mem_embed = False
        sam_prompt_encoder = BombPromptEncoder()
        sam_mask_decoder = Decoder()

        def forward_image(self, images):
            z = images.new_zeros
            return {
                "vision_features": z(2, 256, 22, 22),
                "vision_pos_enc": [z(2, 256, 88, 88), z(2, 256, 44, 44), z(2, 256, 22, 22)],
                "backbone_fpn": [z(2, 32, 88, 88), z(2, 64, 44, 44), z(2, 256, 22, 22)],
            }

    model = StaticSAM2DensePrompt(FakeSAM2())
    output = model(torch.randn(2, 3, 352, 352), torch.randn(2, 256, 22, 22))
    assert output["high_res_masks"].shape == (2, 1, 352, 352)


def test_polyp4_similarity_loss_uses_soft_area_targets_and_lambda():
    from utils.losses import compute_polyp4_loss, structure_loss

    masks = torch.zeros(1, 1, 352, 352)
    masks[:, :, 88:264, 88:264] = 1.0
    output = {
        "high_res_masks": torch.zeros(1, 1, 352, 352),
        "similarity_logits": torch.zeros(1, 2, 22, 22),
    }
    actual = compute_polyp4_loss(output, masks, similarity_loss_weight=0.5)
    expected = structure_loss(output["high_res_masks"], masks) + 0.5 * output["similarity_logits"].new_zeros(())
    assert actual >= expected
