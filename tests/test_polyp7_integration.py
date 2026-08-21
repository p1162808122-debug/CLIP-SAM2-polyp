import torch
from torch import nn

from models.static_sam2 import StaticSAM2GatedPrompt


class RecordingMaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_image_embeddings = None
        self.last_dense_prompt_embeddings = None
        self.last_high_res_features = None

    def forward(
        self,
        *,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        repeat_image,
        high_res_features,
    ):
        self.last_image_embeddings = image_embeddings
        self.last_dense_prompt_embeddings = dense_prompt_embeddings
        self.last_high_res_features = high_res_features
        batch = image_embeddings.shape[0]
        device = image_embeddings.device
        return (
            image_embeddings.new_zeros(batch, 1, 88, 88),
            image_embeddings.new_zeros(batch, 1),
            image_embeddings.new_zeros(batch, 1, 256),
            image_embeddings.new_zeros(batch, 1),
        )


class FakeSAM2(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 352
        self.num_feature_levels = 3
        self.use_high_res_features_in_sam = True
        self.sam_mask_decoder = RecordingMaskDecoder()

    def forward_image(self, images):
        batch = images.shape[0]
        device = images.device
        return {
            "vision_features": torch.randn(batch, 256, 22, 22, device=device),
            "backbone_fpn": [
                torch.randn(batch, 32, 88, 88, device=device),
                torch.randn(batch, 64, 44, 44, device=device),
                torch.randn(batch, 256, 22, 22, device=device),
            ],
            "vision_pos_enc": [torch.randn(1, 256, 22, 22, device=device)],
        }


def test_gated_prompt_uses_fused_maps_and_zero_dense_prompt():
    sam2 = FakeSAM2()
    wrapper = StaticSAM2GatedPrompt(sam2)
    output = wrapper(torch.randn(2, 3, 352, 352), torch.randn(2, 256, 22, 22))

    assert sam2.sam_mask_decoder.last_image_embeddings is output["fused_image_embeddings"]
    assert sam2.sam_mask_decoder.last_high_res_features[0] is output["fused_high_res_features"][0]
    assert sam2.sam_mask_decoder.last_high_res_features[1] is output["fused_high_res_features"][1]
    assert torch.count_nonzero(sam2.sam_mask_decoder.last_dense_prompt_embeddings) == 0
    assert output["selected_mask_logits"].shape == (2, 1, 88, 88)
