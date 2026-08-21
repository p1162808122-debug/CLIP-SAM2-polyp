import torch
import torch.nn.functional as F
from torch import nn

from models.lora import LoRALinear, inject_lora_into_hf_text, inject_lora_into_vit
from models.prompt_adapters import SemanticPromptAdapter, SoftMaskPromptAdapter
from models.static_sam2 import StaticSAM2


def test_semantic_prompt_adapter_returns_one_256d_sparse_token():
    adapter = SemanticPromptAdapter()
    output = adapter(torch.randn(2, 512))
    assert output.shape == (2, 1, 256)


def test_soft_mask_prompt_adapter_returns_88x88_probability_map():
    adapter = SoftMaskPromptAdapter()
    output = adapter(torch.randn(2, 196, 512))
    assert output.shape == (2, 1, 88, 88)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_text_prototype_cost_prompt_adapter_returns_cost_maps_and_gradients():
    from models.prompt_adapters import TextPrototypeCostPromptAdapter

    adapter = TextPrototypeCostPromptAdapter(
        initial_background_features=F.normalize(torch.randn(4, 512), dim=-1)
    )
    output = adapter(torch.randn(2, 196, 512), torch.randn(3, 512))
    assert output["mask_prompt"].shape == (2, 1, 88, 88)
    assert output["prompt_logits"].shape == (2, 1, 88, 88)
    assert output["positive_cost"].shape == (2, 1, 14, 14)
    assert output["background_cost"].shape == (2, 1, 14, 14)
    assert output["cost_map"].shape == (2, 1, 14, 14)
    assert float(output["mask_prompt"].min()) >= 0.0
    assert float(output["mask_prompt"].max()) <= 1.0
    output["prompt_logits"].mean().backward()
    assert adapter.background_prototypes.grad is not None


def test_prompt_supervision_loss_ranks_correct_logits_lower_than_inverted_logits():
    from utils.losses import prompt_supervision_loss

    mask = torch.zeros(1, 1, 88, 88)
    mask[:, :, 22:66, 16:72] = 1.0
    correct_logits = torch.where(mask.bool(), torch.full_like(mask, 4.0), torch.full_like(mask, -4.0))
    assert prompt_supervision_loss(correct_logits, mask) < prompt_supervision_loss(-correct_logits, mask)


def test_vit_lora_replaces_all_qkv_and_projection_linears():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24)
            self.proj = nn.Linear(8, 8)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(), Block()])

    trunk = Trunk()
    names = inject_lora_into_vit(trunk, rank=32)
    assert len(names) == 4
    assert all(isinstance(trunk.blocks[i].attn.qkv, LoRALinear) for i in range(2))
    assert all(isinstance(trunk.blocks[i].attn.proj, LoRALinear) for i in range(2))
    assert all(not p.requires_grad for p in trunk.blocks[0].attn.qkv.base.parameters())


def test_hf_text_lora_replaces_attention_linears_and_leaves_only_adapters_trainable():
    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(768, 768)
            self.key = nn.Linear(768, 768)
            self.value = nn.Linear(768, 768)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = SelfAttention()
            self.output = nn.Module()
            self.output.dense = nn.Linear(768, 768)

    class EncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()

    class FakeBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.encoder = nn.Module()
            self.transformer.encoder.layer = nn.ModuleList([EncoderLayer(), EncoderLayer()])

    text_tower = FakeBert()
    target_names = inject_lora_into_hf_text(text_tower, rank=32, alpha=64, dropout=0.0)
    attention_linears = (
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
    )

    assert len(target_names) == 8
    assert set(target_names) == {
        f"transformer.encoder.layer.{layer_index}.{suffix}"
        for layer_index in range(2)
        for suffix in attention_linears
    }
    for layer in text_tower.transformer.encoder.layer:
        for linear in (
            layer.attention.self.query,
            layer.attention.self.key,
            layer.attention.self.value,
            layer.attention.output.dense,
        ):
            assert isinstance(linear, LoRALinear)
            assert all(not parameter.requires_grad for parameter in linear.base.parameters())
            assert linear.lora_A.requires_grad
            assert linear.lora_B.requires_grad

    try:
        inject_lora_into_hf_text(text_tower, rank=32, alpha=64, dropout=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("reinjecting text LoRA must fail")


def test_polyp3_prompt_constants_are_generic_binary_class_descriptions():
    from models.model import POLYP3_BACKGROUND_PROMPTS, POLYP3_POSITIVE_PROMPTS

    forbidden_terms = ("adenomatous", "sessile", "flat")
    assert POLYP3_POSITIVE_PROMPTS == (
        "an endoscopic image of a colorectal polyp",
        "a colorectal polyp in colonoscopy",
        "a polyp lesion on the colonic mucosa",
    )
    assert POLYP3_BACKGROUND_PROMPTS == (
        "normal colonic mucosa in a colonoscopy image",
        "background tissue in a colonoscopy image",
        "a normal colonic fold in an endoscopic image",
        "a non-polyp region in a colonoscopy image",
    )
    assert len(POLYP3_POSITIVE_PROMPTS) == 3
    assert len(POLYP3_BACKGROUND_PROMPTS) == 4
    assert all(isinstance(prompt, str) and prompt.strip() for prompt in (*POLYP3_POSITIVE_PROMPTS, *POLYP3_BACKGROUND_PROMPTS))
    assert all(
        term not in prompt.lower()
        for prompt in (*POLYP3_POSITIVE_PROMPTS, *POLYP3_BACKGROUND_PROMPTS)
        for term in forbidden_terms
    )


def test_static_sam2_forward_does_not_call_memory_modules():
    class Bomb(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("memory module was called")

    class PromptEncoder(nn.Module):
        mask_input_size = (88, 88)

        def get_dense_pe(self):
            return torch.zeros(1, 256, 22, 22)

        def forward(self, points, boxes, masks):
            batch = points[0].shape[0]
            sparse = torch.zeros(batch, 1, 256)
            dense = torch.zeros(batch, 256, 22, 22)
            return sparse, dense

    class MaskDecoder(nn.Module):
        def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                    dense_prompt_embeddings, multimask_output, repeat_image,
                    high_res_features):
            batch = image_embeddings.shape[0]
            low = image_embeddings.new_zeros(batch, 1, 88, 88)
            iou = image_embeddings.new_zeros(batch, 1)
            tokens = image_embeddings.new_zeros(batch, 1, 256)
            obj = image_embeddings.new_zeros(batch, 1)
            return low, iou, tokens, obj

    class FakeSAM2(nn.Module):
        image_size = 352
        num_feature_levels = 3
        use_high_res_features_in_sam = True
        directly_add_no_mem_embed = False
        sam_prompt_encoder = PromptEncoder()
        sam_mask_decoder = MaskDecoder()
        memory_encoder = Bomb()
        memory_attention = Bomb()

        def forward_image(self, images):
            return {
                "vision_features": images.new_zeros(images.shape[0], 256, 22, 22),
                "backbone_fpn": [
                    images.new_zeros(images.shape[0], 32, 88, 88),
                    images.new_zeros(images.shape[0], 64, 44, 44),
                    images.new_zeros(images.shape[0], 256, 22, 22),
                ],
            }

    model = StaticSAM2(FakeSAM2())
    output = model(torch.randn(2, 3, 352, 352))
    assert output["low_res_masks"].shape == (2, 1, 88, 88)
    assert output["high_res_masks"].shape == (2, 1, 352, 352)
