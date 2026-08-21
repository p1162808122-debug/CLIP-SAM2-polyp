# Portable BiomedCLIP model

This folder contains the runtime source needed by the BiomedCLIP checkpoint.
The files under `open_clip/` are the Python implementation from
`open_clip_torch==2.23.0`, including the ViT image tower, Hugging Face
text-tower adapter, Transformer blocks, tokenizer, transforms, and checkpoint
loader. The only package dependency that supplies model architecture code at
runtime is `transformers`, which constructs the local PubMedBERT architecture
from `pretrained/text_encoder/config.json`.

```python
from model.biomedclip import BiomedCLIP
from model.contrastive_loss import clip_loss

model = BiomedCLIP.from_pretrained("../pretrained")
tokens = model.tokenizer(["a biomedical image"])
image_features, text_features, logit_scale = model(images, tokens)
loss = clip_loss(image_features, text_features, logit_scale)
loss.backward()
```

`model.visual` is the complete image encoder. For BiomedCLIP,
`model.model.text` is the complete PubMedBERT encoder plus the OpenCLIP MLP
projection. Both are trainable by default; use `set_trainable()` when a
downstream experiment needs to freeze one tower.
