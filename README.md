# CLIP-SAM2

Research code for a static polyp-segmentation model that combines a SAM2 Hiera-L image encoder with BiomedCLIP semantic prompting.

## What is included

- SAM2 image-encoding and mask-decoding code
- BiomedCLIP/OpenCLIP integration for image and text features
- LoRA adaptation for the SAM2, visual, and text branches
- Three-scale gated fusion and CFBR refinement
- Mask, semantic-similarity, and boundary-similarity losses
- Training, testing, fixed-threshold evaluation, and lightweight tests

The current model entrypoint is `MyTrain.py`; the polyp7 implementation is in `models/polyp5.py` for historical compatibility.

## What is intentionally excluded

This repository is source-only. Datasets, pretrained weights, checkpoints, prediction masks, evaluation tables, explainability outputs, logs, and caches are not included.

Before running the code, provide the required SAM2 and BiomedCLIP pretrained assets and your own dataset/split paths. The command-line defaults use relative paths such as `data/TrainDataset` and `data/TestDataset`.

## Basic usage

```bash
python MyTrain.py \
  --train-path /path/to/TrainDataset \
  --split-dir /path/to/splits \
  --sam-lora-rank 128 \
  --clip-lora-rank 128 \
  --text-lora-rank 64 \
  --batchsize 16

python MyTest.py \
  --run-dir /path/to/checkpoint/run_directory \
  --test-path /path/to/TestDataset

python MyEval.py \
  --models run_name \
  --data-path /path/to/TestDataset
```

This code is intended for research use. Please check and retain the upstream licensing and attribution requirements for the vendored SAM2 and BiomedCLIP components.
