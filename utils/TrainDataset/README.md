# Dataset split files

The dataset organization and split convention used in this project follow the **PraNet** polyp-segmentation setup. The training and testing data used in our experiments were prepared from the dataset resources provided or linked by the PraNet project, and the dataset partitioning strategy is kept consistent with that setup for fair comparison with prior polyp-segmentation work.

The training entry point expects `train.txt` and `val.txt` in this directory.
Each line must contain two paths relative to the training dataset root:

```text
images/example.png masks/example.png
```

The split files used on the private experiment server are intentionally not included in this public repository. When reproducing the experiments, users should prepare the datasets and split files according to the PraNet-style organization used by this project.
