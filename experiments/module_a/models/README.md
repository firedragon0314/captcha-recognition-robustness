# Module A Final Models

This folder keeps the final model package for each Module A experiment.

Each `M*` folder contains:

- `model.pth`: final selected model checkpoint.
- `history.csv`: training history for that model.

The previous per-folder `model_info.txt` files were consolidated here because they shared the same notes except for the source checkpoint path.

## Model Map

| Model | Folder | Source checkpoint |
| --- | --- | --- |
| M1 | `M1_corrupted_to_normal_supervised` | `experiments/train_002_M1/checkpoints/best_val_psnr.pth` |
| M2 | `M2_corrupted_to_normal_unsupervised` | `experiments/train_004_M2/checkpoints/best_val_psnr.pth` |
| M3 | `M3_corrupted_to_normal_semi_supervised` | `experiments/train_005_M3/checkpoints/best_val_psnr.pth` |
| M4 | `M4_corrupted_to_normal_self_supervised` | `experiments/train_006_M4/checkpoints/best_val_psnr.pth` |
| M5 | `M5_normal_to_clean_supervised` | `experiments/train_008_M5/checkpoints/best_val_psnr.pth` |
| M6 | `M6_normal_to_clean_unsupervised` | `experiments/train_009_M6/checkpoints/best_val_psnr.pth` |
| M7 | `M7_normal_to_clean_semi_supervised` | `experiments/train_010_M7/checkpoints/best_val_psnr.pth` |
| M8 | `M8_normal_to_clean_self_supervised` | `experiments/train_011_M8/checkpoints/best_val_psnr.pth` |

## Data Usage

M1-M4:

- Input: `data/dirty/`
- Target: `data/normal/`

M5-M8:

- Input: `data/normal/`
- Target: `data/clean/`

## Filename And Label Rule

Filename format:

```text
2_Q2M8X.png
```

The label is the text after the first underscore.

Example:

```text
2_Q2M8X.png -> Q2M8X
```
