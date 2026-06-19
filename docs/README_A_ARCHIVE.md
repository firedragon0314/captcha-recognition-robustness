# Module A Archive

This folder contains the organized Module A files.

## Folders

- `archives/`: Original A zip files. Extracted image-dataset backups were removed to save space.
- `source_code/`: Module A training, export, status-check, and data generation scripts.
- `training_outputs/`: Clean M1-M8 model outputs. Each model folder keeps one final `model.pth`, `history.csv`, and `model_info.txt`.
- `results_summary/`: Dataset manifests, model summary CSV files, and denoising preview images.
- `reports/`: Module A report files and extracted report text.

## Model Outputs

The clean training outputs are in `training_outputs/models/`:

- `M1_corrupted_to_normal_supervised`
- `M2_corrupted_to_normal_unsupervised`
- `M3_corrupted_to_normal_semi_supervised`
- `M4_corrupted_to_normal_self_supervised`
- `M5_normal_to_clean_supervised`
- `M6_normal_to_clean_unsupervised`
- `M7_normal_to_clean_semi_supervised`
- `M8_normal_to_clean_self_supervised`

Only one final model file, `model.pth`, is kept for each M1-M8 output folder.
