# Module B Archive

This folder contains the organized Module B files from `final project_葉俊廷-20260614T053603Z-3-*.zip`.

## Folders

- `archives/`: Original zip files, all 9 parts.
- `source_code/`: Main training/evaluation source code copied from the extracted project.
- `training_outputs/`: The 8 official Module B experiment folders. Each experiment keeps only its final `latest.pth`.
- `results_summary/`: B result summaries, plots, and B chain-ranking chart files.
- `reports/`: B report files and extracted report text.

## Checkpoint Policy

Only final checkpoint files named `latest.pth` are kept, directly under each experiment folder in the GitHub-ready copy. Intermediate `epoch_*.pth`, `latest_epoch_*.pth`, and `best_*.pth` files were removed or skipped during extraction to save disk space.

## Official Experiments

- `B1_01_supervised_crnn`
- `B1_02_contrastive_crnn`
- `B1_03_semi_supervised_crnn`
- `B1_04_self_supervised_crnn`
- `B2_01_supervised_transformer`
- `B2_02_contrastive_transformer`
- `B2_03_semi_supervised_transformer`
- `B2_04_self_supervised_transformer`
