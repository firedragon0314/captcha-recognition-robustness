# Module C Archive

This folder contains the organized files that were outside `A/` and `B/`.

## Folders

- `source_code/`: Root-level Python scripts and project helper code.
- `training_outputs/`: Module C experiment folders. Each experiment keeps only `latest.pth`.
- `results_summary/`: Generated summaries, charts, renders, downstream evaluation outputs, and report assets.
- `reports/`: Remaining report documents and notes.
- `commands/`: Command files and run scripts.
- `archives/`: Legacy root archive folders, if any.
- `organization/`: Previous organization manifests and notes.
- `cache/`: Python cache files moved out of the root.
- `temp/`: Temporary working files moved out of the root.

## Dataset Cleanup

The root image dataset folders `train/`, `val/`, and `test/` were removed. The extracted A image-dataset backup under `A/archives/extracted/` was also removed. Small result images such as plots, report renders, previews, and failure-case examples were kept.

## Checkpoint Policy

Module C experiment checkpoints keep only `latest.pth`. Intermediate `epoch_*.pth`, `latest_epoch_*.pth`, and `best_*.pth` files were removed to save disk space.
