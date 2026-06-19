# Robust CAPTCHA Recognition

This repository contains the organized source code, experiment summaries, reports, and reproducibility notes for a three-part CAPTCHA robustness project.

## Project Scope

- Module A: CAPTCHA image restoration and data preparation.
- Module B: sequence-based CAPTCHA recognition with CRNN / Transformer + CTC.
- Module C: position-wise multi-head CAPTCHA classification.

## Repository Structure

```text
.
|-- src/
|   |-- module_a/          # Module A source code
|   |-- module_b/          # Module B source code
|   `-- module_c/          # Module C experiment helpers and reporting scripts
|-- experiments/
|   |-- module_a/          # M1-M8 restoration model artifacts under 100 MB
|   |-- module_b/          # 8 official Module B experiment outputs
|   `-- module_c/          # Module C experiment outputs, without >100 MB checkpoints
|-- results/
|   |-- module_a/          # Module A summaries and preview figures
|   |-- module_b/          # Module B summaries, charts, and experiment_summaries/
|   `-- module_c/          # Module C summaries, charts, report assets, and evaluation outputs
|-- reports/               # Module report documents
|-- scripts/               # Convenience command files
`-- docs/                  # Slides, proposal, archive notes, and artifact manifests
```

## Large Artifact Policy

GitHub blocks files larger than 100 MB. This repository excludes only oversized artifacts that cannot be committed normally:

- 8 final Module C `latest.pth` checkpoint files.
- 1 oversized Module C `predictions.csv`.

Artifact references are documented in:

- `docs/DRIVE_ARTIFACTS.md`
- `docs/drive_upload_queue.csv`
- `docs/large_artifacts_manifest.csv`
- `docs/not_uploaded_original_archives.csv`
- `docs/log_artifacts_manifest.csv`

Small model artifacts under 100 MB, including Module A `model.pth` files and Module B final `latest.pth` files, are included.

## Dataset Note

The raw image dataset folders `train/`, `val/`, and `test/` are not included. They were intentionally removed from this GitHub-ready copy because the dataset is stored separately.

Expected dataset layout when restoring locally:

```text
train/
val/
test/
```

## Install

```bash
pip install -r requirements.txt
```

Module B also keeps a local requirements file at `src/module_b/requirements.txt`; it currently matches the root dependency list.

## Main Entry Points

- Module A training: `src/module_a/train_dncnn_a_models.py`
- Module B training scripts: `src/module_b/train_*.py`
- Module C experiments: `src/module_c/run_experiment.py`

## Notes

- Logs are treated as run artifacts. They are ignored by `.gitignore`; retained log locations are listed in `docs/log_artifacts_manifest.csv`.
- Full raw datasets and original zip archives are intentionally not part of this repository.
