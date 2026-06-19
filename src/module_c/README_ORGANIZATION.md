# Module C Organization

Module C treats each CAPTCHA as a fixed five-character string. A shared encoder extracts image features, and five classification heads predict the character at each position.

## Main Files

- `run_experiment.py`: main Module C training and evaluation entry point.
- `experiments.py`: experiment definitions and shared training helpers.
- `experiments_backbone_aligned.py`: backbone-aligned experiment variants.
- `evaluate_downstream_88.py`: downstream evaluation across restoration and recognition chains.
- `compare_runs.py`: utilities for comparing experiment outputs.
- `generate_chain_ranking_charts.py`: chart generation for chain-level rankings.
- `build_module_c_report.py`: report-building helper.

## Related Folders

- `experiments/module_c/`: official Module C experiment outputs and checkpoint placeholders.
- `results/module_c/`: generated charts, report assets, and downstream evaluation summaries.
- `docs/DRIVE_ARTIFACTS.md`: restore instructions for oversized checkpoints and prediction artifacts.

## Artifact Policy

Large Module C checkpoints over 100 MB are not committed. Download links and restore locations are listed in `docs/DRIVE_ARTIFACTS.md` and `docs/large_artifacts_manifest.csv`.
