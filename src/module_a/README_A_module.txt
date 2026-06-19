A Module Handoff

This package contains:
1. M1~M8 trained DnCNN models, separated by model folder.
2. Shared image datasets: dirty, normal, clean.
3. CSV files for label/path handoff.
4. Training code.

Dataset folders:
- data/dirty  : corrupted CAPTCHA images
- data/normal : original normal CAPTCHA images
- data/clean  : generated clean CAPTCHA images

Model mapping:
- M1: dirty -> normal, supervised
- M2: dirty -> normal, unsupervised
- M3: dirty -> normal, semi-supervised
- M4: dirty -> normal, self-supervised
- M5: normal -> clean, supervised
- M6: normal -> clean, unsupervised
- M7: normal -> clean, semi-supervised
- M8: normal -> clean, self-supervised

Current best model:
M1 is the best restoration model.
M1 experiment folder: train_002_M1
M1 val_psnr = 22.944
M1 val_ssim = 0.8699

Filename format:
2_Q2M8X.png

Label extraction:
The label is the text after the first underscore.
Example:
2_Q2M8X.png -> Q2M8X
