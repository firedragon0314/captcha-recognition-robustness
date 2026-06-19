A Module Code Files

Main training script:
- train_dncnn_a_models.py

Helper scripts:
- generate_clean_from_normal.py
- export_restored_from_checkpoint.py
- check_project_status.py

Common commands:
1) Generate corrupted images and test M1:
python train_dncnn_a_models.py --data_root data --make_corrupted --overwrite_corrupted --model_id M1 --epochs 1 --batch_size 4 --collect_failures --export_restored

2) Train one model:
python train_dncnn_a_models.py --data_root data --model_id M1 --epochs 20 --batch_size 64 --num_workers 0

3) Resume training:
python train_dncnn_a_models.py --data_root data --model_id M3 --epochs 20 --batch_size 32 --num_workers 0 --resume_checkpoint "experiments\train_005_M3\checkpoints\latest.pth"

4) Generate clean images:
python generate_clean_from_normal.py

5) Export restored images from checkpoint:
python export_restored_from_checkpoint.py --data_root data --source_folder corrupted --checkpoint "experiments\train_002_M1\checkpoints\best_val_psnr.pth" --exp_dir "experiments\train_002_M1"

6) Check all project status:
python check_project_status.py
