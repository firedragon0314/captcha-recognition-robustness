@echo off
setlocal
cd /d "C:\Users\buckl\.spyder-py3\智慧學習\captchas"
"C:\Users\buckl\anaconda3\envs\env_py_3_12\python.exe" "C:\Users\buckl\.spyder-py3\智慧學習\captchas\experiments_backbone_aligned.py" --epochs 200 --lr 0.000001 --exp_name train_015_pure_resnet_multihead_consistency_seed3027_ratio70_nostn --train_mode consistency --batch_size 32 --seed 3027 --num_workers 0 --train_rotation 5.0 --model_variant pure_resnet_multihead --labeled_ratio 0.7 --pseudo_label_threshold 0.95 --contrastive_weight 1.0 --consistency_weight 1.0 --warmup_epochs 20 --stn_freeze_epochs 15 --stn_lr_scale 0.1 --stn_identity_weight 0.001 --resume 1>> "C:\Users\buckl\.spyder-py3\智慧學習\captchas\experiments\train_015_pure_resnet_multihead_consistency_seed3027_ratio70_nostn\resume_151_200_stdout.log" 2>> "C:\Users\buckl\.spyder-py3\智慧學習\captchas\experiments\train_015_pure_resnet_multihead_consistency_seed3027_ratio70_nostn\resume_151_200_stderr.log"
endlocal
