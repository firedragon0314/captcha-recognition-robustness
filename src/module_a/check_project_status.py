from pathlib import Path
import csv

ROOT = Path('.')
DATA = ROOT / 'data'
EXP = ROOT / 'experiments'
SPLITS = ['train', 'val', 'test']
EXPECTED = {'train': 70000, 'val': 15000, 'test': 15000}


def count_files(path):
    if not path.exists():
        return 0
    return len([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])


def read_last_history(history_path):
    if not history_path.exists():
        return None
    with open(history_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def check_data_counts():
    print('=' * 80)
    print('DATASET CHECK')
    print('=' * 80)
    for folder in ['normal', 'clean', 'corrupted']:
        print(f'\n[{folder}]')
        for split in SPLITS:
            n = count_files(DATA / folder / split)
            expected = EXPECTED[split]
            status = 'OK' if n == expected else 'CHECK'
            print(f'{folder}/{split}: {n} / {expected} [{status}]')


def check_name_matching(a_folder, b_folder):
    print('=' * 80)
    print(f'FILENAME MATCH CHECK: {a_folder} vs {b_folder}')
    print('=' * 80)
    for split in SPLITS:
        a_dir = DATA / a_folder / split
        b_dir = DATA / b_folder / split
        if not a_dir.exists() or not b_dir.exists():
            print(f'{split}: folder missing')
            continue
        a_names = set(p.name for p in a_dir.iterdir() if p.is_file())
        b_names = set(p.name for p in b_dir.iterdir() if p.is_file())
        only_a = sorted(a_names - b_names)
        only_b = sorted(b_names - a_names)
        if not only_a and not only_b:
            print(f'{split}: OK, matched {len(a_names)} files')
        else:
            print(f'{split}: NOT MATCHED')
            print(f'  only in {a_folder}: {len(only_a)} examples={only_a[:5]}')
            print(f'  only in {b_folder}: {len(only_b)} examples={only_b[:5]}')


def check_experiments():
    print('=' * 80)
    print('EXPERIMENT CHECK')
    print('=' * 80)
    if not EXP.exists():
        print('experiments folder not found')
        return
    for exp_dir in sorted([p for p in EXP.iterdir() if p.is_dir() and p.name.startswith('train_')]):
        history = read_last_history(exp_dir / 'history.csv')
        ckpt_dir = exp_dir / 'checkpoints'
        print('-' * 80)
        print(exp_dir.name)
        if history is None:
            print('  history.csv: missing or empty')
        else:
            print(f"  model: {history.get('model_id')}")
            print(f"  pair: {history.get('pair')}")
            print(f"  mode: {history.get('training_mode')}")
            print(f"  epoch: {history.get('epoch')}")
            print(f"  val_psnr: {history.get('val_psnr')}")
            print(f"  val_ssim: {history.get('val_ssim')}")
            print(f"  test_psnr: {history.get('test_psnr')}")
            print(f"  test_ssim: {history.get('test_ssim')}")
        print(f"  latest.pth: {'OK' if (ckpt_dir / 'latest.pth').exists() else 'MISSING'}")
        print(f"  best_val_psnr.pth: {'OK' if (ckpt_dir / 'best_val_psnr.pth').exists() else 'MISSING'}")
        print(f"  best_val_ssim.pth: {'OK' if (ckpt_dir / 'best_val_ssim.pth').exists() else 'MISSING'}")


def main():
    check_data_counts()
    print()
    check_name_matching('normal', 'clean')
    print()
    check_name_matching('normal', 'corrupted')
    print()
    check_experiments()


if __name__ == '__main__':
    main()
