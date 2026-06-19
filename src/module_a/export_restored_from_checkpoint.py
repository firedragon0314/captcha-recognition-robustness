from pathlib import Path
import csv
import argparse

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from train_dncnn_a_models import DnCNN, list_image_files, ensure_dir


def export_restored(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    base_channels = int(config.get('base_channels', 64))
    dncnn_depth = int(config.get('dncnn_depth', 17))
    image_w = int(config.get('image_w', 160))
    image_h = int(config.get('image_h', 60))

    model = DnCNN(in_channels=3, base_channels=base_channels, depth=dncnn_depth).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    data_root = Path(args.data_root)
    exp_dir = Path(args.exp_dir)
    restored_root = exp_dir / 'restored'
    ensure_dir(restored_root)
    rows = []

    print('Device:', device)
    print('Checkpoint:', args.checkpoint)
    print('Output:', restored_root)

    with torch.no_grad():
        for split in args.splits:
            src_dir = data_root / args.source_folder / split
            out_dir = restored_root / split
            ensure_dir(out_dir)
            files = list_image_files(src_dir)
            print(f'Exporting {split}: {len(files)} images')
            for idx, path in enumerate(files, start=1):
                img = Image.open(path).convert('RGB').resize((image_w, image_h))
                x = TF.to_tensor(img).unsqueeze(0).to(device)
                pred = model(x).cpu()[0]
                out_path = out_dir / path.name
                save_image(pred, out_path)
                label = path.stem.split('_', 1)[1] if '_' in path.stem else path.stem
                rows.append({'filename': path.name, 'label': label, 'split': split, 'source_path': str(path), 'restored_path': str(out_path), 'checkpoint': str(args.checkpoint)})
                if idx % 5000 == 0:
                    print(f'  {split}: {idx}/{len(files)}')

    csv_path = exp_dir / 'restored_labels.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'label', 'split', 'source_path', 'restored_path', 'checkpoint']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('Finished export.')
    print('Restored labels:', csv_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--source_folder', type=str, default='corrupted')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    export_restored(parse_args())
