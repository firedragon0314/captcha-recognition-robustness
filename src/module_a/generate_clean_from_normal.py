from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATA_ROOT = Path('data')
IMAGE_W = 160
IMAGE_H = 60


def get_label(filename):
    stem = Path(filename).stem
    return stem.split('_', 1)[1] if '_' in stem else stem


def load_font(size=38):
    for fp in [
        r'C:\Windows\Fonts\arialbd.ttf',
        r'C:\Windows\Fonts\arial.ttf',
        r'C:\Windows\Fonts\calibrib.ttf',
        r'C:\Windows\Fonts\calibri.ttf',
    ]:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size=size)
    return ImageFont.load_default()


def make_clean_captcha(label, out_path):
    img = Image.new('RGB', (IMAGE_W, IMAGE_H), 'white')
    draw = ImageDraw.Draw(img)
    font = load_font(size=38)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (IMAGE_W - text_w) // 2
    y = (IMAGE_H - text_h) // 2 - 3
    draw.text((x, y), label, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    for split in ['train', 'val', 'test']:
        normal_dir = DATA_ROOT / 'normal' / split
        clean_dir = DATA_ROOT / 'clean' / split
        clean_dir.mkdir(parents=True, exist_ok=True)
        files = sorted([p for p in normal_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        print(f'{split}: generating {len(files)} clean images')
        for i, p in enumerate(files, start=1):
            label = get_label(p.name)
            make_clean_captcha(label, clean_dir / p.name)
            if i % 5000 == 0:
                print(f'  {split}: {i}/{len(files)}')
    print('Finished generating clean images.')


if __name__ == '__main__':
    main()
