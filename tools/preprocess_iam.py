"""
preprocess_iam.py
-----------------
Converts the Kaggle IAM words dataset (archive/iam_words) into the
data/ directory structure expected by ScribeSynth.

Usage:
    python tools/preprocess_iam.py --archive archive/iam_words

Output:
    data/
        images/train/<form_id>/*.png   (3-channel, height-normalised)
        images/test/<form_id>/*.png
        style/train/<form_id>/*.png    (grayscale strips)
        style/test/<form_id>/*.png
        freq/train/<form_id>/*.png     (Laplacian edge-filtered strips)
        freq/test/<form_id>/*.png
        IAM_train.txt
        IAM_test.txt
"""

import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm

# Vocabulary that the model knows about (must match dataset/loader.py)
LETTERS = '_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
VALID_CHARS = set(LETTERS)
TARGET_HEIGHT = 64   # resize all images to this height
TRAIN_RATIO   = 0.9  # 90% writers for training
MIN_SAMPLES   = 3    # minimum word samples per writer (form)


def apply_freq_filter(gray_img):
    """Simple Laplacian edge map — provides initial high-freq input.
    The LearnableFrequencyFilter will refine this during training."""
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    lap = np.clip(np.absolute(lap), 0, 255).astype(np.uint8)
    return lap


def parse_words_txt(words_txt):
    """Parse words.txt and return list of valid word entries."""
    entries = []
    with open(words_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(' ')
            if len(parts) < 9:
                continue

            image_id   = parts[0]   # e.g. a01-000u-00-00
            status     = parts[1]   # ok / er
            transcription = parts[-1]

            if status != 'ok':
                continue
            # Only keep words whose every character is in the model vocabulary
            if not all(c in VALID_CHARS for c in transcription):
                continue
            # Max word length the model handles
            if len(transcription) > 32:
                continue

            id_parts = image_id.split('-')
            if len(id_parts) < 4:
                continue

            group     = id_parts[0]                    # a01
            form_id   = id_parts[0] + '-' + id_parts[1]  # a01-000u  (= writer proxy)
            image_file = image_id + '.png'

            entries.append({
                'image_id':     image_id,
                'writer_id':    form_id,
                'image_file':   image_file,
                'rel_path':     os.path.join(group, form_id, image_file),
                'transcription': transcription,
            })
    return entries


def main(args):
    words_txt  = os.path.join(args.archive, 'words.txt')
    words_dir  = os.path.join(args.archive, 'words')
    output_dir = args.output

    print(f'Parsing {words_txt} ...')
    entries = parse_words_txt(words_txt)
    print(f'  Valid entries: {len(entries)}')

    # Group by writer (form)
    by_writer = {}
    for e in entries:
        by_writer.setdefault(e['writer_id'], []).append(e)

    # Keep only writers with enough samples
    by_writer = {k: v for k, v in by_writer.items() if len(v) >= MIN_SAMPLES}
    print(f'  Writers with >= {MIN_SAMPLES} samples: {len(by_writer)}')

    # Reproducible train / test split by writer
    all_writers = sorted(by_writer.keys())
    random.seed(42)
    random.shuffle(all_writers)
    split       = int(len(all_writers) * TRAIN_RATIO)
    train_set   = set(all_writers[:split])
    test_set    = set(all_writers[split:])

    # Create output directories
    for split_name in ('train', 'test'):
        for folder in ('images', 'style', 'freq'):
            os.makedirs(os.path.join(output_dir, folder, split_name), exist_ok=True)

    train_lines, test_lines = [], []
    skipped = 0

    print('Processing images ...')
    for writer_id, writer_entries in tqdm(by_writer.items()):
        split_name = 'train' if writer_id in train_set else 'test'
        text_list  = train_lines if split_name == 'train' else test_lines

        for folder in ('images', 'style', 'freq'):
            os.makedirs(
                os.path.join(output_dir, folder, split_name, writer_id),
                exist_ok=True)

        for e in writer_entries:
            src = os.path.join(words_dir, e['rel_path'])
            if not os.path.exists(src):
                skipped += 1
                continue

            gray = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                skipped += 1
                continue

            # Resize to TARGET_HEIGHT keeping aspect ratio
            h, w = gray.shape
            new_w = max(1, int(w * TARGET_HEIGHT / h))
            gray  = cv2.resize(gray, (new_w, TARGET_HEIGHT))

            img_file = e['image_file']

            # images/ — 3-channel (model VAE expects RGB)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(output_dir, 'images', split_name, writer_id, img_file), rgb)

            # style/ — grayscale
            cv2.imwrite(os.path.join(output_dir, 'style', split_name, writer_id, img_file), gray)

            # freq/ — Laplacian edge map
            freq = apply_freq_filter(gray)
            cv2.imwrite(os.path.join(output_dir, 'freq', split_name, writer_id, img_file), freq)

            # Text file entry:  <writer_id>,<image_name_no_ext> <transcription>
            name_no_ext = img_file.replace('.png', '')
            text_list.append(f'{writer_id},{name_no_ext} {e["transcription"]}')

    # Write split text files
    with open(os.path.join(output_dir, 'IAM_train.txt'), 'w') as f:
        f.write('\n'.join(train_lines))
    with open(os.path.join(output_dir, 'IAM_test.txt'), 'w') as f:
        f.write('\n'.join(test_lines))

    print(f'\nDone!')
    print(f'  Train samples : {len(train_lines)}  ({len(train_set)} writers)')
    print(f'  Test  samples : {len(test_lines)}  ({len(test_set)} writers)')
    if skipped:
        print(f'  Skipped       : {skipped}  (missing/unreadable images)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default='archive/iam_words',
                        help='Path to the extracted iam_words folder')
    parser.add_argument('--output', default='data',
                        help='Output data directory (default: data/)')
    args = parser.parse_args()
    main(args)
