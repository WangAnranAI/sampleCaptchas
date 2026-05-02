"""Leave-one-out cross-validation for the CAPTCHA solver.

For each labeled image, rebuild templates from the remaining images and
predict the held-out one. Reports image-level and character-level accuracy,
and flags failures caused by characters that appear in only one image.

Run: python validate.py
"""
import os
import numpy as np
from PIL import Image
from captcha import Captcha


def _patches(img):
    return [img[Captcha._ROW_SLICE, c0:c1] for c0, c1 in Captcha._CHAR_COLS]


def _load_pairs(input_dir, output_dir, n=25):
    pairs = []
    for idx in range(n):
        ip = os.path.join(input_dir, f'input{idx:02d}.jpg')
        lp = os.path.join(output_dir, f'output{idx:02d}.txt')
        if os.path.exists(ip) and os.path.exists(lp):
            img = np.array(Image.open(ip).convert('L'), dtype=np.float32)
            label = open(lp).read().strip()
            pairs.append((idx, img, label))
    return pairs


def _build_templates(pairs, exclude_idx):
    buckets = {}
    for idx, img, label in pairs:
        if idx == exclude_idx:
            continue
        for char, patch in zip(label, _patches(img)):
            buckets.setdefault(char, []).append(patch)
    return {c: np.mean(ps, axis=0) for c, ps in buckets.items()}


def _predict(img, templates):
    out = ''
    for patch in _patches(img):
        out += min(templates, key=lambda c: np.sum((patch - templates[c]) ** 2))
    return out


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    pairs = _load_pairs(os.path.join(base, 'input'), os.path.join(base, 'output'))

    img_correct = 0
    char_correct = 0
    char_total = 0
    failures = []

    for held_idx, held_img, held_label in pairs:
        templates = _build_templates(pairs, exclude_idx=held_idx)
        missing = sorted(set(held_label) - set(templates))
        pred = _predict(held_img, templates)

        if pred == held_label:
            img_correct += 1
        else:
            note = f'  (missing templates after holdout: {missing})' if missing else ''
            failures.append(f'  input{held_idx:02d}: expected={held_label} got={pred}{note}')

        for p, e in zip(pred, held_label):
            if p == e:
                char_correct += 1
            char_total += 1

    n = len(pairs)
    print(f'Leave-one-out image accuracy:     {img_correct}/{n} ({100 * img_correct / n:.1f}%)')
    print(f'Leave-one-out character accuracy: {char_correct}/{char_total} ({100 * char_correct / char_total:.1f}%)')
    if failures:
        print('\nFailures:')
        for f in failures:
            print(f)


if __name__ == '__main__':
    main()
