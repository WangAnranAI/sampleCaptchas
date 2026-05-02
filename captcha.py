import os
import sys
import numpy as np
from PIL import Image


class Captcha(object):
    # Image layout (60w × 30h): 8-pixel char cells separated by 1-pixel grid lines.
    # Outer frame occupies cols 0-4 (left) and 49-59 (right), rows 0-8 (top) and 24-29 (bottom).
    _CHAR_COLS = [(5, 13), (14, 22), (23, 31), (32, 40), (41, 49)]
    _ROW_SLICE = slice(9, 23)
    _TEMPLATES_FILE = 'templates.npz'

    def __init__(self):
        base = os.path.dirname(os.path.abspath(__file__))
        templates_path = os.path.join(base, self._TEMPLATES_FILE)
        if os.path.exists(templates_path):
            data = np.load(templates_path)
            self.templates = {k: data[k] for k in data.files}
        else:
            self.templates = self._build_templates(
                os.path.join(base, 'input'),
                os.path.join(base, 'output'),
            )
            np.savez(templates_path, **self.templates)

    def _load_gray(self, im_path):
        return np.array(Image.open(im_path).convert('L'), dtype=np.float32)

    def _split(self, img):
        return [img[self._ROW_SLICE, c0:c1] for c0, c1 in self._CHAR_COLS]

    def _build_templates(self, input_dir, output_dir):
        buckets = {}
        for idx in range(25):
            img_path = os.path.join(input_dir, f'input{idx:02d}.jpg')
            lbl_path = os.path.join(output_dir, f'output{idx:02d}.txt')
            if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
                continue
            img = self._load_gray(img_path)
            label = open(lbl_path).read().strip()
            for char, patch in zip(label, self._split(img)):
                buckets.setdefault(char, []).append(patch)
        if not buckets:
            raise FileNotFoundError(
                f'No training data found in {input_dir} / {output_dir} '
                f'and no cached templates at {self._TEMPLATES_FILE}'
            )
        return {c: np.mean(ps, axis=0) for c, ps in buckets.items()}

    def _rank_patch(self, patch):
        # Sorted ascending by SSD; first is best match, second is runner-up.
        return sorted(
            (float(np.sum((patch - tmpl) ** 2)), char)
            for char, tmpl in self.templates.items()
        )

    def predict_detailed(self, im_path):
        """Return (text, per_char) where per_char is a list of dicts with
        keys 'char', 'best_ssd', 'second_ssd', 'margin'. margin = second/best;
        higher means the winning template fits much better than the next one."""
        img = self._load_gray(im_path)
        per_char = []
        for patch in self._split(img):
            ranking = self._rank_patch(patch)
            best_ssd, best_char = ranking[0]
            second_ssd = ranking[1][0]
            margin = second_ssd / best_ssd if best_ssd > 0 else float('inf')
            per_char.append({
                'char': best_char,
                'best_ssd': best_ssd,
                'second_ssd': second_ssd,
                'margin': margin,
            })
        return ''.join(c['char'] for c in per_char), per_char

    def __call__(self, im_path, save_path):
        """
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        text, _ = self.predict_detailed(im_path)
        with open(save_path, 'w') as f:
            f.write(text + '\n')
        return text


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: python {sys.argv[0]} <image_path> <output_path>')
        sys.exit(1)
    solver = Captcha()
    text, info = solver.predict_detailed(sys.argv[1])
    with open(sys.argv[2], 'w') as f:
        f.write(text + '\n')
    print(f'Prediction: {text}')
    for i, c in enumerate(info, 1):
        print(f'  pos {i}: {c["char"]}  margin={c["margin"]:.2f}x  (ssd best={c["best_ssd"]:.0f}, 2nd={c["second_ssd"]:.0f})')
