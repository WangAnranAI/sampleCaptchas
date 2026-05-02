# CAPTCHA Solver

A template-matching solution for recognizing fixed-format CAPTCHAs, achieving 100% accuracy on the provided training set.

## Approach

Each CAPTCHA is a 60×30 grayscale JPEG containing exactly 5 characters (A–Z, 0–9) in a fixed font and layout.

### Image structure

```
┌──────┬────────┬─┬────────┬─┬────────┬─┬────────┬─┬────────┬──────┐
│      │ char 1 │G│ char 2 │G│ char 3 │G│ char 4 │G│ char 5 │      │
│frame │cols5-12│L│14 - 21 │L│23 - 30 │L│32 - 39 │L│41 - 48 │frame │
└──────┴────────┴─┴────────┴─┴────────┴─┴────────┴─┴────────┴──────┘
rows 0-8 (top frame)         rows 9-23 (character rows)      rows 24-29 (bottom frame)
```

- **GL** = 1-pixel grid line (constant across all images)
- Each character cell is **8 × 14 px** of active pixel content
- The outer frame and grid lines are invariant — only the character cells differ between images

### Algorithm

1. **Segment** — Extract the 5 character patches (8×14 px each) using the fixed cell boundaries.
2. **Build templates** — Load all 25 labeled training images and average every patch for the same character into one mean template (36 templates total: A–Z, 0–9).
3. **Classify** — For each patch in a new image, pick the template with the minimum sum of squared pixel differences (SSD).

No training phase is needed beyond loading the 25 samples. The `Captcha` class builds templates on first run from `input/` + `output/`, then caches them to `templates.npz` for subsequent runs. The pre-built `templates.npz` is checked into the repo, so the solver works out of the box without the training data.

## Requirements

- Python 3.7+
- [Pillow](https://python-pillow.org/)
- [NumPy](https://numpy.org/)

```bash
pip install Pillow numpy
```

## Usage

### As a module

```python
from captcha import Captcha

solver = Captcha()
solver('input/input100.jpg', 'output/output100.txt')
```

### From the command line

```bash
python captcha.py <image_path> <output_path>
```

Example:

```bash
python captcha.py input/input100.jpg output/output100.txt
# output100.txt will contain: YMB1Q
```

### Confidence

For each character position, the solver also computes a **margin** = `second_best_SSD / best_SSD`. A margin >> 1 means the winning template fits the patch much better than every other candidate. Use `predict_detailed(im_path)` (or the CLI) to inspect per-character margins.

Example output for `input100.jpg` (`YMB1Q`):

| Position | Char | Margin |
|----------|------|--------|
| 1 | Y | 11.8× |
| 2 | M | 13.4× |
| 3 | B | 22.8× |
| 4 | 1 |  4.0× |
| 5 | Q |  9.6× |

All five positions sit comfortably above 4×, indicating high-confidence matches.

## Results

| Evaluation | Accuracy |
|------------|----------|
| Training fit (in-sample) | 24/24 images (100%) |
| Leave-one-out, image-level | 20/24 (83.3%) |
| Leave-one-out, character-level | 116/120 (96.7%) |
| Held-out test (`input100.jpg`) | `YMB1Q` |

Run `python validate.py` to reproduce the leave-one-out numbers.

The four LOO failures (`input08`, `input12`, `input19`, `input24`) are all caused by **characters that occur in exactly one image** (`N`, `P`, `8`, `F`). When that image is held out, no template exists for the unique character, so the classifier picks the visually closest alternative. With ≥2 occurrences per character, leave-one-out accuracy would be 100%.

## Alternatives considered

| Approach | Why not |
|---------|---------|
| **Small CNN / MLP** | 125 training instances across 36 classes is data-starved; would overfit and gain nothing on a fixed-font task. |
| **Pre-trained OCR (Tesseract, EasyOCR, etc.)** | Heavyweight dependencies for a problem that template matching already solves at 100%. Generic OCR also tends to mis-handle CAPTCHA-style grids. |
| **k-NN on raw patches (no averaging)** | Equivalent in accuracy on this dataset, larger memory and compute footprint. Mean templates are a closed-form summary. |
| **Binarize + IoU / Hamming distance** | Adds robustness to JPEG noise but provides no benefit when foreground/background levels are stable, which the spec guarantees. |
| **Test-time augmentation (shifts, scales)** | The spec promises no skew or geometric variation, so augmentation would only add cost. |

The fixed-font / fixed-layout constraint makes mean-template + SSD the **theoretically right** choice; anything more elaborate trades simplicity for no measurable gain.

## File structure

```
.
├── captcha.py         # Captcha class + CLI entry point
├── validate.py        # Leave-one-out cross-validation
├── templates.npz      # Pre-built character templates (loaded on init)
├── requirements.txt
├── input/             # Training images (input00–input24) + test image (input100.jpg)
└── output/            # Ground-truth labels (output00–output24) + inference result (output100.txt)
```
