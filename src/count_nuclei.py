"""Experimental 2D nucleus-centre detection baseline for blastoid fluorescence images.

NOT validated for biological counting. Outputs candidate points for review.
This is not a cell segmentation model and does not infer marker positivity.

Usage:
    python blastoid_nucleus_counter_v1.py --input B01_SOX2_max.tif --output results

Dependencies: numpy, scipy, scikit-image, tifffile, pillow
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from PIL import Image, ImageDraw, ImageFont


def normalize_image(img):
    """Scale high-end grayscale intensity to 255 (for candidate detection only)."""
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D image; got {img.shape}. Export one marker channel first.")
    arr = img.astype(np.float32)
    lo = np.percentile(arr, 0.1)
    hi = np.percentile(arr, 99.9)
    if hi <= lo:
        raise ValueError("Image has no usable intensity variation")
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255)


def candidates(image, smooth_sigma=2.4, background_sigma=11.0,
               min_distance=8, contrast_threshold=10.0, min_smoothed_intensity=18.0):
    im = normalize_image(image)
    smooth = gaussian_filter(im, smooth_sigma)
    background = gaussian_filter(im, background_sigma)
    contrast = np.maximum(smooth - background, 0)
    points = peak_local_max(
        contrast, min_distance=min_distance,
        threshold_abs=contrast_threshold,
        exclude_border=False,
    )
    records = []
    for y, x in points:
        if smooth[y, x] < min_smoothed_intensity:
            continue
        records.append(dict(x=int(x), y=int(y), local_contrast=float(contrast[y, x]),
                            smoothed_intensity=float(smooth[y, x]),
                            raw_intensity=float(im[y, x])))
    records.sort(key=lambda r: (r['y'], r['x']))
    return records, im, contrast


def annotated_overlay(im, records, count, scale=3):
    base = Image.fromarray(np.uint8(np.clip(im, 0, 255)), mode='L').convert('RGB')
    base = base.resize((base.width * scale, base.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12 * scale // 2)
    except OSError:
        font = ImageFont.load_default()
    for i, r in enumerate(records, 1):
        x = r['x'] * scale + scale // 2
        y = r['y'] * scale + scale // 2
        radius = 3 * scale
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=(30, 255, 100), width=2)
        draw.text((x + radius + 1, y - radius), str(i), font=font, fill=(255, 220, 30))
    return base


def run(argv=None):
    parser = argparse.ArgumentParser(description='Experimental blob/centre detection baseline')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', default=Path('candidate_results'), type=Path)
    parser.add_argument('--min-distance', type=int, default=8, help='minimum peak spacing, in pixels')
    parser.add_argument('--contrast-threshold', type=float, default=10.0)
    parser.add_argument('--min-smoothed-intensity', type=float, default=18.0)
    parser.add_argument('--smooth-sigma', type=float, default=2.4)
    parser.add_argument('--background-sigma', type=float, default=11.0)
    args = parser.parse_args(argv)
    if args.smooth_sigma >= args.background_sigma:
        parser.error('smooth-sigma must be smaller than background-sigma')
    image = tifffile.imread(str(args.input))
    rec, normal, contrast = candidates(image,
        smooth_sigma=args.smooth_sigma, background_sigma=args.background_sigma,
        min_distance=args.min_distance, contrast_threshold=args.contrast_threshold,
        min_smoothed_intensity=args.min_smoothed_intensity,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    with (args.output / f'{stem}_candidate_centres.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id','x','y','local_contrast','smoothed_intensity','raw_intensity'])
        writer.writeheader()
        for i, row in enumerate(rec, 1):
            writer.writerow({'id':i,**row})
    overlay = annotated_overlay(normal, rec, len(rec))
    overlay.save(args.output / f'{stem}_candidate_overlay.png')
    Image.fromarray(np.uint8(np.clip(contrast/np.max(contrast)*255 if contrast.max() else contrast,0,255))).save(
        args.output / f'{stem}_contrast_map.png')
    print(f'{len(rec)} candidate nucleus centres — UNVERIFIED; not a biological count.')
    print('Files saved to:', args.output.resolve())

if __name__ == '__main__':
    run()
