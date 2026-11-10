
"""Experiment 05: adaptive-threshold DAPI nuclear segmentation + watershed.

Input:
    data/raw/blank/*.tif

Expected raw TIFF shape:
    Z, C, Y, X
    C0 = DAPI, C1 = SOX2, C2 = GATA3, C3 = SOX17

This experiment only evaluates DAPI nuclear segmentation.
It does NOT determine SOX2/GATA3/SOX17 positivity.

Outputs:
    results/counting/adaptive_watershed_v1/
        dapi_max/
        overlays/
        labels/
        dapi_counts.csv
"""

from pathlib import Path
import csv
import re

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import watershed, find_boundaries


# --------------------------------------------------
# 1. Paths and parameters
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/raw/blank"
MANUAL_CSV = ROOT / "data/ground_truth/manual_counts.csv"
OUT = ROOT / "results/counting/adaptive_watershed_v1"

DAPI_CHANNEL = 0

# Light smoothing to reduce pixel noise without excessive blurring.
SMOOTH_SIGMA = 1.0

# Size of the neighbourhood used to estimate local background.
LOCAL_WINDOW = 31

# Local threshold = local mean + LOCAL_STD_FACTOR * local standard deviation.
LOCAL_STD_FACTOR = 0.6

# Background intensity floor, in robust standard-deviation units.
BACKGROUND_FLOOR_FACTOR = 2.5

# Watershed parameters.
PEAK_DISTANCE = 5
MIN_SEED_DISTANCE_VALUE = 2.0

# Initial plausible nucleus area, in pixels.
MIN_AREA = 15
MAX_AREA = 900


# --------------------------------------------------
# 2. Read the manual count table
# --------------------------------------------------

def read_manual_counts(path):
    """Read manual DAPI counts, if the CSV contains them."""
    if not path.exists():
        print(f"Manual count file not found: {path}")
        print("The experiment will run without manual-count comparison.")
        return {}

    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    header_index = next(
        (
            i for i, row in enumerate(rows)
            if any(cell.strip() == "Blastoid ID" for cell in row)
        ),
        None,
    )

    if header_index is None:
        raise ValueError(
            "Cannot find 'Blastoid ID' in manual_counts.csv."
        )

    headers = [cell.strip() for cell in rows[header_index]]
    manual = {}

    for row in rows[header_index + 1:]:
        row = row + [""] * max(0, len(headers) - len(row))
        record = dict(zip(headers, [cell.strip() for cell in row]))

        try:
            sample_id = int(float(record.get("Blastoid ID", "")))
        except (ValueError, TypeError):
            continue

        method = record.get("Method", "").strip().lower()
        if method and method != "manual":
            continue

        try:
            count = int(float(record.get("DAPI", "")))
        except (ValueError, TypeError):
            count = None

        manual[sample_id] = count

    return manual


# --------------------------------------------------
# 3. Extract the DAPI MAX projection from raw TIFF
# --------------------------------------------------

def read_dapi_max(path):
    """Read a raw ZCYX TIFF and take the maximum over Z."""
    image = tifffile.imread(str(path))

    if image.ndim == 4:
        # Expected shape: Z, C, Y, X.
        if image.shape[1] != 4:
            raise ValueError(
                f"{path.name}: expected 4 channels in ZCYX data; "
                f"received shape {image.shape}"
            )

        dapi_stack = image[:, DAPI_CHANNEL, :, :]
        dapi_max = np.max(dapi_stack, axis=0)

    elif image.ndim == 3:
        # Also allow an already-projected CYX image.
        if image.shape[0] != 4:
            raise ValueError(
                f"{path.name}: expected CYX data with 4 channels; "
                f"received shape {image.shape}"
            )

        dapi_max = image[DAPI_CHANNEL, :, :]

    else:
        raise ValueError(
            f"{path.name}: expected ZCYX or CYX raw data; "
            f"received shape {image.shape}"
        )

    return dapi_max.astype(np.float32)


# --------------------------------------------------
# 4. Adaptive foreground mask
# --------------------------------------------------

def make_adaptive_mask(image):
    """
    Estimate local background and noise.

    A pixel is foreground only when it exceeds both:
      1. Its local mean + a multiple of local standard deviation.
      2. A robust global background floor.

    The floor is intended to reduce background-noise detections.
    Its effectiveness must be checked on real images.
    """
    smooth = ndi.gaussian_filter(image, sigma=SMOOTH_SIGMA)

    # Local mean and standard deviation.
    local_mean = ndi.uniform_filter(
        smooth, size=LOCAL_WINDOW, mode="reflect"
    )

    local_mean_square = ndi.uniform_filter(
        smooth ** 2, size=LOCAL_WINDOW, mode="reflect"
    )

    local_variance = np.maximum(
        local_mean_square - local_mean ** 2, 0
    )
    local_std = np.sqrt(local_variance)

    local_threshold = (
        local_mean + LOCAL_STD_FACTOR * local_std
    )

    # Robust estimate of background level and noise.
    global_median = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - global_median)))
    robust_std = 1.4826 * mad

    background_floor = (
        global_median + BACKGROUND_FLOOR_FACTOR * robust_std
    )

    threshold = np.maximum(local_threshold, background_floor)
    mask = smooth > threshold

    # Remove isolated tiny foreground components before watershed.
    components, n_components = ndi.label(mask)
    if n_components:
        sizes = np.bincount(components.ravel())
        keep = sizes >= MIN_AREA
        keep[0] = False
        mask = keep[components]

    return smooth, mask, threshold, background_floor


# --------------------------------------------------
# 5. Watershed: separate adjacent nuclei
# --------------------------------------------------

def segment_dapi(image):
    smooth, mask, threshold, background_floor = (
        make_adaptive_mask(image)
    )

    if not mask.any():
        return np.zeros(image.shape, np.int32), {
            "foreground_pixels": 0,
            "background_floor": background_floor,
        }

    # Distance from each foreground pixel to the nearest background.
    distance = ndi.distance_transform_edt(mask)

    # Peaks of the distance map act as nucleus-centre seeds.
    coords = __import__(
        "skimage.feature", fromlist=["peak_local_max"]
    ).peak_local_max(
        distance,
        min_distance=PEAK_DISTANCE,
        threshold_abs=MIN_SEED_DISTANCE_VALUE,
        labels=mask.astype(np.uint8),
        exclude_border=False,
    )

    seeds = np.zeros(mask.shape, dtype=np.int32)

    for seed_id, (y, x) in enumerate(coords, start=1):
        seeds[y, x] = seed_id

    if len(coords) == 0:
        # Fallback: retain connected foreground components.
        labels, _ = ndi.label(mask)
    else:
        labels = watershed(-distance, seeds, mask=mask)

    # Filter implausibly small or large objects and relabel sequentially.
    cleaned = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1

    for region in regionprops(labels):
        if MIN_AREA <= region.area <= MAX_AREA:
            cleaned[labels == region.label] = next_id
            next_id += 1

    diagnostics = {
        "foreground_pixels": int(mask.sum()),
        "background_floor": float(background_floor),
        "seed_count": int(len(coords)),
    }

    return cleaned, diagnostics


# --------------------------------------------------
# 6. Save MAX projection, label image and overlay
# --------------------------------------------------

def save_overlay(image, labels, output_path):
    """Save a review image; cyan outlines indicate detected objects."""
    low, high = np.percentile(image, [1, 99.5])

    if high <= low:
        display = np.zeros(image.shape, dtype=np.uint8)
    else:
        display = np.clip(
            (image - low) / (high - low) * 255, 0, 255
        ).astype(np.uint8)

    rgb = np.repeat(display[:, :, None], 3, axis=2)

    boundaries = find_boundaries(labels, mode="outer")
    rgb[boundaries] = [0, 255, 255]

    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)

    for region in regionprops(labels):
        y, x = region.centroid
        radius = 3
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 60, 60),
            width=1,
        )

    canvas.save(output_path)


def get_sample_id(filename):
    """Map Position001-1.tif to sample ID 1, etc."""
    match = re.search(r"Position\d+-(\d+)", filename)
    if not match:
        return None
    return int(match.group(1))


# --------------------------------------------------
# 7. Batch process the raw/blank folder
# --------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    max_dir = OUT / "dapi_max"
    overlay_dir = OUT / "overlays"
    labels_dir = OUT / "labels"

    for folder in (max_dir, overlay_dir, labels_dir):
        folder.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT.glob("*.tif"))
    files += sorted(INPUT.glob("*.tiff"))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"No TIFF files found in {INPUT}"
        )

    manual = read_manual_counts(MANUAL_CSV)
    summary = []

    print(f"Found {len(files)} raw TIFF files in {INPUT}")

    for path in files:
        sample_id = get_sample_id(path.name)

        if sample_id is None:
            print(f"Skipping unrecognised filename: {path.name}")
            continue

        try:
            dapi_max = read_dapi_max(path)
            labels, diagnostics = segment_dapi(dapi_max)

            detected_count = int(labels.max())
            manual_count = manual.get(sample_id)

            stem = path.stem

            tifffile.imwrite(
                str(max_dir / f"{stem}_DAPI_max.tif"),
                dapi_max.astype(np.float32),
            )

            tifffile.imwrite(
                str(labels_dir / f"{stem}_DAPI_labels.tif"),
                labels.astype(np.uint16),
            )

            save_overlay(
                dapi_max,
                labels,
                overlay_dir / f"{stem}_DAPI_overlay.png",
            )

            difference = (
                detected_count - manual_count
                if manual_count is not None else None
            )

            summary.append({
                "sample_id": sample_id,
                "source_file": path.name,
                "manual_DAPI_count": manual_count,
                "adaptive_watershed_count": detected_count,
                "count_difference_auto_minus_manual": difference,
                "foreground_pixels": diagnostics["foreground_pixels"],
                "seed_count": diagnostics.get("seed_count"),
                "background_floor": diagnostics["background_floor"],
                "status": "OK",
            })

            print(
                f"Blastoid {sample_id}: "
                f"DAPI detected={detected_count}, "
                f"manual={manual_count}"
            )

        except Exception as exc:
            print(f"ERROR processing {path.name}: {exc}")

            summary.append({
                "sample_id": sample_id,
                "source_file": path.name,
                "manual_DAPI_count": manual.get(sample_id),
                "adaptive_watershed_count": None,
                "count_difference_auto_minus_manual": None,
                "foreground_pixels": None,
                "seed_count": None,
                "background_floor": None,
                "status": f"ERROR: {exc}",
            })

    csv_path = OUT / "dapi_counts.csv"

    fieldnames = [
        "sample_id",
        "source_file",
        "manual_DAPI_count",
        "adaptive_watershed_count",
        "count_difference_auto_minus_manual",
        "foreground_pixels",
        "seed_count",
        "background_floor",
        "status",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print("\nExperiment complete.")
    print(f"Processed samples: {len(summary)}")
    print(f"Count report: {csv_path}")
    print(f"Overlays: {overlay_dir}")
    print(f"Label images: {labels_dir}")


if __name__ == "__main__":
    main()
