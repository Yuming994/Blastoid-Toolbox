"""Experiment 06: focused DAPI nucleus-count diagnosis on Blank Blastoid 14.

Run from the Blastoid-Toolbox project root:
    python -m experiments.exp06_b14_method_comparison

Input:
    data/raw/blank/*Position004-14.tif[f]

Assumptions:
    - TIFF axes are ZCYX
    - C0 is DAPI
    - C1, C2, C3 are the other marker channels

This script reads the ORIGINAL multi-channel Z-stack and computes a MAX
projection of DAPI only.

It compares two unvalidated candidate-detection methods.
It does NOT use Cellpose.
It does NOT classify marker positivity.
"""

from pathlib import Path
import csv

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, blob_log
from skimage.measure import regionprops
from skimage.segmentation import watershed


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "blank"

OUT_DIR = ROOT / "results" / "counting" / "b14_diagnostics"

# Manual DAPI count for Blastoid 14.
# Used only for reporting, NOT to force the algorithm to return 106.
MANUAL_DAPI_B14 = 106


# ============================================================
# 2. INITIAL PARAMETERS
# ============================================================

# Method A: Adaptive threshold + Watershed

SMOOTH_SIGMA = 0.9
BACKGROUND_SIGMA = 12.0

LOCAL_WINDOW = 31
LOCAL_STD_FACTOR = 0.15

SEED_SPACING = 5
SEED_MIN_DISTANCE_TRANSFORM = 1.2

MIN_OBJECT_AREA = 12
MAX_OBJECT_AREA = 900


# Method B: Multiscale Laplacian of Gaussian

LOG_MIN_SIGMA = 1.7
LOG_MAX_SIGMA = 5.0
LOG_NUM_SIGMA = 8

LOG_THRESHOLD = 0.025
LOG_OVERLAP = 0.55


# ============================================================
# 3. FIND AND LOAD ORIGINAL BLASTOID 14
# ============================================================

def find_original_b14():
    """Find the original Position004-14 TIFF."""

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            "Raw-data folder was not found.\n"
            f"Expected folder: {RAW_DIR}\n"
            "Check that the original TIFFs are in data/raw/blank/."
        )

    matches = set(RAW_DIR.glob("*Position004-14.tif"))
    matches.update(RAW_DIR.glob("*Position004-14.tiff"))

    matches = sorted(matches)

    if len(matches) == 0:
        available = sorted(
            p.name for p in RAW_DIR.glob("*.tif")
        )

        raise FileNotFoundError(
            "Could not find the original TIFF for Blastoid 14.\n"
            f"Search folder: {RAW_DIR}\n"
            "Expected filename ending in Position004-14.tif.\n"
            f"Number of .tif files: {len(available)}"
        )

    if len(matches) > 1:
        names = "\n".join(p.name for p in matches)

        raise RuntimeError(
            "More than one possible Blastoid 14 TIFF was found:\n"
            + names
        )

    return matches[0]


def get_original_b14():
    """Load original ZCYX TIFF and return DAPI maximum projection."""

    file_path = find_original_b14()

    with tifffile.TiffFile(file_path) as tif:

        if len(tif.series) == 0:
            raise ValueError(
                f"No image series found in {file_path.name}"
            )

        series = tif.series[0]

        axes = series.axes
        arr = series.asarray()

    print(f"Loading original file: {file_path.name}")
    print(f"TIFF axes: {axes}")
    print(f"Original TIFF shape: {arr.shape}")
    print(f"Original dtype: {arr.dtype}")

    if axes != "ZCYX":
        raise ValueError(
            f"Expected TIFF axes 'ZCYX', but found '{axes}'."
        )

    if arr.ndim != 4 or arr.shape[1] != 4:
        raise ValueError(
            f"Expected (Z, 4, Y, X), but received {arr.shape}."
        )

    # Channel 0 = DAPI
    dapi_stack = arr[:, 0, :, :]

    # Maximum intensity projection across Z
    dapi_max = np.max(
        dapi_stack,
        axis=0
    ).astype(np.float32)

    if not np.isfinite(dapi_max).all():
        raise ValueError(
            "DAPI projection contains NaN or infinite values."
        )

    if dapi_max.max() <= dapi_max.min():
        raise ValueError(
            "DAPI projection has no intensity variation."
        )

    print(f"DAPI stack shape: {dapi_stack.shape}")
    print(f"DAPI MAX projection shape: {dapi_max.shape}")

    print(
        f"DAPI intensity range: "
        f"{dapi_max.min():.1f} to {dapi_max.max():.1f}"
    )

    return file_path, dapi_max


# ============================================================
# 4. SHARED IMAGE PREPARATION
# ============================================================

def get_border_pixels(image):
    """Extract border pixels to estimate background noise."""

    height, width = image.shape

    band = max(
        4,
        min(height, width) // 12
    )

    border_mask = np.ones(
        image.shape,
        dtype=bool
    )

    border_mask[band:-band, band:-band] = False

    return image[border_mask]


def prepare_dapi(dapi):
    """Light smoothing and background estimation."""

    smooth = ndi.gaussian_filter(
        dapi,
        sigma=SMOOTH_SIGMA
    )

    border = get_border_pixels(smooth)

    border_median = float(
        np.median(border)
    )

    border_mad = float(
        1.4826 * np.median(
            np.abs(border - border_median)
        )
    )

    border_noise = max(
        border_mad,
        1.0
    )

    background = ndi.gaussian_filter(
        smooth,
        sigma=BACKGROUND_SIGMA
    )

    residual = smooth - background

    return (
        smooth,
        residual,
        border_median,
        border_noise
    )


def remove_small_binary_components(mask, min_area):
    """Remove connected components smaller than min_area."""

    labels, number = ndi.label(mask)

    if number == 0:
        return np.zeros(
            mask.shape,
            dtype=bool
        )

    sizes = np.bincount(
        labels.ravel()
    )

    keep = sizes >= int(min_area)

    keep[0] = False

    return keep[labels]


# ============================================================
# 5. METHOD A: ADAPTIVE THRESHOLD + WATERSHED
# ============================================================

def adaptive_watershed(dapi):

    smooth, residual, border_median, border_noise = (
        prepare_dapi(dapi)
    )

    # Calculate local mean
    local_mean = ndi.uniform_filter(
        smooth,
        size=LOCAL_WINDOW
    )

    # Calculate local standard deviation
    local_sq_mean = ndi.uniform_filter(
        smooth * smooth,
        size=LOCAL_WINDOW
    )

    local_variance = np.maximum(
        local_sq_mean - local_mean * local_mean,
        0.0
    )

    local_std = np.sqrt(
        local_variance
    )

    # Adaptive threshold
    local_threshold = (
        local_mean
        + LOCAL_STD_FACTOR * local_std
    )

    # Background noise floor
    intensity_floor = (
        border_median
        + 2.0 * border_noise
    )

    # Candidate foreground
    foreground = (
        (smooth > local_threshold)
        & (smooth > intensity_floor)
    )

    foreground = remove_small_binary_components(
        foreground,
        MIN_OBJECT_AREA
    )

    if not np.any(foreground):

        empty_labels = np.zeros(
            dapi.shape,
            dtype=np.uint16
        )

        return (
            [],
            empty_labels,
            foreground,
            intensity_floor,
            border_noise
        )

    # Distance transform
    distance = ndi.distance_transform_edt(
        foreground
    )

    # Find local maxima as watershed seeds
    seed_coordinates = peak_local_max(
        distance,
        min_distance=SEED_SPACING,
        threshold_abs=SEED_MIN_DISTANCE_TRANSFORM,
        labels=foreground.astype(np.uint8),
        exclude_border=False
    )

    seed_image = np.zeros(
        foreground.shape,
        dtype=np.int32
    )

    for seed_id, (y, x) in enumerate(
        seed_coordinates,
        start=1
    ):
        seed_image[y, x] = seed_id

    if len(seed_coordinates) > 0:

        labels = watershed(
            -distance,
            seed_image,
            mask=foreground
        )

    else:

        labels, _ = ndi.label(
            foreground
        )

    # Filter objects by area
    clean_labels = np.zeros(
        labels.shape,
        dtype=np.uint16
    )

    candidate_centres = []

    next_label = 1

    for region in regionprops(labels):

        if (
            MIN_OBJECT_AREA
            <= region.area
            <= MAX_OBJECT_AREA
        ):

            clean_labels[
                labels == region.label
            ] = next_label

            y, x = region.centroid

            candidate_centres.append(
                (float(y), float(x))
            )

            next_label += 1

    return (
        candidate_centres,
        clean_labels,
        foreground,
        intensity_floor,
        border_noise
    )


# ============================================================
# 6. METHOD B: MULTISCALE LoG
# ============================================================

def multiscale_log(dapi):

    smooth, residual, border_median, border_noise = (
        prepare_dapi(dapi)
    )

    positive_residual = np.maximum(
        residual,
        0.0
    )

    scale = max(
        float(
            np.percentile(
                positive_residual,
                99.5
            )
        ),
        1.0
    )

    feature_image = np.clip(
        positive_residual / scale,
        0.0,
        1.0
    )

    # Detect blob-like structures at multiple scales
    blobs = blob_log(
        feature_image,
        min_sigma=LOG_MIN_SIGMA,
        max_sigma=LOG_MAX_SIGMA,
        num_sigma=LOG_NUM_SIGMA,
        threshold=LOG_THRESHOLD,
        overlap=LOG_OVERLAP,
        exclude_border=False
    )

    points = []

    height, width = dapi.shape

    for y, x, sigma in blobs:

        yi = int(
            np.clip(
                round(y),
                0,
                height - 1
            )
        )

        xi = int(
            np.clip(
                round(x),
                0,
                width - 1
            )
        )

        # Reject candidates without sufficient signal support
        residual_floor = max(
            2.0,
            1.5 * border_noise
        )

        raw_floor = (
            border_median
            + 2.0 * border_noise
        )

        if residual[yi, xi] < residual_floor:
            continue

        if smooth[yi, xi] < raw_floor:
            continue

        points.append(
            (
                float(y),
                float(x),
                float(sigma)
            )
        )

    return points, border_noise


# ============================================================
# 7. VISUALISATION
# ============================================================

def make_display_image(dapi):
    """Display-only contrast stretching."""

    low, high = np.percentile(
        dapi,
        [0.3, 99.5]
    )

    if high <= low:
        high = low + 1.0

    display = np.clip(
        (dapi - low) / (high - low) * 255.0,
        0,
        255
    )

    return Image.fromarray(
        display.astype(np.uint8)
    ).convert("RGB")


def make_candidate_panel(
    dapi,
    points,
    title,
    colour
):

    panel = make_display_image(dapi)

    draw = ImageDraw.Draw(panel)

    # Circles indicate candidate CENTRES only.
    # They do not represent nucleus boundaries.

    for point in points:

        y, x = point[0], point[1]

        radius = 3

        draw.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ),
            outline=colour,
            width=1
        )

    return title, panel


def compose_comparison(
    panels,
    output_path
):

    panel_width, panel_height = (
        panels[0][1].size
    )

    title_height = 30

    canvas = Image.new(
        "RGB",
        (
            panel_width * 2,
            (panel_height + title_height) * 2
        ),
        "#161616"
    )

    draw = ImageDraw.Draw(canvas)

    try:

        font = ImageFont.truetype(
            "DejaVuSans.ttf",
            13
        )

    except OSError:

        font = ImageFont.load_default()

    for index, (title, panel) in enumerate(panels):

        x = (
            index % 2
        ) * panel_width

        y = (
            index // 2
        ) * (
            panel_height + title_height
        )

        draw.text(
            (x + 5, y + 7),
            title,
            font=font,
            fill="white"
        )

        canvas.paste(
            panel,
            (x, y + title_height)
        )

    # Upscale for easier inspection
    canvas = canvas.resize(
        (
            canvas.width * 2,
            canvas.height * 2
        ),
        Image.Resampling.NEAREST
    )

    canvas.save(output_path)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

def save_outputs(dapi, file_path):

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Method A
    (
        ws_points,
        ws_labels,
        foreground,
        intensity_floor,
        ws_noise
    ) = adaptive_watershed(dapi)

    # Method B
    log_points, log_noise = multiscale_log(dapi)

    # Foreground mask
    foreground_image = Image.fromarray(
        foreground.astype(np.uint8) * 255
    ).convert("L")

    foreground_panel = foreground_image.convert(
        "RGB"
    )

    # Four comparison panels
    panels = [

        (
            "Original DAPI MAX (display stretch only)",
            make_display_image(dapi)
        ),

        make_candidate_panel(
            dapi,
            ws_points,
            f"A: adaptive threshold + watershed | candidates={len(ws_points)}",
            (0, 255, 255)
        ),

        make_candidate_panel(
            dapi,
            log_points,
            f"B: multiscale LoG centres | candidates={len(log_points)}",
            (255, 140, 0)
        ),

        (
            "A: foreground mask (white = eligible pixels)",
            foreground_panel
        )
    ]

    # Save comparison image
    comparison_path = (
        OUT_DIR / "B14_method_comparison.png"
    )

    compose_comparison(
        panels,
        comparison_path
    )

    # Save watershed label image
    labels_path = (
        OUT_DIR / "B14_DAPI_labels_method_A.tif"
    )

    tifffile.imwrite(
        labels_path,
        ws_labels.astype(np.uint16)
    )

    # Save foreground mask
    mask_path = (
        OUT_DIR / "B14_foreground_mask_method_A.tif"
    )

    tifffile.imwrite(
        mask_path,
        foreground.astype(np.uint8) * 255
    )

    # Save candidate coordinates
    candidates_path = (
        OUT_DIR / "B14_candidates.csv"
    )

    with candidates_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "method",
                "id",
                "x",
                "y",
                "sigma_if_log"
            ]
        )

        for method_name, points in [

            ("adaptive_watershed", ws_points),

            ("multiscale_log", log_points)

        ]:

            for candidate_id, point in enumerate(
                points,
                start=1
            ):

                sigma = (
                    round(point[2], 3)
                    if len(point) > 2
                    else ""
                )

                writer.writerow(
                    [
                        method_name,
                        candidate_id,
                        round(point[1], 2),
                        round(point[0], 2),
                        sigma
                    ]
                )

    # Save summary
    summary_path = (
        OUT_DIR / "B14_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "blastoid",
                "source_file",
                "manual_dapi_count",
                "method",
                "candidate_count",
                "signed_error_auto_minus_manual",
                "absolute_error"
            ]
        )

        for method_name, count in [

            ("adaptive_watershed", len(ws_points)),

            ("multiscale_log", len(log_points))

        ]:

            difference = (
                count - MANUAL_DAPI_B14
            )

            writer.writerow(
                [
                    14,
                    file_path.name,
                    MANUAL_DAPI_B14,
                    method_name,
                    count,
                    difference,
                    abs(difference)
                ]
            )

    # Print results
    print(
        "\n========== BLASTOID 14 DIAGNOSTIC RESULTS =========="
    )

    print(
        f"Input file: {file_path.name}"
    )

    print(
        f"Manual DAPI count: {MANUAL_DAPI_B14}"
    )

    print(
        f"Method A - adaptive watershed: {len(ws_points)}"
    )

    print(
        f"Method B - multiscale LoG: {len(log_points)}"
    )

    print(
        f"Border-noise estimate (A): {ws_noise:.3f}"
    )

    print(
        f"Border-noise estimate (B): {log_noise:.3f}"
    )

    print(
        f"Method A intensity floor: {intensity_floor:.3f}"
    )

    print("\nSaved files:")

    print(
        f"Comparison image: {comparison_path}"
    )

    print(
        f"Candidate coordinates: {candidates_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Method A label image: {labels_path}"
    )

    print(
        f"Method A foreground mask: {mask_path}"
    )

    print(
        "\nIMPORTANT: These are unvalidated candidates, "
        "not confirmed biological nucleus counts."
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print(f"Project root: {ROOT}")

    print(f"Raw input folder: {RAW_DIR}")

    file_path, dapi = get_original_b14()

    save_outputs(
        dapi,
        file_path
    )


if __name__ == "__main__":
    main()