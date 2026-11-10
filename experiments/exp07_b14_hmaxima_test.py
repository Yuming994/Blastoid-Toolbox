"""
Experiment 07: Iterative H-maxima marker-controlled watershed for B14.

Purpose:
1. Reuse the EXACT foreground mask from Exp06 Method A.
2. Replace peak_local_max seed generation with H-maxima.
3. Test several H values to reduce over-segmentation.
4. Export overlays, markers, labels and CSV summaries.

Run from the Blastoid-Toolbox project root:
    python -m experiments.exp07_b14_hmaxima_test

This experiment does not modify Exp06.
It does not use Jessie counts to force the output.
"""

from pathlib import Path
import csv

import numpy as np
import tifffile

from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

# Reuse the original, tested Exp06 implementation.
from experiments.exp06_b14_method_comparison import (
    get_original_b14,
    adaptive_watershed,
    make_display_image,
    MIN_OBJECT_AREA,
    MAX_OBJECT_AREA,
)


# ---------------------------------------------------------
# 1. Paths and parameters
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    ROOT
    / "results"
    / "counting"
    / "b14_hmaxima_test"
)

# H is measured in pixels of the distance transform.
# We test a range rather than selecting one value in advance.
H_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]

DISPLAY_SCALE = 2


# ---------------------------------------------------------
# 2. Extract candidate regions from labels
# ---------------------------------------------------------

def extract_regions(labels):

    records = []

    for region in regionprops(labels.astype(np.int32)):

        y, x = region.centroid

        records.append({
            "candidate_id": len(records) + 1,
            "x": float(x),
            "y": float(y),
            "area_px": int(region.area),
        })

    return records


# ---------------------------------------------------------
# 3. H-maxima marker generation + watershed
# ---------------------------------------------------------

def run_hmaxima_watershed(distance, foreground, h):

    # Step 1:
    # Find regional maxima whose prominence is at least h.
    maxima_mask = h_maxima(
        distance,
        h=float(h),
    )

    # Only allow markers inside the original Method A mask.
    maxima_mask = maxima_mask & foreground

    # Step 2:
    # Each connected maxima region becomes one watershed marker.
    markers, seed_count = ndi.label(maxima_mask)

    # Step 3:
    # Run watershed using exactly the same foreground mask.
    if seed_count > 0:

        raw_labels = watershed(
            -distance,
            markers,
            mask=foreground,
        )

    else:

        raw_labels = np.zeros(
            foreground.shape,
            dtype=np.int32,
        )

    # Step 4:
    # Apply the SAME object-area filter used in Exp06.
    clean_labels = np.zeros(
        foreground.shape,
        dtype=np.uint16,
    )

    records = []
    next_id = 1

    for region in regionprops(raw_labels):

        if not (
            MIN_OBJECT_AREA
            <= region.area
            <= MAX_OBJECT_AREA
        ):
            continue

        clean_labels[
            raw_labels == region.label
        ] = next_id

        y, x = region.centroid

        records.append({
            "candidate_id": next_id,
            "x": float(x),
            "y": float(y),
            "area_px": int(region.area),
        })

        next_id += 1

    return (
        maxima_mask,
        markers,
        seed_count,
        clean_labels,
        records,
    )


# ---------------------------------------------------------
# 4. Visualisation: mark CENTRES only
# ---------------------------------------------------------

def make_overlay(dapi, records):

    base = make_display_image(dapi)

    base = base.resize(
        (
            base.width * DISPLAY_SCALE,
            base.height * DISPLAY_SCALE,
        ),
        Image.Resampling.NEAREST,
    )

    draw = ImageDraw.Draw(base)

    for record in records:

        x = int(round(record["x"] * DISPLAY_SCALE))
        y = int(round(record["y"] * DISPLAY_SCALE))

        radius = 3

        # Small circle and cross = candidate CENTRE.
        # This does not represent a nucleus boundary.
        draw.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            ),
            outline=(0, 255, 255),
            width=2,
        )

        draw.line(
            (x - radius - 2, y, x + radius + 2, y),
            fill=(0, 255, 255),
            width=1,
        )

        draw.line(
            (x, y - radius - 2, x, y + radius + 2),
            fill=(0, 255, 255),
            width=1,
        )

    return base


def upscale_original(dapi):

    base = make_display_image(dapi)

    return base.resize(
        (
            base.width * DISPLAY_SCALE,
            base.height * DISPLAY_SCALE,
        ),
        Image.Resampling.NEAREST,
    )


# ---------------------------------------------------------
# 5. Compose comparison image
# ---------------------------------------------------------

def save_comparison_grid(panels, output_path):

    panel_width, panel_height = panels[0][1].size

    columns = 2
    rows = (len(panels) + columns - 1) // columns

    title_height = 32

    canvas = Image.new(
        "RGB",
        (
            columns * panel_width,
            rows * (panel_height + title_height),
        ),
        "#161616",
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf",
            13,
        )
    except OSError:
        font = ImageFont.load_default()

    for i, (title, panel) in enumerate(panels):

        x = (i % columns) * panel_width

        y = (
            i // columns
        ) * (panel_height + title_height)

        draw.text(
            (x + 5, y + 7),
            title,
            font=font,
            fill="white",
        )

        canvas.paste(
            panel,
            (x, y + title_height),
        )

    canvas.save(output_path)


# ---------------------------------------------------------
# 6. Main experiment
# ---------------------------------------------------------

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("EXPERIMENT 07: B14 H-MAXIMA WATERSHED")
    print("=" * 60)

    # Load the original ZCYX TIFF.
    file_path, dapi = get_original_b14()

    print(f"\nInput: {file_path.name}")
    print(f"DAPI shape: {dapi.shape}")

    # -----------------------------------------------------
    # IMPORTANT:
    # Reuse the exact Exp06 Method A foreground mask.
    # Do not modify its thresholding or preprocessing.
    # -----------------------------------------------------

    (
        baseline_points,
        baseline_labels,
        foreground,
        intensity_floor,
        border_noise,
    ) = adaptive_watershed(dapi)

    baseline_records = extract_regions(
        baseline_labels
    )

    print("\nOriginal Method A:")
    print(f"  Candidates: {len(baseline_records)}")
    print(f"  Foreground pixels: {int(foreground.sum())}")
    print(f"  Intensity floor: {intensity_floor:.3f}")

    # Check whether the mask matches the previously saved Exp06 mask.
    old_mask_path = (
        ROOT
        / "results"
        / "counting"
        / "b14_diagnostics"
        / "B14_foreground_mask_method_A.tif"
    )

    if old_mask_path.exists():

        old_mask = tifffile.imread(
            old_mask_path
        ) > 0

        if old_mask.shape == foreground.shape:

            mask_matches = np.array_equal(
                old_mask,
                foreground,
            )

            print(
                f"  Matches saved Exp06 mask: {mask_matches}"
            )

        else:

            print(
                "  Saved Exp06 mask has a different shape."
            )

    # Save the unchanged foreground mask.
    tifffile.imwrite(
        OUT_DIR / "B14_foreground_mask_exp07.tif",
        foreground.astype(np.uint8) * 255,
    )

    # Save original Method A labels for comparison.
    tifffile.imwrite(
        OUT_DIR / "B14_baseline_labels.tif",
        baseline_labels.astype(np.uint16),
    )

    # -----------------------------------------------------
    # Distance transform: identical input for every H value.
    # -----------------------------------------------------

    distance = ndi.distance_transform_edt(
        foreground
    )

    # -----------------------------------------------------
    # Output containers
    # -----------------------------------------------------

    panels = []

    panels.append((
        "Original DAPI MAX",
        upscale_original(dapi),
    ))

    panels.append((
        f"Original Method A | candidates={len(baseline_records)}",
        make_overlay(dapi, baseline_records),
    ))

    summary_rows = []

    summary_rows.append({
        "method": "Method_A_baseline",
        "h_value_px": "",
        "seed_count": "",
        "candidate_count": len(baseline_records),
        "foreground_pixels": int(foreground.sum()),
    })

    candidate_rows = []

    for record in baseline_records:

        candidate_rows.append({
            "method": "Method_A_baseline",
            "h_value_px": "",
            **record,
        })

    # -----------------------------------------------------
    # Test H-maxima at different prominence values.
    # -----------------------------------------------------

    for h in H_VALUES:

        print(f"\nTesting H = {h:.1f} pixels...")

        (
            maxima_mask,
            markers,
            seed_count,
            labels,
            records,
        ) = run_hmaxima_watershed(
            distance,
            foreground,
            h,
        )

        candidate_count = len(records)

        print(f"  H-maxima seeds: {seed_count}")
        print(f"  Final candidates: {candidate_count}")

        # Use a filesystem-safe H value.
        h_name = str(h).replace(".", "p")

        # Save marker map.
        tifffile.imwrite(
            OUT_DIR / f"B14_markers_h{h_name}.tif",
            markers.astype(np.uint16),
        )

        # Save final watershed labels.
        tifffile.imwrite(
            OUT_DIR / f"B14_labels_h{h_name}.tif",
            labels.astype(np.uint16),
        )

        # Save individual centre overlay.
        overlay = make_overlay(
            dapi,
            records,
        )

        overlay.save(
            OUT_DIR / f"B14_overlay_h{h_name}.png"
        )

        # Add to comparison grid.
        panels.append((
            (
                f"H={h:.1f} px | "
                f"seeds={seed_count} | "
                f"candidates={candidate_count}"
            ),
            overlay,
        ))

        # Save summary row.
        summary_rows.append({
            "method": "H_maxima_watershed",
            "h_value_px": h,
            "seed_count": seed_count,
            "candidate_count": candidate_count,
            "foreground_pixels": int(foreground.sum()),
        })

        # Save individual candidate coordinates.
        for record in records:

            candidate_rows.append({
                "method": "H_maxima_watershed",
                "h_value_px": h,
                **record,
            })

    # -----------------------------------------------------
    # Save comparison image and CSV reports.
    # -----------------------------------------------------

    comparison_path = (
        OUT_DIR / "B14_hmaxima_comparison.png"
    )

    save_comparison_grid(
        panels,
        comparison_path,
    )

    summary_path = (
        OUT_DIR / "B14_hmaxima_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "h_value_px",
                "seed_count",
                "candidate_count",
                "foreground_pixels",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    candidates_path = (
        OUT_DIR / "B14_hmaxima_candidates.csv"
    )

    with candidates_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "h_value_px",
                "candidate_id",
                "x",
                "y",
                "area_px",
            ],
        )

        writer.writeheader()
        writer.writerows(candidate_rows)

    # -----------------------------------------------------
    # Final report.
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT 07 COMPLETED")
    print("=" * 60)

    print(f"\nOriginal Method A: {len(baseline_records)}")

    for row in summary_rows[1:]:

        print(
            f"H={row['h_value_px']}: "
            f"seeds={row['seed_count']}, "
            f"candidates={row['candidate_count']}"
        )

    print("\nSaved outputs:")
    print(f"  Comparison: {comparison_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Coordinates: {candidates_path}")
    print(f"  Output folder: {OUT_DIR}")

    print("\nForeground mask was reused from Exp06.")
    print("Only marker generation was changed.")
    print("Candidate counts are NOT validated biological counts.")


if __name__ == "__main__":
    main()