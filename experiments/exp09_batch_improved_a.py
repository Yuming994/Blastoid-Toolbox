"""Batch test of EXACT Exp08 Improved Method A on 20 Blank TIFFs x 4 channels.

Run from the Blastoid-Toolbox root:
    python -m experiments.exp09_batch_improved_a

Inputs:
    data/raw/blank/*.tif[f]  (original microscope ZCYX hyperstacks)
    data/ground_truth/manual_counts.csv (optional, for comparison only)

Outputs (results/counting/improved_a_batch_v1/):
    summary_counts.csv       # one row per image/channel, normally 80 rows
    summary_wide.csv         # one row per blastoid, normally 20 rows
    metrics_by_marker.csv    # aggregate candidate-count errors, not segmentation F1
    all_centres.csv          # positions of every refined candidate
    all_pair_decisions.csv   # evidence for each considered merge
    overlays/B01_DAPI.png etc. (unless --skip-overlays)

Important interpretation:
  * DAPI output: putative nucleus centres, NOT confirmed true nuclei.
  * SOX2/GATA3/SOX17: independent intensity-blob candidates, NOT validated
    lineage-positive cell counts. No DAPI colocalization or positivity test.
  * The model and fixed parameters are identical across channels, uncalibrated.
  * Manual counts are used only to calculate differences AFTER detection.
  * Image display uses contrast stretching; detector uses raw MAX intensities.

The baseline and refinement functions are imported UNMODIFIED from Exp08.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from skimage.measure import regionprops

from experiments.exp08_b14_improved_a import (
    method_a,
    refine,
    to_rgb,
    draw_points,
)

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ("DAPI", "SOX2", "GATA3", "SOX17")
SAMPLE_RE = re.compile(r"Position\d+-(\d+)\.tiff?$", re.IGNORECASE)

COUNT_FIELDS = (
    "sample_id", "source_file", "marker", "status", "baseline_candidates",
    "improved_candidates", "merged_away", "review_weak_signal", "median_baseline_area_px2",
    "manual_count", "auto_minus_manual", "absolute_count_error", "error_message",
)
CENTRE_FIELDS = (
    "sample_id", "marker", "source_file", "final_id", "x", "y",
    "area_px", "n_original_candidates", "original_ids", "local_contrast",
    "review_weak_signal",
)
PAIR_FIELDS = (
    "sample_id", "marker", "source_file", "candidate_a", "candidate_b",
    "decision", "shared_edge_px", "distance_px", "valley_ratio", "union_area_px",
    "union_solidity", "union_eccentricity", "failed_tests",
)


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_number(value):
    value = str(value).strip()
    if not value:
        return None
    try:
        number = float(value)
        if not np.isfinite(number) or not number.is_integer():
            return None
        return int(number)
    except ValueError:
        return None


def read_manual_counts(path: Path):
    """Supports the user's 2-row-header Excel-exported CSV (and 1-row CSVs)."""
    if not path.exists():
        print(f"[INFO] Manual file absent: {path}. Counts still run; error columns left blank.")
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    header_row = None
    for index, row in enumerate(rows):
        cleaned = [cell.strip().lower() for cell in row]
        if "blastoid id" in cleaned and "dapi" in cleaned:
            header_row = index
            break
    if header_row is None:
        raise ValueError(
            f"Could not find 'Blastoid ID' and 'DAPI' header in {path}. "
            "Expecting Jessie's manual_counts.csv (with one or two header rows)."
        )

    names = [name.strip() for name in rows[header_row]]
    name_to_col = {name.strip().lower(): idx for idx, name in enumerate(names)}
    requested_cols = {"DAPI": "dapi", "SOX2": "sox2+", "GATA3": "gata3+", "SOX17": "sox17+"}
    for marker, col in requested_cols.items():
        if col not in name_to_col:
            raise ValueError(f"Missing manual column {col} for {marker} in {path}. Actual columns: {names}")

    mapping = {}
    id_col = name_to_col["blastoid id"]
    for line_num, row in enumerate(rows[header_row + 1:], header_row + 2):
        sid = parse_number(row[id_col] if len(row) > id_col else "")
        if sid is None:
            continue
        if sid in mapping:
            raise ValueError(f"Duplicate manual Blastoid ID {sid} at CSV line {line_num}")
        mapping[sid] = {}
        for marker, col in requested_cols.items():
            j = name_to_col[col]
            mapping[sid][marker] = parse_number(row[j] if len(row) > j else "")
    print(f"[INFO] Loaded manual totals for {len(mapping)} blastoids from {path.name}")
    return mapping


def locate_raw_tiffs(raw: Path):
    if not raw.is_dir():
        raise FileNotFoundError(f"Original TIFF folder not found: {raw}")
    paths = sorted(
        path for path in raw.iterdir()
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff")
    )
    if not paths:
        raise FileNotFoundError(f"No TIFF files in: {raw}")

    collected = {}
    for path in paths:
        match = SAMPLE_RE.search(path.name)
        if not match:
            raise ValueError(
                f"Unexpected TIFF name: {path.name}. "
                "Use only original raw hyperstacks named *Position004-14.tif etc. "
                "Do not put *_max.tif files in data/raw/blank."
            )
        sid = int(match.group(1))
        if sid in collected:
            raise ValueError(f"Duplicate Blastoid ID {sid}: {collected[sid].name} and {path.name}")
        collected[sid] = path
    return sorted(collected.items())


def panel(dapi, initial_labels, clusters, weak_ids, marker, sample_id, output: Path):
    """Four panels using Exp08's SAME display styling; no processing changes."""
    original_pts = [(p.centroid[1], p.centroid[0]) for p in regionprops(initial_labels)]
    refined_pts = [(r["x"], r["y"]) for r in clusters]
    merged_pts = [(r["x"], r["y"]) for r in clusters if r["n_original_candidates"] > 1]
    flagged_pts = [(r["x"], r["y"]) for r in clusters if r["final_id"] in weak_ids]

    imgs = [to_rgb(dapi) for _ in range(4)]
    draw_points(imgs[1], original_pts, (0, 220, 255))
    draw_points(imgs[2], refined_pts, (40, 255, 110))
    draw_points(imgs[2], flagged_pts, (255, 185, 20), size=8)
    draw_points(imgs[3], merged_pts, (255, 50, 255), size=8)
    draw_points(imgs[3], flagged_pts, (255, 185, 20), size=8)
    w, h = imgs[0].size
    band = 35
    canvas = Image.new("RGB", (2*w, 2*(h+band)), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    texts = [
        f"B{sample_id:02d} {marker}: raw MAX (display stretch only)",
        f"Baseline A: {len(original_pts)} candidates",
        f"Improved A: {len(clusters)} candidates",
        f"Merged groups: {len(merged_pts)} | weak review: {len(flagged_pts)}",
    ]
    for i, img in enumerate(imgs):
        left = (i % 2)*w
        top = (i // 2)*(h + band)
        draw.text((left+8, top+8), texts[i], fill="white")
        canvas.paste(img, (left, top+band))
    canvas.save(output)


def compute_metrics(count_rows):
    rows = []
    for marker in MARKERS:
        selected = [r for r in count_rows if r["marker"] == marker and r["status"] == "OK"]
        paired = [r for r in selected if r["manual_count"] is not None]
        errors = [r["auto_minus_manual"] for r in paired]
        rows.append({
            "marker": marker,
            "images_ok": len(selected),
            "images_with_manual": len(paired),
            "sum_manual": sum(r["manual_count"] for r in paired),
            "sum_auto_on_paired": sum(r["improved_candidates"] for r in paired),
            "mean_abs_count_error": round(statistics.mean(abs(e) for e in errors), 3) if errors else "",
            "mean_signed_count_error": round(statistics.mean(errors), 3) if errors else "",
            "manual_zero_auto_nonzero_images": sum(
                r["manual_count"] == 0 and r["improved_candidates"] > 0
                for r in paired
            ),
        })
    return rows


def compute_wide(count_rows):
    wide = {}
    for row in count_rows:
        sid = row["sample_id"]
        sample = wide.setdefault(sid, {"sample_id": sid})
        marker = row["marker"]
        sample[f"{marker}_baseline"] = row["baseline_candidates"]
        sample[f"{marker}_improved"] = row["improved_candidates"]
        sample[f"{marker}_manual"] = row["manual_count"]
        sample[f"{marker}_error"] = row["auto_minus_manual"]
        sample[f"{marker}_status"] = row["status"]
    fields = ["sample_id"]
    for m in MARKERS:
        fields += [f"{m}_{suffix}" for suffix in ("baseline", "improved", "manual", "error", "status")]
    return fields, [wide[sid] for sid in sorted(wide)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/blank")
    parser.add_argument("--manual", type=Path, default=ROOT / "data/ground_truth/manual_counts.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/counting/improved_a_batch_v1")
    parser.add_argument("--skip-overlays", action="store_true", help="Optional: do not write overlay PNG files")
    args = parser.parse_args()

    files = locate_raw_tiffs(args.raw)
    manual = read_manual_counts(args.manual)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    overlay_path = output / "overlays"
    if not args.skip_overlays:
        overlay_path.mkdir(exist_ok=True)

    print(f"[INFO] Found {len(files)} original TIFFs: {len(files)*4} channel images expected")
    if len(files) != 20:
        print("[WARNING] Expected 20 blastoids in Blank condition; check raw folder.")
    count_rows, centre_rows, pair_rows = [], [], []

    for ix, (sample_id, path) in enumerate(files, 1):
        print(f"[{ix}/{len(files)}] B{sample_id:02d} {path.name}", flush=True)
        try:
            with tifffile.TiffFile(path) as tif:
                series = tif.series[0]
                axes = series.axes
                raw = series.asarray()
            if axes != "ZCYX" or raw.ndim != 4 or raw.shape[1] != 4:
                raise ValueError(f"Expected ZCYX with 4 channels, got {axes}, {raw.shape}")

            for ci, marker in enumerate(MARKERS):
                result = {"sample_id": sample_id, "source_file": path.name,
                          "marker": marker, "status": "OK", "error_message": ""}
                try:
                    # EXACT same projection and Exp08 pipeline, independently per channel.
                    max_img = raw[:, ci, :, :].max(axis=0).astype(np.float32)
                    initial, foreground, bg, noise = method_a(max_img)
                    refined, decisions, clusters, review, med_area = refine(initial, max_img)
                    baseline_n = len(regionprops(initial))
                    improved_n = len(clusters)
                    reference = manual.get(sample_id, {}).get(marker)
                    error = (improved_n - reference) if reference is not None else None

                    result.update({
                        "baseline_candidates": baseline_n, "improved_candidates": improved_n,
                        "merged_away": baseline_n - improved_n, "review_weak_signal": len(review),
                        "median_baseline_area_px2": round(float(med_area), 3),
                        "manual_count": reference, "auto_minus_manual": error,
                        "absolute_count_error": abs(error) if error is not None else None,
                    })
                    for rec in clusters:
                        centre_rows.append({"sample_id": sample_id, "source_file": path.name,
                                            "marker": marker, **rec})
                    for rec in decisions:
                        pair_rows.append({"sample_id": sample_id, "source_file": path.name,
                                          "marker": marker, **rec})
                    if not args.skip_overlays:
                        panel(max_img, initial, clusters, review, marker, sample_id,
                              overlay_path / f"B{sample_id:02d}_{marker}.png")
                    print(f"   {marker:5s} | A={baseline_n:3d} -> Improved={improved_n:3d} "
                          f"| Manual={reference if reference is not None else 'NA'} "
                          f"| review={len(review)}", flush=True)
                except Exception as exc:
                    result.update({"status": "ERROR", "error_message": f"{type(exc).__name__}: {exc}"})
                    print(f"   [ERROR] {marker}: {result['error_message']}", file=sys.stderr, flush=True)
                count_rows.append(result)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"   [ERROR] Failed to load original TIFF: {msg}", file=sys.stderr, flush=True)
            for marker in MARKERS:
                count_rows.append({"sample_id": sample_id, "source_file": path.name,
                                   "marker": marker, "status": "ERROR", "error_message": msg})

    write_csv(output / "summary_counts.csv", COUNT_FIELDS, count_rows)
    wide_fields, wide_rows = compute_wide(count_rows)
    write_csv(output / "summary_wide.csv", wide_fields, wide_rows)
    write_csv(output / "all_centres.csv", CENTRE_FIELDS, centre_rows)
    write_csv(output / "all_pair_decisions.csv", PAIR_FIELDS, pair_rows)
    metric_fields = ("marker", "images_ok", "images_with_manual", "sum_manual",
                     "sum_auto_on_paired", "mean_abs_count_error", "mean_signed_count_error",
                     "manual_zero_auto_nonzero_images")
    metrics = compute_metrics(count_rows)
    write_csv(output / "metrics_by_marker.csv", metric_fields, metrics)

    print("\n=== FINAL STATUS ===")
    statuses = Counter(row["status"] for row in count_rows)
    print(f"Channel rows: {len(count_rows)} | OK: {statuses['OK']} | ERROR: {statuses['ERROR']}")
    for m in metrics:
        print(f"  {m['marker']}: OK={m['images_ok']} / manual_pairs={m['images_with_manual']} "
              f"/ MAE={m['mean_abs_count_error']} / manual_zero_auto_nonzero="
              f"{m['manual_zero_auto_nonzero_images']}")
    print(f"Outputs saved to: {output.resolve()}")
    print("CAUTION: marker-channel candidates are NOT validated marker-positive nuclei.")
    if statuses['ERROR']:
        print("There were errors. Inspect summary_counts.csv status/error_message columns.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
