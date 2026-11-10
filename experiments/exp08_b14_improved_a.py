"""Exp08: conservative refinement of Exp06 Method A, Blank Blastoid 14.

Run inside Blastoid-Toolbox:
    python -m experiments.exp08_b14_improved_a

Input: data/raw/blank/*Position004-14.tif[f] (original 4-channel ZCYX).
Output: results/counting/b14_improved_a/

Method A's initial mask, watershed and parameters are preserved from Exp06.
Only touching candidate regions can be proposed for merging. A merge is
accepted when the image has a shallow intensity valley and the combined
region has plausible geometry. Weak/noisy candidates are FLAGGED, not
silently removed. Not trained on, or tuned to, manual count=106.

This is a diagnostic baseline, NOT a validated nucleus counter.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

# === Paths (independent of VS Code's current working directory) ===
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "blank"
OUTPUT = ROOT / "results" / "counting" / "b14_improved_a"

# === Original Exp06 Method A constants: kept unchanged ===
SMOOTH_SIGMA = 0.9
BACKGROUND_SIGMA = 12.0
LOCAL_WINDOW = 31
LOCAL_STD_FACTOR = 0.15
SEED_SPACING = 5
SEED_MIN_DISTANCE_TRANSFORM = 1.2
MIN_OBJECT_AREA = 12
MAX_OBJECT_AREA = 900

# === Conservative, transparent refinement criteria ===
# These are starting hypotheses, NOT calibrated values.
MAX_PAIR_DISTANCE_DIAMETERS = 1.00
MAX_MERGED_AREA_MEDIANS = 2.10
MIN_MERGED_SOLIDITY = 0.82
MAX_MERGED_ECCENTRICITY = 0.92
MIN_SHALLOW_VALLEY_RATIO = 0.78
MIN_SHARED_EDGE_PIXELS = 2


def load_b14(path: Path | None):
    if path is None:
        matches = sorted(RAW_DIR.glob("*Position004-14.tif"))
        matches += sorted(RAW_DIR.glob("*Position004-14.tiff"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one B14 TIFF at {RAW_DIR}, found {len(matches)}. "
                "Use --input /full/path/to/file.tif if needed."
            )
        path = matches[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes
        stack = series.asarray()
    if axes != "ZCYX" or stack.ndim != 4 or stack.shape[1] != 4:
        raise ValueError(f"Expected ZCYX with 4 channels, got axes={axes}, shape={stack.shape}")
    dapi = stack[:, 0, :, :].max(axis=0).astype(np.float32)
    return path, dapi


def border_pixels(image):
    h, w = image.shape
    band = max(4, min(h, w) // 12)
    outside = np.ones(image.shape, dtype=bool)
    outside[band:-band, band:-band] = False
    return image[outside]


def method_a(dapi):
    """Exact Exp06 Method A logic, copied here to avoid brittle imports."""
    smooth = ndi.gaussian_filter(dapi, sigma=SMOOTH_SIGMA)
    border = border_pixels(smooth)
    med = float(np.median(border))
    noise = max(float(1.4826 * np.median(np.abs(border - med))), 1.0)
    # Kept for parity with the original Exp06 preparation.
    _residual = smooth - ndi.gaussian_filter(smooth, BACKGROUND_SIGMA)

    local_mean = ndi.uniform_filter(smooth, size=LOCAL_WINDOW)
    local_sq_mean = ndi.uniform_filter(smooth * smooth, size=LOCAL_WINDOW)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0))
    floor = med + 2.0 * noise
    mask = (smooth > local_mean + LOCAL_STD_FACTOR * local_std) & (smooth > floor)

    comps, number = ndi.label(mask)
    if number:
        sizes = np.bincount(comps.ravel())
        keep = sizes >= MIN_OBJECT_AREA
        keep[0] = False
        mask = keep[comps]
    else:
        mask = np.zeros(dapi.shape, dtype=bool)

    if not mask.any():
        return np.zeros(dapi.shape, dtype=np.int32), mask, med, noise

    distance = ndi.distance_transform_edt(mask)
    seeds_xy = peak_local_max(
        distance, min_distance=SEED_SPACING,
        threshold_abs=SEED_MIN_DISTANCE_TRANSFORM,
        labels=mask.astype(np.uint8), exclude_border=False,
    )
    seeds = np.zeros(mask.shape, dtype=np.int32)
    for sid, (y, x) in enumerate(seeds_xy, start=1):
        seeds[y, x] = sid
    if len(seeds_xy) > 0:
        raw_labels = watershed(-distance, seeds, mask=mask)
    else:
        raw_labels, _ = ndi.label(mask)

    clean = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for p in regionprops(raw_labels):
        if MIN_OBJECT_AREA <= p.area <= MAX_OBJECT_AREA:
            clean[raw_labels == p.label] = next_id
            next_id += 1
    return clean, mask, med, noise


def adjacent_pairs(labels):
    """Only 4-connected, shared-edge watershed neighbors may be merged."""
    counts = defaultdict(int)
    for a, b in ((labels[:-1, :], labels[1:, :]),
                 (labels[:, :-1], labels[:, 1:])):
        ok = (a > 0) & (b > 0) & (a != b)
        if not np.any(ok):
            continue
        pairs = np.sort(np.column_stack((a[ok], b[ok])), axis=1)
        uniq, freq = np.unique(pairs, axis=0, return_counts=True)
        for pair, n in zip(uniq, freq):
            counts[(int(pair[0]), int(pair[1]))] += int(n)
    return counts


def valley_ratio(image, c1, c2):
    """Smoothed signal along the line: 1=shallow valley, 0=deep valley."""
    y1, x1 = c1
    y2, x2 = c2
    steps = max(11, int(np.hypot(y2 - y1, x2 - x1) * 3))
    ys = np.linspace(y1, y2, steps)
    xs = np.linspace(x1, x2, steps)
    values = ndi.map_coordinates(image, [ys, xs], order=1, mode="nearest")
    k = max(2, int(steps * 0.2))
    peak1 = float(values[:k].max())
    peak2 = float(values[-k:].max())
    valley = float(values[k:-k].min()) if steps > 2 * k else float(values.min())
    return float(np.clip(valley / max(min(peak1, peak2), 1e-6), 0, 1))


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}
        self.members = {i: {i} for i in ids}

    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if len(self.members[a]) < len(self.members[b]):
            a, b = b, a
        self.parent[b] = a
        self.members[a].update(self.members.pop(b))
        return True


def refine(labels, dapi):
    props = {p.label: p for p in regionprops(labels)}
    if not props:
        return labels.copy(), [], [], [], 0

    median_area = float(np.median([p.area for p in props.values()]))
    typical_diameter = float(np.sqrt(4 * median_area / np.pi))
    smooth = ndi.gaussian_filter(dapi, sigma=1.5)
    edges = adjacent_pairs(labels)
    uf = UnionFind(props)
    evaluations = []

    # Process most promising pairs first; union geometry is checked anew.
    ordering = []
    for (a, b), shared in edges.items():
        p, q = props[a], props[b]
        d = float(np.hypot(*(np.asarray(p.centroid) - np.asarray(q.centroid))))
        v = valley_ratio(smooth, p.centroid, q.centroid)
        ordering.append((-v, d, a, b, shared))

    for neg_v, d, a, b, shared in sorted(ordering):
        r1, r2 = uf.find(a), uf.find(b)
        if r1 == r2:
            continue
        members = uf.members[r1] | uf.members[r2]
        merged_mask = np.isin(labels, list(members))
        region = regionprops(merged_mask.astype(np.uint8))[0]
        v = -neg_v
        tests = {
            "touching": shared >= MIN_SHARED_EDGE_PIXELS,
            "distance": d <= MAX_PAIR_DISTANCE_DIAMETERS * typical_diameter,
            "shallow_valley": v >= MIN_SHALLOW_VALLEY_RATIO,
            "size": region.area <= MAX_MERGED_AREA_MEDIANS * median_area,
            "solidity": region.solidity >= MIN_MERGED_SOLIDITY,
            "shape": region.eccentricity <= MAX_MERGED_ECCENTRICITY,
        }
        if all(tests.values()):
            status = "MERGED"
            uf.join(a, b)
        elif tests["touching"] and tests["distance"] and (
                tests["shallow_valley"] or tests["size"]):
            status = "REVIEW_NOT_MERGED"
        else:
            status = "KEEP_SEPARATE"
        evaluations.append({
            "candidate_a": a, "candidate_b": b, "decision": status,
            "shared_edge_px": shared, "distance_px": round(d, 2),
            "valley_ratio": round(v, 3),
            "union_area_px": int(region.area),
            "union_solidity": round(float(region.solidity), 3),
            "union_eccentricity": round(float(region.eccentricity), 3),
            "failed_tests": ";".join(k for k, passed in tests.items() if not passed),
        })

    refined = np.zeros(labels.shape, dtype=np.int32)
    cluster_rows = []
    for new_id, (root, ids) in enumerate(
            sorted(uf.members.items(), key=lambda e: min(e[1])), start=1):
        member_ids = sorted(ids)
        mask = np.isin(labels, member_ids)
        refined[mask] = new_id
        p = regionprops(mask.astype(np.uint8))[0]
        cluster_rows.append({
            "final_id": new_id, "x": round(float(p.centroid[1]), 2),
            "y": round(float(p.centroid[0]), 2),
            "area_px": int(p.area), "n_original_candidates": len(ids),
            "original_ids": ";".join(map(str, member_ids)),
        })

    # Evaluate original signal evidence without deleting possibly genuine dim nuclei.
    border = border_pixels(smooth)
    bg = float(np.median(border))
    noise = max(float(1.4826 * np.median(np.abs(border - bg))), 1.0)
    review_ids = []
    for row in cluster_rows:
        mask = refined == row["final_id"]
        region_value = float(np.percentile(smooth[mask], 75))
        # Annulus excludes the candidate itself. Neighboring nuclei may overlap
        # the ring, so this is only a review indicator, not definitive evidence.
        outer = ndi.binary_dilation(mask, iterations=6)
        inner = ndi.binary_dilation(mask, iterations=2)
        ring = outer & ~inner
        local_bg = float(np.median(smooth[ring])) if ring.any() else bg
        contrast = region_value - local_bg
        weak = (contrast < 1.5 * noise) and (row["area_px"] < median_area)
        row["local_contrast"] = round(contrast, 2)
        row["review_weak_signal"] = int(weak)
        if weak:
            review_ids.append(row["final_id"])

    return refined, evaluations, cluster_rows, review_ids, median_area


def to_rgb(dapi, scale=3):
    lo, hi = np.percentile(dapi, [0.3, 99.5])
    gray = np.uint8(np.clip((dapi - lo) / max(hi - lo, 1e-6) * 255, 0, 255))
    img = Image.fromarray(gray, "L").convert("RGB")
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


def draw_points(img, points, color, size=5):
    draw = ImageDraw.Draw(img)
    for x, y in points:
        x, y = x * 3, y * 3
        draw.ellipse((x-size, y-size, x+size, y+size), outline=color, width=2)


def save_panels(dapi, original, refined, clusters, review_ids, output):
    initial = regionprops(original)
    initial_pts = [(p.centroid[1], p.centroid[0]) for p in initial]
    final_pts = [(r["x"], r["y"]) for r in clusters]
    weak_pts = [(r["x"], r["y"]) for r in clusters if r["final_id"] in review_ids]
    merged = [r for r in clusters if r["n_original_candidates"] > 1]
    merged_pts = [(r["x"], r["y"]) for r in merged]

    a = to_rgb(dapi)
    b = to_rgb(dapi)
    draw_points(b, initial_pts, (0, 220, 255))
    c = to_rgb(dapi)
    draw_points(c, final_pts, (40, 255, 110))
    draw_points(c, weak_pts, (255, 185, 20), size=8)
    d = to_rgb(dapi)
    draw_points(d, merged_pts, (255, 50, 255), size=8)
    draw_points(d, weak_pts, (255, 185, 20), size=8)

    w, h = a.size
    label_h = 36
    composite = Image.new("RGB", (w*2, (h+label_h)*2), (12, 12, 12))
    labels = ["Original DAPI MAX", f"Exp06 Method A | {len(initial)} candidates",
              f"Improved A | {len(clusters)} centres (amber=review)",
              f"Changes | {len(merged)} merged groups, {len(review_ids)} review"]
    draw = ImageDraw.Draw(composite)
    for i, panel in enumerate((a, b, c, d)):
        x = (i % 2)*w
        y = (i // 2)*(h+label_h)
        draw.text((x+8, y+8), labels[i], fill="white")
        composite.paste(panel, (x, y+label_h))
    composite.save(output)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Override B14 TIFF auto-discovery")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    file, dapi = load_b14(args.input)
    initial, fg, bg, noise = method_a(dapi)
    refined, decisions, clusters, review, med_area = refine(initial, dapi)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    save_panels(dapi, initial, refined, clusters, review, out / "B14_improved_comparison.png")
    tifffile.imwrite(out / "B14_initial_labels.tif", initial.astype(np.uint16))
    tifffile.imwrite(out / "B14_refined_labels.tif", refined.astype(np.uint16))
    write_csv(out / "B14_centres.csv", clusters, [
        "final_id", "x", "y", "area_px", "n_original_candidates",
        "original_ids", "local_contrast", "review_weak_signal",
    ])
    write_csv(out / "B14_pair_decisions.csv", decisions, [
        "candidate_a", "candidate_b", "decision", "shared_edge_px", "distance_px",
        "valley_ratio", "union_area_px", "union_solidity", "union_eccentricity",
        "failed_tests",
    ])
    print(f"File: {file.name} | DAPI MAX: {dapi.shape}")
    print(f"Baseline Method A: {int(initial.max())} candidates")
    print(f"Refined: {int(refined.max())} centres | merged away: {int(initial.max()-refined.max())}")
    print(f"Weak-signal review (not removed): {len(review)}")
    print(f"Median baseline area: {med_area:.1f} px^2")
    print(f"Output: {out.resolve()}")
    print("NOTE: no marker positivity, no manual-count optimization; image review required.")


if __name__ == "__main__":
    main()
