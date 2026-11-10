"""DAPI-guided counting pilot for the 20-blastoid blank dataset.

Reads data/exported/blank/*_DAPI_max.tif and matching marker MAX images.
Segments nuclei from DAPI, measures marker signal within each nucleus, then
fits one marker threshold per channel against manual count totals.

This is calibration, not independent validation. Inspect overlays before use.
"""
from pathlib import Path
import csv
import re

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import watershed, find_boundaries

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/projections/blank"
MANUAL_CSV = ROOT / "data/ground_truth/manual_counts.csv"
OUT = ROOT / "results/counting/dapi_guided_v1"
MARKERS = {"SOX2": "SOX2+", "GATA3": "GATA3+", "SOX17": "SOX17+"}

# Initial parameters: review the saved DAPI overlays before trusting counts.
SMOOTH_SIGMA = 1.0
PEAK_DISTANCE = 6
MIN_AREA, MAX_AREA = 15, 900


def read_image(path):
    a = np.squeeze(tifffile.imread(str(path)))
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., :3].astype(np.float32).mean(axis=-1)
    if a.ndim != 2:
        raise ValueError(f"{path.name}: expected 2D MAX image, got {a.shape}")
    return a.astype(np.float32)


def segment_dapi(image):
    smooth = ndi.gaussian_filter(image, SMOOTH_SIGMA)
    if np.ptp(smooth) == 0:
        return np.zeros(image.shape, np.int32)
    mask = smooth > threshold_otsu(smooth)
    mask = remove_small_objects(mask, min_size=15)
    mask = remove_small_holes(mask, area_threshold=20)
    if not mask.any():
        return np.zeros(image.shape, np.int32)

    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(
        distance, min_distance=PEAK_DISTANCE,
        labels=mask.astype(np.uint8), exclude_border=False
    )
    seeds = np.zeros(mask.shape, np.int32)
    if len(coords):
        seeds[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1)
        labels = watershed(-distance, seeds, mask=mask)
    else:
        labels, _ = ndi.label(mask)

    clean = np.zeros(mask.shape, np.int32)
    new_id = 1
    for p in regionprops(labels):
        if MIN_AREA <= p.area <= MAX_AREA:
            clean[labels == p.label] = new_id
            new_id += 1
    return clean




def read_manual(path):
    # 读取人工计数 CSV
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    # 找到真正的表头行
    header_index = next(
        (
            i for i, row in enumerate(rows)
            if any(cell.strip() == "Blastoid ID" for cell in row)
        ),
        None,
    )

    if header_index is None:
        raise ValueError(
            "Could not find 'Blastoid ID' header in manual_counts.csv"
        )

    headers = [cell.strip() for cell in rows[header_index]]
    manual = {}

    for row in rows[header_index + 1:]:
        # 补齐缺失的 CSV 列
        row += [""] * max(0, len(headers) - len(row))
        record = dict(
            zip(headers, [cell.strip() for cell in row])
        )

        try:
            blastoid_id = int(float(record.get("Blastoid ID", "")))
        except (ValueError, TypeError):
            continue

        # 只读取人工计数记录
        method = record.get("Method", "")
        if method and method.lower() != "manual":
            continue

        manual[blastoid_id] = {}

        for column in ["DAPI", *MARKERS.values()]:
            try:
                value = record.get(column, "")
                manual[blastoid_id][column] = (
                    int(float(value)) if value else None
                )
            except (ValueError, TypeError):
                manual[blastoid_id][column] = None

    if not manual:
        raise ValueError(
            "No manual rows found in manual_counts.csv"
        )

    return manual



def nucleus_scores(marker_img, labels, sid, marker):
    records = []
    for p in regionprops(labels):
        mask = labels == p.label
        ring = ndi.binary_dilation(mask, iterations=3) & (labels == 0)
        if ring.sum() < 5:
            ring = ndi.binary_dilation(mask, iterations=3) & ~mask
        bg = float(np.median(marker_img[ring])) if ring.any() else float(np.median(marker_img))
        inside = marker_img[mask]
        p90 = float(np.percentile(inside, 90))
        records.append({
            "blastoid_id": sid, "nucleus_id": int(p.label), "marker": marker,
            "x": float(p.centroid[1]), "y": float(p.centroid[0]),
            "area_px": int(p.area), "raw_mean_intensity": float(inside.mean()),
            "raw_p90_intensity": p90, "local_background_median": bg,
            "score": p90 - bg,
        })
    return records


def fit_threshold(marker, records, manual):
    by_sample = {}
    for r in records:
        if r["marker"] == marker:
            by_sample.setdefault(r["blastoid_id"], []).append(r["score"])
    usable = [(sid, np.asarray(scores), manual[sid][MARKERS[marker]])
              for sid, scores in by_sample.items()
              if sid in manual and manual[sid].get(MARKERS[marker]) is not None]
    if not usable:
        raise ValueError(f"No manual targets found for {marker}")
    all_scores = np.concatenate([x[1] for x in usable])
    candidates = np.unique(np.r_[all_scores, all_scores.max() + 1e-6])
    best = None
    for t in candidates:
        err = sum(abs(int(np.sum(scores >= t)) - target)
                  for _, scores, target in usable)
        key = (err, -float(t))  # ties: prefer higher threshold to limit false positives
        if best is None or key < best[0]:
            best = (key, float(t))
    return best[1], best[0][0] / len(usable), len(usable)


def shown(image):
    lo, hi = np.percentile(image, [1, 99.7])
    if hi <= lo:
        hi = lo + 1
    a = np.clip((image - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(a, "L").convert("RGB")


def panel(image, labels, records=None, threshold=None, dapi=False):
    im = shown(image)
    rgb = np.array(im)
    rgb[find_boundaries(labels, mode="outer")] = (0, 220, 255)
    im = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(im)
    by_id = {r["nucleus_id"]: r for r in (records or [])}
    for p in regionprops(labels):
        x, y = p.centroid[1], p.centroid[0]
        if dapi:
            colour = (0, 255, 255)
        else:
            r = by_id.get(int(p.label))
            colour = ((30, 255, 80) if r and r["score"] >= threshold
                      else (255, 90, 70))
        draw.ellipse((x-3, y-3, x+3, y+3), outline=colour, width=1)
    return im


def save_overlay(sid, sample, thresholds):
    panels = [("DAPI: cyan = detected nuclei",
               panel(sample["dapi"], sample["labels"], dapi=True))]
    for marker in MARKERS:
        panels.append((f"{marker}: green=positive, red=negative",
                       panel(sample["markers"][marker], sample["labels"],
                             sample["records"][marker], thresholds[marker])))
    w, h = panels[0][1].size
    title_h = 26
    canvas = Image.new("RGB", (2*w, 2*(h+title_h)), "black")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for i, (title, img) in enumerate(panels):
        x, y = (i % 2)*w, (i // 2)*(h+title_h)
        draw.text((x+4, y+4), title, fill="white", font=font)
        canvas.paste(img, (x, y+title_h))
    canvas.save(OUT / "overlays" / f"blastoid_{sid:02d}.png")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT}")
    if not MANUAL_CSV.exists():
        raise FileNotFoundError(f"Manual CSV not found: {MANUAL_CSV}")
    manual = read_manual(MANUAL_CSV)
    dapi_files = sorted(INPUT.glob("*_DAPI_max.tif"))
    if not dapi_files:
        raise FileNotFoundError(f"No *_DAPI_max.tif files in {INPUT}")

    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)
    samples, all_records = {}, []

    for dp in dapi_files:
        match = re.search(r"-(\d+)_DAPI_max$", dp.stem)
        if not match:
            print("SKIP: cannot parse ID:", dp.name)
            continue
        sid = int(match.group(1))
        prefix = dp.stem[:-len("_DAPI_max")]
        dapi = read_image(dp)
        labels = segment_dapi(dapi)
        marker_imgs, records_by_marker = {}, {}
        tifffile.imwrite(str(OUT / "labels" / f"blastoid_{sid:02d}_labels.tif"),
                         labels.astype(np.uint16))

        for marker in MARKERS:
            mp = INPUT / f"{prefix}_{marker}_max{dp.suffix}"
            if not mp.exists():
                raise FileNotFoundError(f"Missing marker file: {mp}")
            img = read_image(mp)
            if img.shape != dapi.shape:
                raise ValueError(f"Shape mismatch: {dp.name} vs {mp.name}")
            marker_imgs[marker] = img
            recs = nucleus_scores(img, labels, sid, marker)
            records_by_marker[marker] = recs
            all_records.extend(recs)

        samples[sid] = {"dapi": dapi, "labels": labels,
                        "dapi_count": len(regionprops(labels)),
                        "markers": marker_imgs, "records": records_by_marker}
        print(f"Blastoid {sid:02d}: DAPI nuclei detected = {samples[sid]['dapi_count']}")

    thresholds, threshold_rows = {}, []
    for marker in MARKERS:
        t, mae, n = fit_threshold(marker, all_records, manual)
        thresholds[marker] = t
        threshold_rows.append({"marker": marker, "score": "nuclear p90 - local background median",
                               "threshold": t, "manual_count_fit_MAE_same_samples": mae,
                               "manual_samples": n})

    with (OUT / "marker_thresholds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=threshold_rows[0].keys())
        w.writeheader()
        w.writerows(threshold_rows)
    with (OUT / "nuclear_marker_intensities.csv").open("w", newline="") as f:
        fields = ["blastoid_id", "nucleus_id", "marker", "x", "y", "area_px",
                  "raw_mean_intensity", "raw_p90_intensity",
                  "local_background_median", "score"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_records)

    count_rows = []
    for sid, s in sorted(samples.items()):
        m = manual.get(sid, {})
        row = {"blastoid_id": sid, "DAPI_manual": m.get("DAPI"),
               "DAPI_predicted": s["dapi_count"]}
        row["DAPI_abs_error"] = (abs(row["DAPI_predicted"] - row["DAPI_manual"])
                                 if row["DAPI_manual"] is not None else None)
        for marker in MARKERS:
            pred = sum(r["score"] >= thresholds[marker] for r in s["records"][marker])
            target = m.get(MARKERS[marker])
            row[f"{marker}_manual"] = target
            row[f"{marker}_predicted"] = int(pred)
            row[f"{marker}_abs_error"] = abs(int(pred)-target) if target is not None else None
        count_rows.append(row)
        save_overlay(sid, s, thresholds)

    with (OUT / "counts_vs_manual.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=count_rows[0].keys())
        w.writeheader()
        w.writerows(count_rows)

    print("\nThresholds fitted against the manual count totals:")
    for marker, t in thresholds.items():
        print(f"  {marker}: {t:.4f}")
    print(f"\nProcessed {len(samples)} blastoids / up to {len(samples)*4} channel images.")
    print("Counts:", OUT / "counts_vs_manual.csv")
    print("Per-nucleus signals:", OUT / "nuclear_marker_intensities.csv")
    print("Thresholds:", OUT / "marker_thresholds.csv")
    print("Overlays:", OUT / "overlays")
    print("IMPORTANT: thresholds are calibrated on these same manual counts;")
    print("the fit error is not independent validation. Inspect overlays first.")


if __name__ == "__main__":
    main()
