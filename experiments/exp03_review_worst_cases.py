from pathlib import Path
import re
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from PIL import Image

# ---------- paths ----------
manual_path = Path("data/ground_truth/manual_counts.csv")
summary_path = Path("results/counting/baseline_v1/summary_counts.csv")
proj_dir = Path("results/projections/blank")
count_dir = Path("results/counting/baseline_v1")
out_dir = Path("results/evaluation/worst_cases")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- read manual counts ----------
raw = pd.read_csv(manual_path, header=0)

manual = raw.iloc[1:].copy()
manual.columns = [
    "blastoid", "method",
    "DAPI", "SOX2", "GATA3", "SOX17",
    "SOX2_GATA3", "SOX2_SOX17", "SOX17_GATA3",
    "TRIPLE", "Analysis"
]

manual = manual[["blastoid", "DAPI", "SOX2", "GATA3", "SOX17"]].copy()
manual["blastoid"] = manual["blastoid"].astype(int)

manual_long = manual.melt(
    id_vars="blastoid",
    var_name="marker",
    value_name="manual_count"
)

# Convert manual counts to numeric values
manual_long["manual_count"] = pd.to_numeric(
    manual_long["manual_count"],
    errors="raise"
)



# ---------- read auto summary ----------
auto = pd.read_csv(summary_path)

# Convert automatic counts to numeric values
auto["candidate_count"] = pd.to_numeric(
    auto["candidate_count"],
    errors="raise"
)

# extract blastoid id from filename
auto["blastoid"] = auto["file"].str.extract(r"Position\d+-(\d+)").astype(int)

# unify marker naming
auto["marker"] = auto["marker"].replace({
    "SOX2+": "SOX2",
    "GATA3+": "GATA3",
    "SOX17+": "SOX17"
})

# ---------- merge ----------
merged = auto.merge(manual_long, on=["blastoid", "marker"], how="left")
merged["error"] = merged["candidate_count"] - merged["manual_count"]
merged["abs_error"] = merged["error"].abs()

# save full merged table
merged.to_csv(out_dir / "baseline_v1_vs_manual.csv", index=False)

# sort by worst error
worst = merged.sort_values("abs_error", ascending=False).reset_index(drop=True)
worst.to_csv(out_dir / "worst_cases_ranked.csv", index=False)

print("Top 10 worst cases:")
print(worst[["blastoid", "marker", "manual_count", "candidate_count", "error", "file"]].head(10))

# ---------- generate review panels ----------
top_n = 8

for i in range(min(top_n, len(worst))):
    row = worst.iloc[i]

    file_name = row["file"]
    marker = row["marker"]
    blastoid = row["blastoid"]
    manual_count = row["manual_count"]
    auto_count = row["candidate_count"]
    error = row["error"]

    raw_img_path = proj_dir / file_name
    overlay_path = count_dir / file_name.replace(".tif", "_overlay.png")
    dapi_path = proj_dir / file_name.replace(f"_{marker}_max.tif", "_DAPI_max.tif")

    if not raw_img_path.exists():
        print(f"Missing raw image: {raw_img_path}")
        continue
    if not overlay_path.exists():
        print(f"Missing overlay image: {overlay_path}")
        continue
    if not dapi_path.exists():
        print(f"Missing DAPI image: {dapi_path}")
        continue

    raw_img = tifffile.imread(raw_img_path)
    dapi_img = tifffile.imread(dapi_path)
    overlay_img = Image.open(overlay_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_img, cmap="gray")
    axes[0].set_title(f"Raw {marker} MAX")
    axes[0].axis("off")

    axes[1].imshow(overlay_img)
    axes[1].set_title("Auto detection overlay")
    axes[1].axis("off")

    axes[2].imshow(dapi_img, cmap="gray")
    axes[2].set_title("Reference DAPI MAX")
    axes[2].axis("off")

    fig.suptitle(
        f"Blastoid {blastoid} | {marker} | manual={manual_count}, auto={auto_count}, error={error}",
        fontsize=14
    )

    out_path = out_dir / f"rank_{i+1:02d}_B{blastoid:02d}_{marker}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

print(f"\nSaved review panels to: {out_dir}")