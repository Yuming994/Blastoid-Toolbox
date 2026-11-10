from pathlib import Path
import csv
import numpy as np
import tifffile

# Input and output folders
input_dir = Path("data/raw/blank")
output_dir = Path("results/projections/blank")
output_dir.mkdir(parents=True, exist_ok=True)

# Report file
report_path = output_dir / "projection_report.csv"

# Channel names based on current project convention
channel_names = ["DAPI", "SOX2", "GATA3", "SOX17"]

# Find TIFF files
files = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))

print(f"Found {len(files)} TIFF files.")

rows = []

for file in files:
    try:
        image = tifffile.imread(file)

        # Expect shape (Z, C, Y, X)
        if image.ndim != 4:
            raise ValueError(f"Expected 4D image, got shape {image.shape}")

        if image.shape[1] != 4:
            raise ValueError(f"Expected 4 channels, got shape {image.shape}")

        for c, channel_name in enumerate(channel_names):
            stack = image[:, c, :, :]
            max_proj = np.max(stack, axis=0)

            output_file = output_dir / f"{file.stem}_{channel_name}_max.tif"
            tifffile.imwrite(output_file, max_proj)

        rows.append({
            "file": file.name,
            "status": "success",
            "shape": str(image.shape),
            "dtype": str(image.dtype)
        })

        print(f"Processed: {file.name}")

    except Exception as e:
        rows.append({
            "file": file.name,
            "status": f"failed: {e}",
            "shape": "",
            "dtype": ""
        })
        print(f"Failed: {file.name} -> {e}")

# Save report
with open(report_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "status", "shape", "dtype"])
    writer.writeheader()
    writer.writerows(rows)

print("\nDone.")
print(f"Projection images saved to: {output_dir}")
print(f"Report saved to: {report_path}")