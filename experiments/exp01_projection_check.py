
from pathlib import Path
import tifffile

# Folder containing the original TIFF files
folder = Path("data/raw/blank")

# Find TIFF files
files = sorted(
    list(folder.glob("*.tif")) +
    list(folder.glob("*.tiff"))
)

print(f"Found {len(files)} TIFF files.")

if not files:
    raise FileNotFoundError("No TIFF files found.")

# Inspect only the first image
file = files[0]

print("\nFile:", file.name)

with tifffile.TiffFile(file) as tif:
    print("Number of series:", len(tif.series))

    for i, series in enumerate(tif.series):
        print(f"\nSeries {i}")
        print("Shape:", series.shape)
        print("Axes:", series.axes)
        print("Dtype:", series.dtype)

    # Print only metadata keys, not the full metadata
    if tif.imagej_metadata:
        print("\nImageJ metadata keys:")
        print(list(tif.imagej_metadata.keys()))

print("\nInspection completed.")

