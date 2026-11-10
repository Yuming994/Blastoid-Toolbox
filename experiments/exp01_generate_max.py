
from pathlib import Path
import numpy as np
import tifffile

# Original B01 TIFF
input_file = Path(
    "data/raw/blank/"
    "20260514_MBlpb-006_SOX2+GATA3+SOX17_"
    "Blank_Position001-1.tif"
)

# Output folder
output_dir = Path("results/projections")
output_dir.mkdir(parents=True, exist_ok=True)

# Read image: (Z, C, Y, X)
image = tifffile.imread(input_file)

print("Original shape:", image.shape)
print("Original dtype:", image.dtype)

assert image.ndim == 4
assert image.shape[1] == 4

# SOX2: second stored channel (index 1)
sox2_stack = image[:, 1, :, :]

# Maximum Intensity Projection across Z
sox2_max = np.max(sox2_stack, axis=0)

print("MAX shape:", sox2_max.shape)
print("MAX dtype:", sox2_max.dtype)

# Save without changing pixel values
output_file = output_dir / "B01_SOX2_Python_max.tif"
tifffile.imwrite(output_file, sox2_max)

print("Saved:", output_file)
