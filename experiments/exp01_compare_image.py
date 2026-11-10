
from pathlib import Path
import numpy as np
import tifffile

# Two images to compare
imagej_path = Path("data/validation/B01_SOX2_max.tif")
python_path = Path("results/projections/B01_SOX2_Python_max.tif")

imagej = tifffile.imread(imagej_path)
python_img = tifffile.imread(python_path)

print("ImageJ shape:", imagej.shape)
print("Python shape:", python_img.shape)

print("ImageJ dtype:", imagej.dtype)
print("Python dtype:", python_img.dtype)

if imagej.shape != python_img.shape:
    print("\nFAILED: Image dimensions are different.")
else:
    difference = (
        imagej.astype(np.float64)
        - python_img.astype(np.float64)
    )

    different_pixels = np.count_nonzero(difference)
    max_error = np.max(np.abs(difference))

    print("\nDifferent pixels:", different_pixels)
    print("Maximum absolute error:", max_error)

    if different_pixels == 0:
        print("\nPASS: Pixel values are identical!")
    else:
        print("\nNOT IDENTICAL: Further investigation needed.")
