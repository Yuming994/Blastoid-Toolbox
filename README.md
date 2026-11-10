# Blastoid-Toolbox (research starter)

This starter combines the **original Pilot v1 nucleus-centre detector** with batch conversion, count summaries and ImageJ projection validation. The nucleus-centre algorithm is **unvalidated**, not an established cell-counting system. Changes in image scale/bit depth can affect its predictions.

## Folders
- `data/raw/blank/`: ORIGINAL 4-channel Z-stack TIFFs. Never edit these.
- `data/exported/blank/`: Jessie’s 120 exported TIFFs for reference (not necessarily MIPs).
- `data/validation/`: ImageJ-exported full-resolution, native-bit-depth SOX2 MIP for one TIFF.
- `data/ground_truth/manual_counts.csv`: manual counts in `file,manual_count` format; an empty template is provided. Export your Excel table to CSV or map the manual counts later.
- `src/preprocess.py`: channel/Z-aware MIP converter (from prior script); preserves original numeric dtype and **rejects ambiguous TIFFs**. Names assume C1 DAPI, C2 SOX2, C3 GATA3, C4 SOX17; verify first.
- `src/count_nuclei.py`: the previous Pilot v1 local-peak algorithm, unchanged.
- `src/batch_process.py`: run detector on all generated 2D MIPs; outputs PNG overlays, per-image coordinates and `summary_counts.csv`.
- `src/evaluate.py`: optional manual count comparisons after matching file keys.
- `src/spatial_analysis.py`: placeholder only, not implemented.
- `experiments/`: commands to inspect first raw file and compare MIPs.
- `results/`: generated after running commands (empty at the beginning).

## One step at a time (VS Code Terminal at project root)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/exp01_projection_check.py
```

**Send the output `axes`, `shape`, `dtype` for confirmation before converting.**

When axes and ImageJ reference have been confirmed:
```sh
python -m src.preprocess --input-dir data/raw/blank --output-dir results/projections
python -m experiments.exp01_compare_imagej --raw 'data/raw/blank/YOUR_EXACT_RAW_FILE.tif' --imagej data/validation/B01_SOX2_ImageJ_max.tif
```
The comparison intentionally refuses mismatched crop/resizing or RGB exports. `B01_SOX2_max.tif` from an earlier cropped 8-bit experiment may not be suitable as a pixel-equality reference to the original full-resolution raw stack.

After validation:
```sh
python -m src.batch_process --input-dir results/projections --output-dir results/counting
```

The initial candidate detector uses parameters fixed from an earlier single-image pilot; its default numerical scale is NOT calibrated for your original raw-resolution TIFFs or all four channels. It can fail and must be benchmarked before trusting counts.

### Notes
- The earlier manual counting PDF and your own Excel workbook are sources of labels; CSV = comma-separated values, a simple text table that Excel/Numbers opens. We include only a blank CSV **template**, not invented counts.
- Some Leica TIFFs store channels as separate series or ambiguous page order. If converter skips a file, inspect its metadata, do **not** force a guessed axis.
- The first TIFF in `data/raw/blank/` is used for inspection; actual TIFF files are not included in this ZIP.
