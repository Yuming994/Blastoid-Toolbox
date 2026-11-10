#!/usr/bin/env python3
"""Batch extract four lossless channel-wise maximum intensity projections from TIFF Z-stacks.
Checks TIFF axis metadata; refuses ambiguous axes rather than guessing channel identity.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import tifffile

MARKERS = ('DAPI', 'SOX2', 'GATA3', 'SOX17')

def project(path):
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = series.axes.upper()
        arr = series.asarray()
    # Arbitrary singleton time/position dimensions are safe to remove.
    while any(a not in 'ZCYX' for a in axes):
        i = next(i for i,a in enumerate(axes) if a not in 'ZCYX')
        if arr.shape[i] != 1:
            raise ValueError(f'ambiguous non-singleton axis {axes[i]}={arr.shape[i]} in {axes}; inspect manually')
        arr = np.take(arr, 0, axis=i)
        axes = axes[:i] + axes[i+1:]
    if len(set(axes)) != len(axes) or 'C' not in axes or 'Y' not in axes or 'X' not in axes:
        raise ValueError(f'unsupported/ambiguous axes={axes}, shape={arr.shape}; channel order must be verified')
    if arr.shape[axes.index('C')] != 4:
        raise ValueError(f'expected 4 channels in {axes}, got {arr.shape[axes.index("C")]}')
    if 'Z' not in axes:
        raise ValueError(f'expected Z-stack, got {axes}; do not auto-convert composite or channel images')
    arr = np.moveaxis(arr, [axes.index(a) for a in 'CZ YX'.replace(' ','')], [0,1,2,3])
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f'unexpected dtype {arr.dtype}')
    return arr.max(axis=1), axes, str(arr.dtype)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',required=True,type=Path)
    parser.add_argument('--output-dir',default=Path('mip_images'),type=Path)
    args=parser.parse_args()
    if not args.input_dir.is_dir(): parser.error('input folder not found')
    paths=sorted(p for p in args.input_dir.rglob('*') if p.is_file() and p.suffix.lower() in ('.tif','.tiff'))
    if not paths: parser.error('no TIFF images found')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for p in paths:
        row={'source':str(p.relative_to(args.input_dir)),'axes':'','dtype':'','status':'','note':'','output_prefix':''}
        try:
            imgs,axes,dtype=project(p)
            # Preserve relative folder hierarchy and full stem; avoid naming collisions.
            prefix=args.output_dir/p.relative_to(args.input_dir).with_suffix('')
            prefix.parent.mkdir(parents=True,exist_ok=True)
            for marker,img in zip(MARKERS,imgs):
                tifffile.imwrite(prefix.parent/f'{prefix.name}_{marker}_max.tif', img, photometric='minisblack')
            row.update(axes=axes,dtype=dtype,status='OK',output_prefix=str(prefix))
        except Exception as e:
            row.update(status='SKIPPED',note=str(e))
        results.append(row)
        print(f'{row["status"]}: {row["source"]} {row["note"]}')
    out=args.output_dir/'conversion_report.csv'
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
    print(f'Converted {sum(r["status"]=="OK" for r in results)}/{len(results)}; report: {out}')

if __name__=='__main__':main()
