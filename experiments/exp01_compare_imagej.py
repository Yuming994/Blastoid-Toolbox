"""Compare a Python-generated SOX2 MIP with an ImageJ export at native resolution."""
import argparse
import numpy as np
import tifffile
from src.preprocess import project

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',required=True)
    parser.add_argument('--imagej',required=True)
    args=parser.parse_args()
    imgs,axes,dtype=project(args.raw)
    expected=imgs[1]  # C2 is SOX2: verify on one file first
    actual=tifffile.imread(args.imagej)
    print('Source axes:',axes,'original dtype:',dtype)
    print('Python shape/dtype:',expected.shape,expected.dtype)
    print('ImageJ shape/dtype:',actual.shape,actual.dtype)
    if actual.ndim != 2 or actual.shape != expected.shape:
        print('NOT COMPARABLE: ImageJ export has a different shape or RGB channels. Export full-size single-channel MIP at native bit depth.')
        return 1
    equal=np.array_equal(expected,actual)
    print('Exactly equal:',equal)
    diff=expected.astype(np.float64)-actual.astype(np.float64)
    print('Different pixels:',int(np.count_nonzero(diff)))
    print('Maximum absolute difference:',float(np.max(np.abs(diff))))
    if not equal:
        print('Check crop, 8-bit conversion, LUT/RGB export, channel identity and Z range.')
        return 1
    return 0
if __name__=='__main__':
    raise SystemExit(main())
