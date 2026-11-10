"""Run pilot nucleus-centre detector across preprocessed 2D MIP TIFFs."""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from src.count_nuclei import candidates, annotated_overlay

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,default=Path('results/projections'))
    p.add_argument('--output-dir',type=Path,default=Path('results/counting'))
    p.add_argument('--min-distance',type=int,default=8)
    p.add_argument('--contrast-threshold',type=float,default=10.)
    p.add_argument('--min-smoothed-intensity',type=float,default=18.)
    a=p.parse_args()
    files=sorted(x for x in a.input_dir.rglob('*') if x.suffix.lower() in ('.tif','.tiff')) if a.input_dir.exists() else []
    if not files:
        p.error('No TIFF files in input directory. Run the preprocessing step first.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for x in files:
        rel=x.relative_to(a.input_dir)
        out=a.output_dir/rel.parent
        out.mkdir(parents=True,exist_ok=True)
        record={'file':str(rel),'marker':next((k for k in ('DAPI','SOX2','GATA3','SOX17') if x.stem.endswith('_'+k+'_max')),'UNKNOWN'),'candidate_count':'','status':'','error':''}
        try:
            img=tifffile.imread(x)
            pts,norm,contrast=candidates(img,min_distance=a.min_distance,contrast_threshold=a.contrast_threshold,min_smoothed_intensity=a.min_smoothed_intensity)
            record['candidate_count']=len(pts)
            record['status']='OK'
            annotated_overlay(norm,pts,len(pts)).save(out/(x.stem+'_overlay.png'))
            with (out/(x.stem+'_centres.csv')).open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=['id','x','y','local_contrast','smoothed_intensity','raw_intensity'])
                w.writeheader()
                for i,item in enumerate(pts,1):w.writerow({'id':i,**item})
        except Exception as ex:
            record.update(status='ERROR',error=str(ex))
        rows.append(record)
        print(record['status'],record['file'],record['candidate_count'],record['error'])
    with (a.output_dir/'summary_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('Summary:',(a.output_dir/'summary_counts.csv').resolve())
if __name__=='__main__':main()
