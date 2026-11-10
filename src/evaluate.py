"""Compare predicted counts and manual counts via explicit image keys.
Input manual CSV columns: file,manual_count; file matches summary_counts.csv.
"""
import argparse,csv
from pathlib import Path

def read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pred',default='results/counting/summary_counts.csv')
    p.add_argument('--manual',default='data/ground_truth/manual_counts.csv')
    p.add_argument('--out',default='results/evaluation/count_comparison.csv')
    a=p.parse_args()
    predicted=read(Path(a.pred));manual=read(Path(a.manual))
    lookup={r['file']:int(r['manual_count']) for r in manual}
    rows=[]
    for r in predicted:
        if r['status']!='OK' or r['file'] not in lookup:continue
        truth=lookup[r['file']];count=int(r['candidate_count'])
        rows.append(dict(file=r['file'],marker=r['marker'],manual_count=truth,candidate_count=count,signed_error=count-truth,absolute_error=abs(count-truth)))
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True)
    with o.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['file','marker','manual_count','candidate_count','signed_error','absolute_error']);w.writeheader();w.writerows(rows)
    print('Matched:',len(rows),'MAE:',sum(r['absolute_error'] for r in rows)/len(rows) if rows else 'N/A')
    print('Unmatched entries are excluded. Check manual keys.')
if __name__=='__main__':main()
