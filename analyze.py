"""Summarize aggregate-only results without individual records or training dependencies."""
from pathlib import Path
import argparse,csv,json,statistics,math

GROUPS=('baseline','robust','joint','dpctl','dp','elsa','fsweep','sketchdim','nullspace')
def summarize(root):
    groups={};checked=0
    for group in GROUPS:
        for p in sorted((Path(root)/group).glob('*.json')):
            o=json.loads(p.read_text());c=o['cfg'];t=o['test']
            if o.get('release_schema')!='aggregate-only-v2':raise ValueError('Unexpected result schema: '+str(p))
            cm=t.get('confusion')
            if cm:
                n=sum(map(sum,cm));C=len(cm);den=[sum(cm[i])+sum(r[i] for r in cm) for i in range(C)]
                recomputed={'accuracy':sum(cm[i][i] for i in range(C))/n,
                    'macro_f1':statistics.fmean(2*cm[i][i]/den[i] if den[i] else 0 for i in range(C)),
                    'tmr':sum(r[0] for r in cm[1:])/sum(map(sum,cm[1:])),
                    'far':sum(cm[0][1:])/sum(cm[0])}
                for k,v in recomputed.items():
                    if abs(v-t[k])>1e-10:raise ValueError('Metric does not match confusion matrix: '+str(p))
                checked+=1
            key=(group,c['agg'] if c['K']>1 else 'central',str(c['partition']),c.get('attack','none'),
                 ('none' if not c.get('dp') else ('sigma0' if c['dp'].get('epsilon') is None else str(c['dp']['epsilon']))))
            groups.setdefault(key,[]).append(t)
    rows=[]
    for key,values in sorted(groups.items()):
        row=dict(zip(('group','method','partition','attack','epsilon'),key));row['runs']=len(values)
        for m in ('accuracy','macro_f1','tmr','far'):
            v=[x[m] for x in values];row[m+'_mean']=statistics.fmean(v)
            row[m+'_sd']=statistics.stdev(v) if len(v)>1 else 0.0
        rows.append(row)
    return rows,checked

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='public_results')
    p.add_argument('--output',default='results/aggregate_summary.csv');a=p.parse_args()
    rows,checked=summarize(a.input)
    if not rows:raise SystemExit('No aggregate runs found')
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Checked {checked} confusion matrices; summarized {sum(r["runs"] for r in rows)} runs into {target}')

if __name__=='__main__':main()
