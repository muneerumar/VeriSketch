"""Export aggregate-only experiment results for an explicit public release.

This removes record-level predictions, post identifiers, client profiles and
client identifiers. It does not establish differential privacy of the release.
"""
from pathlib import Path
import argparse,csv,json,math,statistics

GROUPS=('baseline','robust','joint','dp','elsa','fsweep','sketchdim','nullspace')
BLOCKED={'test_pred','post_id','post_ids','tweet_text','tokens','rationale','targets',
         'partition_hist','partition_sizes','malicious','selected','mal_selected',
         'accepted','real_sizes','shadow_size_range','sketch_norms','key_sha256',
         'source_index','v1_index','param_hash'}

def clean(obj):
    if isinstance(obj,dict):
        return {k:clean(v) for k,v in obj.items() if k not in BLOCKED and
                not k.startswith('_') and not k.endswith('_per_client_mae')}
    if isinstance(obj,list):return [clean(v) for v in obj]
    if isinstance(obj,float) and not math.isfinite(obj):return None
    return obj

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')

def mean(xs):return statistics.fmean(xs) if xs else None

def run_summary(o):
    h=o.get('history',[]);mal=set(o.get('malicious',[]));K=o['cfg']['K']
    sizes=o.get('partition_sizes',[1]*K)
    accepted=[len(t.get('selected',[])) for t in h]
    incl=[len(set(t.get('selected',[]))&mal)/len(mal) for t in h] if mal else []
    excl=[1-len(set(t.get('selected',[]))-mal)/max(K-len(mal),1) for t in h]
    weight=[]
    for t in h:
        ids=t.get('selected',[]);den=sum(sizes[i] for i in ids)
        weight.append(sum(sizes[i] for i in ids if i in mal)/max(den,1))
    return dict(accepted_size_mean=mean(accepted),malicious_inclusion_mean=mean(incl),
                honest_exclusion_mean=mean(excl),malicious_weight_fraction_mean=mean(weight),
                rounds=len(h),aborted_rounds=o.get('n_aborted',0),
                all_accepted_checks_pass=all(t.get('aborted',False) or all(t.get(k,True) for k in
                    ['verified','sketch_consistent','in_range']) for t in h))

def export_run(o,labels=None):
    # Export known fields only; newly introduced private metadata stays local.
    out={k:clean(o[k]) for k in ['n_aborted','sigma','wall','best_round'] if k in o}
    config_fields={'K','T','seed','agg','lr','batch','local_epochs','dropout','d_hidden','clip_update',
        'f_assumed','attack','n_malicious','partition','eval_every','feat','dp_q','dp_lr','dp_clip',
        'select_best_val','rows','width','sigma_mode','sketch_layers','class_weighting','normalisation','eval_test'}
    out['cfg']={k:clean(v) for k,v in o['cfg'].items() if k in config_fields}
    out['cfg']['dp']=({k:v for k,v in o['cfg']['dp'].items() if k in {'epsilon','delta','clip','q'}}
                      if o['cfg'].get('dp') else None)
    out['test']={k:clean(v) for k,v in o['test'].items() if k in
                 {'accuracy','macro_f1','tmr','far','per_class_f1','confusion','per_class_recall'}}
    history_fields={'round','t_train','t_round','t_client_mask','t_sketch','t_unmask','t_verify',
        'aborted','verified','in_range','sketch_consistent','val_accuracy','val_macro_f1','val_tmr','val_far',
        'upd_norm_median','upd_norm_max'}
    out['history']=[{k:clean(v) for k,v in t.items() if k in history_fields} for t in o.get('history',[])]
    if o.get('mia'):out['mia']={k:v for k,v in o['mia'].items() if k in {'auc','tpr_at_1fpr'}}
    out['release_schema']='aggregate-only-v1';out['run_summary']=run_summary(o)
    for original,new in zip(o.get('history',[]),out.get('history',[])):
        ids=original.get('selected',[]);mal=set(o.get('malicious',[]))
        new['accepted_count']=len(ids);new['accepted_malicious_count']=len(set(ids)&mal)
    pred=o.get('test_pred')
    if pred is not None and labels is not None:
        if len(pred)!=len(labels):raise ValueError('Prediction/label length mismatch')
        C=max(labels)+1;cm=[[0]*C for _ in range(C)]
        for y,p in zip(labels,pred):cm[y][p]+=1
        out['test']['confusion']=cm
    return out

def export_results(source,destination,manifest=None):
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Use a separate destination')
    # A fresh staging directory prevents old private files surviving re-export.
    if destination.exists() and any(destination.iterdir()):raise ValueError('Destination must be empty')
    labels=None
    if manifest:
        with Path(manifest).open() as f:labels=[int(r['label']) for r in csv.DictReader(f) if r['split']=='test']
    count=0
    for group in GROUPS:
        for p in sorted((source/group).glob('*.json')):
            o=json.loads(p.read_text());write(destination/group/p.name,export_run(o,labels));count+=1
    for folder,name in [('bench','bench.json'),('tamper','tamper_results.json'),
                        ('fidelity','sketch_fidelity.json'),('leak','leak_results.json'),
                        ('moderation','moderation.json')]:
        p=source/folder/name
        if p.exists():
            value=json.loads(p.read_text())
            if folder=='moderation':
                # The legacy label was misleading: this is a binary harm threshold,
                # not six-class argmax. Preserve every measured numeric value.
                for record in value:
                    for point in record.get('points',[])+record.get('second_source',{}).get('points',[]):
                        if point.get('rule')=='argmax':point['rule']='harm05'
            write(destination/folder/name,value)
    p=source/'near_dup_audit.json'
    if p.exists():write(destination/p.name,json.loads(p.read_text()))
    # Aggregate each attribution model; no post rows or post identifiers leave the private folder.
    rows=[]
    for p in sorted((source/'xai').glob('*_xai.csv')):
        with p.open() as f:rr=list(csv.DictReader(f))
        if not rr:continue
        fields=[k for k in rr[0] if k.startswith(('pred_','gold_')) or k in
                ('correct','comprehensiveness','sufficiency','ig_residual','ig_delta')]
        means={}
        for k in fields:
            vals=[]
            for row in rr:
                try:v=float(row[k])
                except (TypeError,ValueError):continue
                if math.isfinite(v):vals.append(v)
            means[k]=mean(vals)
        rows.append(dict(model=p.stem.removesuffix('_xai'),n_evaluated=len(rr),means=means))
    if rows:write(destination/'xai'/'model_summaries.json',rows)
    for p in sorted((source/'xai').glob('*_cls.json')):
        write(destination/'xai'/p.name,export_run(json.loads(p.read_text())))
    for p in sorted((source/'analysis').glob('*.csv')):
        with p.open() as f:rr=list(csv.DictReader(f))
        if rr and any(k in BLOCKED or k.endswith('_per_client_mae') for k in rr[0]):continue
        target=destination/'analysis'/p.name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(p.read_text())
    return count

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='results');p.add_argument('--destination',required=True)
    p.add_argument('--manifest',help='Local split CSV; used only to derive aggregate confusion matrices')
    a=p.parse_args();print('Exported',export_results(a.source,a.destination,a.manifest),'aggregate run files')

if __name__=='__main__':main()
