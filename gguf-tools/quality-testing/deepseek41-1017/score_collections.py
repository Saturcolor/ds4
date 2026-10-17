"""Complete exact-token collection comparison with case-bootstrap intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_archives import Case,compare


def summarize(records,bootstrap=1000):
    if not records:raise ValueError('empty score group')
    names=set(records[0]['arrays'])
    if any(set(row['arrays'])!=names for row in records):raise ValueError('metric coverage differs')
    lengths=np.array([len(next(iter(row['arrays'].values()))) for row in records],dtype=np.int64)
    if np.any(lengths<=0):raise ValueError('empty score case')
    rng=np.random.default_rng(1017)
    indices=rng.integers(0,len(records),size=(bootstrap,len(records)))
    result={'cases':len(records),'positions':int(lengths.sum()),'metrics':{}}
    for name in sorted(names):
        arrays=[np.asarray(row['arrays'][name],dtype=np.float64) for row in records]
        if any(value.shape!=(size,) or not np.isfinite(value).all() for value,size in zip(arrays,lengths)):
            raise ValueError('invalid per-position metric')
        values=np.concatenate(arrays);sums=np.array([value.sum() for value in arrays])
        estimates=sums[indices].sum(1)/lengths[indices].sum(1)
        result['metrics'][name]={'token_weighted_mean':float(values.mean()),
                                'case_mean':float(np.mean([value.mean() for value in arrays])),
                                'p50':float(np.quantile(values,.5)),'p95':float(np.quantile(values,.95)),
                                'p99':float(np.quantile(values,.99)),'maximum':float(values.max()),
                                'case_bootstrap_95ci':np.quantile(estimates,[.025,.975]).tolist()}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--vision-whole-pipeline',action='store_true')
    a=p.parse_args();state=json.loads((a.reference/'collection.json').read_text())
    ids=state['completed_cases']
    if not ids or len(ids)!=len(set(ids)):raise ValueError('empty or duplicate collection cases')
    if state.get('collection_complete') is not True or set(state.get('selected_case_ids',[]))!=set(ids):
        raise ValueError('the entire selected collection must be complete before scoring')
    reference_ids={path.parent.name for path in a.reference.glob('*/case.json')}
    candidate_ids={path.parent.name for path in a.candidate.glob('*/case.json')}
    if reference_ids!=set(ids) or candidate_ids!=set(ids):
        raise ValueError('reference/candidate must cover exactly every committed collection case')
    a.out.mkdir(parents=True,exist_ok=False);records=[]
    for case_id in ids:
        reference=Case(a.reference/case_id);candidate=Case(a.candidate/case_id)
        if not reference.meta.get('native_outputs_collected') or candidate.meta.get('native_outputs_collected'):
            raise ValueError('requires native reference and explicitly identified candidate')
        path=a.out/(case_id+'.json')
        compare(a.reference/case_id,a.candidate/case_id,path,vision_whole_pipeline=a.vision_whole_pipeline)
        with np.load(path.with_suffix('.per-position.npz'),allow_pickle=False) as data:
            arrays={key:data[key].copy() for key in data.files}
        records.append({'case_id':case_id,'meta':reference.meta,'arrays':arrays})
    groups={}
    for row in records:
        meta=row['meta'];keys=['group:'+meta['group']]
        if 'context_bucket' in meta:keys.append('context:'+str(meta['context_bucket']))
        for key in keys:groups.setdefault(key,[]).append(row)
        kinds=meta.get('position_kinds',['continuation']*len(meta['positions']))
        if len(kinds)!=len(meta['positions']):raise ValueError('position-kind coverage mismatch')
        for kind in sorted(set(kinds)):
            mask=np.asarray(kinds)==kind
            groups.setdefault('position_kind:'+kind,[]).append(dict(row,arrays={key:value[mask] for key,value in row['arrays'].items()}))
    result={'schema':'deepseek-v4.1-flash-1017.quant-score.v1','complete_committed_collection':True,
            'model_revision':records[0]['meta']['model_revision'],'reference_identity':state['identity'],
            'bootstrap':'1000 case-resampled replicates; token-weighted mean; seed1017',
            'interpretation':'measures conversion, quantization and engine arithmetic together; no acceptance threshold implied',
            'overall':summarize(records),'groups':{key:summarize(value) for key,value in groups.items()},
            'worst_forward_kl_cases':sorted([{'case_id':row['case_id'],'mean_forward_kl':float(row['arrays']['forward_kl'].mean())}
                                             for row in records],key=lambda row:row['mean_forward_kl'],reverse=True)[:20]}
    if a.vision_whole_pipeline:
        result['interpretation']='Whole image pipeline includes existing antirez preprocessing and encoder, quantization and engine arithmetic; not isolated quantization error.'
    (a.out/'SCORE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'event':'complete_collection_score','cases':len(ids),'positions':result['overall']['positions']}),flush=True)


if __name__=='__main__':main()
