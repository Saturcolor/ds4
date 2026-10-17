"""Compare saved natural image embeddings, without model inference or Torch."""
import argparse
import json
from pathlib import Path
import struct
import ml_dtypes  # Registers NumPy's BF16 dtype for safetensors.
import numpy as np
from safetensors.numpy import load_file
from q2_packets import file_hash
from reference_contract import MODEL_REVISION,require_token_contract


def candidate_features(path,shape,grid):
    path=Path(path)
    if path.stat().st_size!=16+4*int(np.prod(shape)):
        raise ValueError('incomplete candidate image feature output')
    with path.open('rb') as stream:header=struct.unpack('<IIII',stream.read(16))
    if header!=(*shape,*grid):raise ValueError('candidate natural image feature geometry differs')
    data=np.memmap(path,mode='r',dtype='<f4',offset=16,shape=shape)
    if not np.isfinite(data).all():raise ValueError('nonfinite candidate image features')
    return data


def metrics(reference,candidate):
    p=np.asarray(reference,dtype=np.float64);q=np.asarray(candidate,dtype=np.float64)
    if p.ndim!=2 or p.shape!=q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('invalid paired image features')
    delta=q-p;pn=np.linalg.norm(p,axis=1);qn=np.linalg.norm(q,axis=1)
    denominator=pn*qn
    cosine=np.divide((p*q).sum(1),denominator,out=np.zeros(len(p)),where=denominator>0)
    cosine[(pn==0)&(qn==0)]=1
    energy=float(np.square(p).sum());error=float(np.square(delta).sum())
    return {'rows':len(p),'values':p.size,'squared_error':error,
            'rmse':float(np.sqrt(error/p.size)),'max_abs':float(np.abs(delta).max()),
            'mean_row_cosine':float(np.clip(cosine,-1,1).mean()),
            'relative_l2':float(np.sqrt(error/energy)) if energy else None,
            'diagnostic_candidate_bf16_equal_fraction':float(np.mean(
                np.asarray(candidate).astype(ml_dtypes.bfloat16).astype(np.float32)==
                np.asarray(reference,dtype=np.float32)))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--pack',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,help='Explicit selected packet manifest; raw files stay under --pack')
    parser.add_argument('--allow-subset',action='store_true',help='Report explicitly selected cases without claiming full collection coverage')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    state=json.loads((args.reference/'collection.json').read_text())
    pack=json.loads((args.manifest or args.pack/'pack.json').read_text())
    if state.get('collection_complete') is not True or pack['source_revision']!=MODEL_REVISION:
        raise ValueError('complete exact1017 reference collection required')
    committed=state['completed_cases'];ids=committed
    items={item['case']['case_id']:item for item in pack['cases']}
    if args.allow_subset:
        ids=[item['case']['case_id'] for item in pack['cases']]
        if not ids or not set(ids)<=set(committed):raise ValueError('unknown or empty selected image cases')
    if (len(committed)!=len(set(committed)) or set(committed)!=set(state['selected_case_ids']) or
            len(ids)!=len(set(ids)) or
            len(items)!=len(pack['cases']) or set(items)!=set(ids)):
        raise ValueError('image comparison must cover the entire frozen collection')
    results=[]
    for case_id in ids:
        case=json.loads((args.reference/case_id/'case.json').read_text());require_token_contract(case)
        item=items[case_id]
        if Path(item['raw']).name!=item['raw']:raise ValueError('unsafe candidate feature path')
        if item['case']!=case:raise ValueError('candidate packet frozen case differs')
        files=case['vision_features']['files']
        aligners={entry['image_index']:entry for entry in files if entry['kind']=='aligner'}
        if len(aligners)!=len(case['image_identity']):raise ValueError('incomplete native aligner features')
        for index,image in enumerate(case['image_identity']):
            entry=aligners[index]
            if Path(entry['file']).name!=entry['file']:raise ValueError('unsafe native feature path')
            native=args.reference/case_id/entry['file']
            if file_hash(native)!=entry['sha256']:raise ValueError('native image feature integrity differs')
            reference=load_file(native)['features']
            if reference.dtype!=np.dtype(ml_dtypes.bfloat16) or list(reference.shape)!=entry['shape']:
                raise ValueError('native image feature shape/dtype differs')
            shape=tuple(entry['shape']);grid=((image['n_vit_h']+2)//3,(image['n_vit_w']+2)//3)
            if shape!=(grid[0]*grid[1],5120):raise ValueError('native natural aligner geometry differs')
            raw=args.pack/(item['raw']+f'.vision-{index}.f32')
            candidate=candidate_features(raw,shape,grid)
            results.append(dict(case_id=case_id,image_index=index,raw_image_sha256=image['raw_sha256'],
                                candidate_features_sha256=file_hash(raw),**metrics(reference,candidate)))
    if not results:raise ValueError('no paired image features')
    total_values=sum(row['values'] for row in results);total_rows=sum(row['rows'] for row in results)
    result={'model_revision':MODEL_REVISION,'cases':len(ids),'image_occurrences':len(results),
            'complete_selected_cases':True,'complete_committed_collection':ids==committed,
            'stage':'natural aligner output before prompt sentinel insertion',
            'interpretation':'Existing whole encoder/preprocessing arithmetic against native BF16 reference; not isolated language quantization error and no acceptance threshold implied.',
            'overall':{'rmse':float(np.sqrt(sum(row['squared_error'] for row in results)/total_values)),
                       'max_abs':max(row['max_abs'] for row in results),
                       'mean_row_cosine':sum(row['mean_row_cosine']*row['rows'] for row in results)/total_rows},
            'images':results}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='images'}),flush=True)


if __name__=='__main__':main()
