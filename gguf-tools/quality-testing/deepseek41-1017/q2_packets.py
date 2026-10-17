"""Pack exact token IDs for the C scorer and import checked binary logit frames."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
import numpy as np
from reference_contract import require_token_contract,MODEL_REVISION


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def pack(records,out,*,vision=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    jobs=[];manifest=[]
    for case in records:
        require_token_contract(case)
        if case['vocab']!=129280:raise ValueError('native1017 vocabulary required')
        if bool(case.get('image_identity'))!=vision:raise ValueError('explicit matching text/vision packet mode required')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',case['case_id']):raise ValueError('unsafe case ID')
        n,m=len(case['input_ids']),len(case['positions'])
        if not 1<=n<=131072 or not 1<=m<=n:raise ValueError('packet dimensions out of range')
        path=out/(case['case_id']+'.bin')
        with path.open('xb') as f:
            f.write(b'DS411017'+struct.pack('<IIII',case['vocab'],n,m,0))
            for name in ('input_ids','positions','target_ids'):
                f.write(np.asarray(case[name],dtype='<u4').tobytes())
        raw=out/(case['case_id']+'.raw')
        jobs.append(str(path.resolve())+'\t'+str(raw.resolve())+'\n')
        manifest.append({'case':case,'packet':path.name,'packet_sha256':file_hash(path),'raw':raw.name})
    (out/'jobs.tsv').write_text(''.join(jobs))
    result={'source_revision':MODEL_REVISION,'cases':manifest,
            'required_context':max(1024,((max(len(row['case']['input_ids']) for row in manifest)+256+255)//256)*256)}
    (out/'pack.json').write_text(json.dumps(result,ensure_ascii=False)+'\n')
    return result


def pack_vision(records,out,images,assessment):
    """Freeze actual raw image paths and separately identify candidate preprocessing."""
    images=Path(images).resolve()
    checked=json.loads(Path(assessment).read_text())
    if not checked['raw_images_geometry_and_semantic_layout_match']:
        raise ValueError('image geometry and semantic token layout must be qualified')
    known={(row['case_id'],row['image']):row for row in checked['rows']}
    result=pack(records,out,vision=True);out=Path(out);jobs=[]
    for item in result['cases']:
        case=item['case'];actual=[];spans=out/(case['case_id']+'.images')
        if not 1<=len(case['image_identity'])<=16:raise ValueError('image count outside scorer limit')
        with spans.open('xb') as stream:
            stream.write(b'DS41IMG1'+struct.pack('<I',len(case['image_identity'])))
            previous=0
            for identity in case['image_identity']:
                image=(images/identity['file']).resolve()
                if not image.is_relative_to(images) or file_hash(image)!=identity['raw_sha256']:
                    raise ValueError('frozen image bytes differ')
                row=known[(case['case_id'],identity['file'])]
                if row['raw_sha256']!=identity['raw_sha256'] or row['start']!=identity['start']:
                    raise ValueError('preprocessing assessment identity differs')
                start=identity['start'];count=row['image_token_count']
                if start<previous or case['input_ids'][start:start+count]!=[129264]*count:
                    raise ValueError('invalid exact image token span')
                previous=start+count
                if previous>case['positions'][0]+1:raise ValueError('image must precede first scored row')
                path=str(image).encode()
                if any(c in path for c in (b'\x00',b'\t',b'\n')) or len(path)>4096:
                    raise ValueError('unsafe image path')
                height=(identity['n_vit_h']+2)//3;width=(identity['n_vit_w']+2)//3
                if count!=height*(width+1)+2:raise ValueError('wrong image block length')
                stream.write(struct.pack('<IIIII',start,count,height,width,len(path))+path)
                actual.append(dict(identity,patch_dtype='float32',patches_sha256=row['candidate_fp32_patches_sha256'],
                                   patches_diagnostic_bf16_sha256=row['candidate_patches_sha256']))
        item['candidate_image_identity']=actual
        item['image_manifest']=spans.name;item['image_manifest_sha256']=file_hash(spans)
        jobs.append(str((out/item['packet']).resolve())+'\t'+str((out/item['raw']).resolve())+'\t'+str(spans.resolve())+'\n')
    result['vision_comparison']='whole existing DS4 image pipeline versus DeepSeek Python reference; includes preprocessing/encoder/quantization/engine arithmetic'
    result['preprocessing_assessment_sha256']=file_hash(assessment)
    (out/'jobs.tsv').write_text(''.join(jobs))
    (out/'pack.json').write_text(json.dumps(result,ensure_ascii=False)+'\n')
    return result


def import_raw(raw,case,directory,provenance,rows_per_chunk=64,candidate_images=None):
    require_token_contract(case)
    if rows_per_chunk<1:raise ValueError('positive chunk size required')
    raw=Path(raw);n=len(case['positions']);vocab=case['vocab']
    if raw.stat().st_size!=16+n*(8+4*vocab):raise ValueError('incomplete raw candidate rows')
    with raw.open('rb') as f:
        if f.read(8)!=b'DS41OUT1' or struct.unpack('<II',f.read(8))!=(vocab,n):raise ValueError('candidate header mismatch')
    dtype=np.dtype([('position','<u4'),('target','<u4'),('logits','<f4',(vocab,))])
    data=np.memmap(raw,mode='r',dtype=dtype,offset=16,shape=(n,))
    if not np.array_equal(data['position'],case['positions']) or not np.array_equal(data['target'],case['target_ids']):
        raise ValueError('candidate exact-token/position mismatch')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    chunks=[]
    for begin in range(0,n,rows_per_chunk):
        end=min(begin+rows_per_chunk,n)
        rows=np.ascontiguousarray(data['logits'][begin:end])
        if not np.isfinite(rows).all():raise ValueError('nonfinite candidate logits')
        path=root/f'logits-{begin:06d}-{end:06d}.npy'
        np.save(path,rows,allow_pickle=False)
        chunks.append({'start_row':begin,'end_row':end,'file':path.name,'sha256':file_hash(path)})
    meta=dict(case)
    if case.get('image_identity'):
        if not candidate_images:raise ValueError('actual candidate image preprocessing identity required')
        meta['reference_image_identity']=case['image_identity']
        meta['image_identity']=candidate_images
        meta['vision_comparison']='whole image pipeline; includes existing antirez preprocessing and encoder arithmetic'
    # The candidate captures logits only. Native hidden vectors/routes and
    # generation settings describe the frozen reference, not this Q2 run.
    for key in ('hidden_file','hidden_sha256','hidden_chunks','hidden_dtype','hidden_positions','hidden_width',
                'routing','vision_features','native_weight_precision','native_runner','collector_identity',
                'head_projection_batch_rows','runtime','seconds'):
        meta.pop(key,None)
    generation={key:meta.pop(key) for key in ('sampling','stop_reason','actual_generated_tokens','generation_prefix_sha256')
                if key in meta}
    if generation:meta['reference_generation']=generation
    meta.update({'schema':'deepseek-v4.1-flash-1017.native-reference.v1','outputs_collected':True,
                 'native_outputs_collected':False,'artifact_kind':'DwarfStar GGUF candidate',
                 'comparison_mode':'teacher-forced logits on frozen native reference token history',
                 'candidate_provenance':provenance,'logit_dtype':'float32','logit_stage':'raw before sampling/penalties',
                 'logit_chunks':chunks,'prediction_convention':'input hidden at positions[j] predicts target_ids[j]'})
    (root/'case.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    create=sub.add_parser('pack');create.add_argument('--cases',type=Path,required=True);create.add_argument('--out',type=Path,required=True)
    create.add_argument('--group');create.add_argument('--max-cases',type=int,default=0)
    consume=sub.add_parser('import');consume.add_argument('--pack',type=Path,required=True);consume.add_argument('--out',type=Path,required=True)
    consume.add_argument('--provenance',type=Path,required=True)
    a=p.parse_args()
    if a.command=='pack':
        records=[json.loads(line) for line in a.cases.read_text().splitlines() if line.strip()]
        records=[case for case in records if a.group is None or case['group']==a.group]
        if a.max_cases:records=records[:a.max_cases]
        if not records:raise ValueError('empty packet job')
        result=pack(records,a.out)
        print(json.dumps({'cases':len(records),'required_context':result['required_context'],'jobs':str(a.out/'jobs.tsv')}))
    else:
        manifest=json.loads((a.pack/'pack.json').read_text());provenance=json.loads(a.provenance.read_text())
        if manifest.get('vision_comparison'):
            provenance=dict(provenance,vision_preprocessing_assessment_sha256=manifest['preprocessing_assessment_sha256'])
        for item in manifest['cases']:
            import_raw(a.pack/item['raw'],item['case'],a.out/item['case']['case_id'],provenance,
                       candidate_images=item.get('candidate_image_identity'))
        print(json.dumps({'imported_cases':len(manifest['cases']),'native_outputs_collected':False}))
