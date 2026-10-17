"""Compare complete native references with exact-token candidate binary frames."""
import json
import hashlib
from pathlib import Path
import struct

import numpy as np

from compare_archives import Case
from reference_contract import MODEL_REVISION, distribution_metrics, require_matching_case
from score_collections import summarize


def raw_frames(path, case):
    path = Path(path)
    count, vocab = len(case['positions']), case['vocab']
    if vocab != 129280 or path.stat().st_size != 16 + count * (8 + 4 * vocab):
        raise ValueError('incomplete or wrong-vocabulary candidate frames')
    with path.open('rb') as stream:
        if stream.read(8) != b'DS41OUT1' or struct.unpack('<II', stream.read(8)) != (vocab, count):
            raise ValueError('candidate frame header differs')
    dtype = np.dtype([('position', '<u4'), ('target', '<u4'), ('logits', '<f4', (vocab,))])
    data = np.memmap(path, mode='r', dtype=dtype, offset=16, shape=(count,))
    if not np.array_equal(data['position'], case['positions']) or not np.array_equal(data['target'], case['target_ids']):
        raise ValueError('candidate prediction positions or target IDs differ')
    return data


def stat_identity(path):
    info = Path(path).stat()
    return dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns, device=info.st_dev, inode=info.st_ino)


def validate_packet(root, item):
    if Path(item['packet']).name != item['packet']:
        raise ValueError('unsafe input packet path')
    blob=(root/item['packet']).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=item['packet_sha256'] or blob[:8]!=b'DS411017':
        raise ValueError('input packet integrity differs')
    case=item['case'];n=len(case['input_ids']);m=len(case['positions'])
    if len(blob)!=24+4*(n+2*m) or struct.unpack_from('<IIII',blob,8)!=(129280,n,m,0):
        raise ValueError('input packet dimensions differ')
    values=np.frombuffer(blob,dtype='<u4',offset=24)
    if (not np.array_equal(values[:n],case['input_ids']) or
            not np.array_equal(values[n:n+m],case['positions']) or
            not np.array_equal(values[n+m:],case['target_ids'])):
        raise ValueError('input packet token history differs from frozen metadata')


def score_panel(reference, pack, output, provenance, *, selected_manifest=None):
    reference, pack, output = Path(reference), Path(pack), Path(output)
    collection = json.loads((reference/'collection.json').read_text())
    manifest = selected_manifest or json.loads((pack/'pack.json').read_text())
    if not collection.get('collection_complete') or manifest['source_revision'] != MODEL_REVISION:
        raise ValueError('complete exact1017 native reference required')
    committed = collection['completed_cases']
    if len(committed) != len(set(committed)) or set(committed) != set(collection['selected_case_ids']):
        raise ValueError('invalid native collection coverage')
    ids = [item['case']['case_id'] for item in manifest['cases']]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(committed):
        raise ValueError('missing, duplicate or unknown selected cases')
    whole_vision = bool(manifest.get('vision_comparison'))
    output.mkdir(parents=True, exist_ok=False)
    records, raw_stats = [], {}
    for number, item in enumerate(manifest['cases'], 1):
        case = item['case']
        case_id = case['case_id']
        if Path(case_id).name != case_id or Path(item['raw']).name != item['raw']:
            raise ValueError('unsafe case or raw output path')
        validate_packet(pack,item)
        native = Case(reference/case_id)
        if not native.meta.get('native_outputs_collected') or case != native.meta:
            raise ValueError('packet differs from the frozen native case')
        candidate = dict(case, native_outputs_collected=False, candidate_provenance=provenance)
        if whole_vision:
            candidate.update(reference_image_identity=case['image_identity'],
                             image_identity=item['candidate_image_identity'],
                             vision_comparison=manifest['vision_comparison'])
        require_matching_case(native.meta, candidate, vision_whole_pipeline=whole_vision)
        raw = pack/item['raw']
        before = stat_identity(raw)
        frames = raw_frames(raw, case)
        arrays = {}
        for start in range(0, len(case['positions']), 32):
            end = min(start+32, len(case['positions']))
            metrics = distribution_metrics(native.rows(start, end), frames['logits'][start:end],
                                           np.asarray(case['target_ids'][start:end]))
            for key, values in metrics.items():
                arrays.setdefault(key, []).append(values)
        arrays = {key: np.concatenate(values).astype(np.float64) for key, values in arrays.items()}
        if stat_identity(raw) != before:
            raise ValueError('candidate output changed during comparison')
        raw_stats[item['raw']] = before
        np.savez(output/(case_id+'.per-position.npz'), **arrays)
        small = {'case_id': case_id, 'positions': len(case['positions']),
                 'metrics': {key: float(values.mean()) for key, values in arrays.items()}}
        (output/(case_id+'.json')).write_text(json.dumps(small, indent=2)+'\n')
        records.append(dict(case_id=case_id, meta=native.meta, arrays=arrays))
        print(f'\rComparing {number}/{len(ids)} cases', end='', flush=True)
    print(flush=True)
    groups = {}
    for record in records:
        meta = record['meta']
        keys = ['group:'+meta['group']]
        if 'context_bucket' in meta:
            keys.append('context:'+str(meta['context_bucket']))
        for key in keys:
            groups.setdefault(key, []).append(record)
        kinds = np.asarray(meta.get('position_kinds', ['continuation']*len(meta['positions'])))
        if kinds.shape != (len(meta['positions']),):
            raise ValueError('position-kind coverage differs')
        for kind in sorted(set(kinds)):
            mask = kinds == kind
            groups.setdefault('position_kind:'+kind, []).append(dict(
                record, arrays={key: value[mask] for key, value in record['arrays'].items()}))
    result = {'schema': 'deepseek-v4.1-flash-1017.deep-score.v1',
              'model_revision': MODEL_REVISION, 'reference_identity': collection['identity'],
              'complete_selected_cases': True, 'complete_committed_collection': ids == committed,
              'reference_case_count': len(committed), 'selected_case_count': len(ids),
              'candidate_provenance': provenance, 'raw_stat_identities': raw_stats,
              'bootstrap': '1000 case-resampled replicates; token-weighted mean; seed1017',
              'interpretation': 'Quantization and runtime divergence on fixed native histories; not task accuracy or a universal pass threshold.',
              'overall': summarize(records),
              'groups': {key: summarize(values) for key, values in groups.items()},
              'worst_forward_kl_cases': sorted([
                  {'case_id': row['case_id'], 'mean_forward_kl': float(row['arrays']['forward_kl'].mean())}
                  for row in records], key=lambda row: row['mean_forward_kl'], reverse=True)[:20]}
    if whole_vision:
        result['interpretation'] = 'Whole image pipeline includes preprocessing, encoder, quantization and runtime arithmetic; not isolated language quantization error.'
    (output/'SCORE.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
