#!/usr/bin/env python3
"""Checkpoint-pinned DeepSeek V4.1 Flash1017 deep scorer."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

HERE = Path(__file__).resolve().parent
REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
SUITES = {
    'short': ['short100'],
    'text': ['short100', 'expanded900-combined'],
    'vision': ['vision32'],
    'long': ['compatibility-long12', 'generation-long4', 'long-fixed8'],
    'deep': ['deep36'],
    'controls': ['generation-pilot', 'control-pilot-prefill'],
}
SUITES['all'] = ['short100', 'expanded900-combined', 'vision32',
                 'compatibility-long12', 'generation-long4', 'long-fixed8',
                 'deep36', 'generation-pilot', 'control-pilot-prefill']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def file_stat(path):
    info = Path(path).stat()
    return dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns, device=info.st_dev, inode=info.st_ino)


@contextmanager
def lock(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another scorer process still owns this run or GPU lock') from None
        yield stream


def load_panel(root, name, expected, limit=0):
    from reference_contract import require_token_contract
    panel = root/name
    collection = panel/'collection.json'
    if digest(collection) != expected['collection_sha256']:
        raise ValueError(f'{name}: frozen collection identity differs')
    state = read(collection)
    ids = state['completed_cases']
    if (state.get('collection_complete') is not True or len(ids) != len(set(ids))
            or ids != state['selected_case_ids'] or len(ids) != expected['cases']
            or state['identity']['model_revision'] != REVISION):
        raise ValueError(f'{name}: incomplete or wrong-checkpoint reference')
    cases = [read(panel/case_id/'case.json') for case_id in ids]
    for case in cases:
        require_token_contract(case)
    if sum(len(case['positions']) for case in cases) != expected['positions']:
        raise ValueError(f'{name}: reference position coverage differs')
    return cases[:limit] if limit else cases


def checked_info(binary, option, path):
    result = subprocess.run([str(binary), option, str(path)], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f'{option} {path.name}: '+(result.stderr.strip() or 'Native GGUF identity check failed'))
    info = json.loads(result.stdout)
    if info['model_revision'] != REVISION:
        raise ValueError('Exact1017 model and encoder required')
    return info


def existing_capture(root, name):
    """Resolve a legacy panel pack or this runner's committed capture."""
    panel=root/name
    packet=panel/'pack'
    if packet.is_dir():
        return panel,packet
    receipt=read(panel/'done.json')
    packet=(panel/receipt['raw_relative']).resolve() if 'raw_relative' in receipt else Path(receipt['raw_root'])
    if not packet.is_relative_to(panel):
        raise ValueError('Committed capture escapes its panel; mount the original run at the same path')
    if not (packet/'pack.json').is_file():
        raise ValueError('Committed capture is missing')
    attempt=packet.parent
    if read(attempt/'native-exit.json')['actual_exit']!=0:
        raise ValueError('Committed native capture did not exit successfully')
    return attempt,packet


def candidate_identity(provenance):
    repo=provenance.get('model_repo')
    checked_header=provenance.get('model_info',{})
    valid_repo=(repo=='deepseek-ai/DeepSeek-V4.1-Flash-1017' or
                (repo is None and checked_header.get('model_revision')==REVISION and checked_header.get('architecture')=='deepseek41'))
    if provenance.get('model_revision')!=REVISION or not valid_repo:
        raise ValueError('Candidate source is not the exact1017 checkpoint')
    stat=provenance.get('stat',provenance.get('model_stat'))
    if not isinstance(stat,dict) or not stat.get('bytes',stat.get('size')):
        raise ValueError('Candidate artifact stat identity is missing')
    return stat,provenance.get('imatrix_sha256')


def collect(binary, model, vision, attempt, context, inherited_fds):
    args = [str(binary), str(model), str(attempt/'pack/jobs.tsv'), str(context)]
    if vision:
        args.append(str(vision))
    env = dict(os.environ, DS4_LOCK_FILE=str(attempt/'engine.lock'))
    with (attempt/'native.log').open('x') as log:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, text=True, env=env,
                                   start_new_session=True, pass_fds=inherited_fds)
        try:
            start = Path(f'/proc/{process.pid}/stat').read_text().split()[21]
        except FileNotFoundError:
            start = None
        save(attempt/'process.json', dict(pid=process.pid, pgid=process.pid,
             start_ticks=start, argv=args, started_utc=datetime.now(timezone.utc).isoformat()))
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(('case=', 'image=')):
                    print('\r'+line.strip().ljust(90), end='', flush=True)
                else:
                    print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            code = process.wait()
            save(attempt/'native-exit.json', {'actual_exit': code, 'interrupted': True})
            raise
    print(flush=True)
    save(attempt/'native-exit.json', {'actual_exit': code, 'interrupted': False})
    if code != 0:
        raise RuntimeError(f'Native scorer exited {code}; attempt preserved')


def render_report(summary):
    lines = ['# DeepSeek V4.1 Flash1017 deep score', '',
             f"Checkpoint: `{REVISION}`. Mode: `{summary['mode']}`. Suite: `{summary['suite']}`.", '',
             '| Panel | Cases | Positions | Native NLL | Candidate NLL | ΔNLL | Forward KL | Top-token agreement |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, score in summary['panels'].items():
        row = score['overall']
        metric = lambda key: row['metrics'][key]['token_weighted_mean']
        lines.append(f"| {name} | {row['cases']} | {row['positions']} | {metric('native_nll'):.6f} | {metric('candidate_nll'):.6f} | {metric('nll_delta'):+.6f} | {metric('forward_kl'):.6f} | {100*metric('top1_same'):.2f}% |")
    lines += ['', 'These are full-vocabulary, temperature-1 comparisons on identical frozen histories. They include quantization and runtime arithmetic; vision includes the complete image pipeline. Top-token agreement is not task accuracy. Native numerical controls are reported separately and are never subtracted.', '',
              'Panels have overlapping histories; there is no combined quality percentage. Long-generation panels teacher-force saved native continuations rather than measure independent candidate task success. The four long native traces reached their 4,096-token caps.', '',
              f"Complete selected suite: `{summary['complete_selected_suite']}`. All nine full panels compared: `{summary['all_nine_panels_complete']}`."]
    for name, score in summary['panels'].items():
        buckets = [(key, value) for key, value in score['groups'].items() if key.startswith('context:')]
        if buckets:
            lines += ['', f'## {name}: context buckets', '', '| Tokens | Cases | Forward KL | Top-token agreement |', '| ---: | ---: | ---: | ---: |']
            for key, value in sorted(buckets, key=lambda row: int(row[0].split(':')[1])):
                m = value['metrics']
                lines.append(f"| {key.split(':')[1]} | {value['cases']} | {m['forward_kl']['token_weighted_mean']:.6f} | {100*m['top1_same']['token_weighted_mean']:.2f}% |")
        if 'aligner' in score:
            a = score['aligner']
            lines += ['', f"Vision aligner: {a['cases']} cases, {a['image_occurrences']} image occurrences; mean row cosine {a['overall']['mean_row_cosine']:.6f}, RMSE {a['overall']['rmse']:.6f}."]
    lines += ['', '## Native runtime controls (reference only)', '',
              '| Control | Positions | Forward KL | ΔNLL | Top-token agreement |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name,row in summary['native_numerical_controls']['controls'].items():
        m=row['metrics']
        lines.append(f"| {name} | {row['positions']} | {m['forward_kl']:.6f} | {m['nll_delta']:+.6f} | {100*m['top1_same']:.2f}% |")
    lines += ['', 'These are measured native execution controls on small fixed samples, not candidate scores or universal error floors. Per-position metrics, case-bootstrap intervals and worst cases are retained in the panel outputs. Native hidden states/router traces remain reference-only diagnostics; no uncollected candidate probes are claimed.', '']
    if summary['mode']=='compare-existing':
        lines += ['Supplied frame files were compared. Original collector exit records are retained in JSON; successful original inference is not inferred from these files.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--vision', type=Path)
    parser.add_argument('--images', type=Path, help='Image asset root; defaults to REFERENCES/vision-assets')
    parser.add_argument('--out', type=Path, required=True)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--suite', choices=SUITES, default='all')
    selection.add_argument('--panel',choices=SUITES['all'],action='append',help='Select one or more complete panels explicitly')
    parser.add_argument('--native', type=Path, default=HERE/'score_native')
    parser.add_argument('--probe', type=Path, default=HERE/'image_probe')
    parser.add_argument('--existing-runs', type=Path, help='CPU-only comparison of already collected PANEL/pack outputs')
    parser.add_argument('--limit', type=int, default=0, help='Explicit smoke subset per panel; never reported as full coverage')
    parser.add_argument('--resume', action='store_true', help='Reuse completed panels and successful collections; preserve interrupted attempts')
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError('Limit must be nonnegative')
    for key in ('references', 'out', 'native', 'probe', 'model', 'vision', 'images', 'existing_runs'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    definitions = read(HERE/'references.json')
    selected = args.panel or SUITES[args.suite]
    if len(selected)!=len(set(selected)):
        raise ValueError('Duplicate selected panels')
    if args.panel:args.suite='selected-panels'
    cases = {name: load_panel(args.references, name, definitions['panels'][name], args.limit) for name in selected}
    mode = 'compare-existing' if args.existing_runs else 'collect-and-compare'
    request = dict(mode=mode, suite=args.suite, limit=args.limit,
                   selected_panels=selected,
                   analysis_runtime={'python':platform.python_version(),'machine':platform.machine(),
                                     'packages':{name:importlib.metadata.version(name) for name in ('numpy','ml_dtypes','safetensors')}},
                   references_manifest_sha256=digest(HERE/'references.json'),
                   native_controls_sha256=digest(HERE/'native-controls.json'),
                   existing_runs=str(args.existing_runs) if args.existing_runs else None,
                   source_files={p.name: digest(p) for p in sorted(HERE.glob('*.py'))})
    if args.existing_runs:
        identities=[]
        request['existing_panel_identities']={}
        for name in selected:
            _,packet=existing_capture(args.existing_runs,name)
            provenance=read(packet/'candidate-provenance.json')
            identities.append(candidate_identity(provenance))
            request['existing_panel_identities'][name]={'pack_sha256':digest(packet/'pack.json'),
                                                       'provenance_sha256':digest(packet/'candidate-provenance.json')}
        if any(identity!=identities[0] for identity in identities):
            raise ValueError('Selected panels describe different candidate artifacts; compare them separately')
    if not args.existing_runs:
        if not args.model or ('vision32' in selected and not args.vision):
            raise ValueError('--model and, for vision/all suites, --vision are required')
        request.update(model_info=checked_info(args.native, '--model-info', args.model),
                       model_path=str(args.model), model_stat=file_stat(args.model),
                       native_sha256=digest(args.native),
                       native_version=subprocess.check_output([str(args.native), '--version'], text=True).strip())
        if 'vision32' in selected:
            request.update(vision_info=checked_info(args.native, '--vision-info', args.vision),
                           vision_path=str(args.vision), vision_stat=file_stat(args.vision),
                           probe_sha256=digest(args.probe))
    if args.out.exists() and not args.resume:
        raise FileExistsError('Output exists; choose a fresh directory or explicit --resume')
    if args.out.exists() and args.resume and not (args.out/'request.json').is_file():
        raise ValueError('Existing directory has no scorer ownership record; choose a fresh output directory')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    with lock(args.out/'run.lock') as run_lock:
        old = args.out/'request.json'
        if old.exists() and read(old) != request:
            raise ValueError('Resume identity differs; preserve the old run and choose a new output directory')
        save(old, request)
        if args.existing_runs:
            return run_panels(args, definitions, cases, request, run_lock, None)
        gpu_path = Path(os.environ.get('XDG_RUNTIME_DIR', tempfile.gettempdir()))/f'ds41-1017-deep-score-{os.getuid()}.lock'
        with lock(gpu_path) as gpu_lock:
            return run_panels(args, definitions, cases, request, run_lock, gpu_lock)


def run_panels(args, definitions, cases, request, run_lock, gpu_lock):
    from q2_packets import pack, pack_vision
    from raw_score import score_panel
    summary = dict(schema='deepseek-v4.1-flash-1017.deep-score-summary.v1',
                   model_revision=REVISION, mode=request['mode'], suite=args.suite,
                   analysis_runtime=request['analysis_runtime'],
                   panels={}, complete_selected_suite=False, all_nine_panels_complete=False,
                   native_numerical_controls=read(HERE/'native-controls.json'))
    for name, records in cases.items():
        panel = args.out/name
        panel.mkdir(exist_ok=True)
        done = panel/'done.json'
        if args.resume and done.exists():
            receipt = read(done)
            result_path = panel/receipt['score']
            if digest(result_path) != receipt['score_sha256']:
                raise ValueError('Saved score changed; refusing to silently reuse it')
            result=read(result_path)
            from raw_score import stat_identity
            if any(stat_identity(Path(receipt['raw_root'])/raw)!=identity
                   for raw,identity in result['raw_stat_identities'].items()):
                raise ValueError('Saved raw frame identities changed; refusing to reuse the score')
            summary['panels'][name] = result
            print(f'{name}: reused complete comparison', flush=True)
            continue
        attempt = panel/datetime.now(timezone.utc).strftime('attempt-%Y%m%dT%H%M%S.%fZ')
        attempt.mkdir()
        if args.existing_runs:
            source,packet=existing_capture(args.existing_runs,name)
            manifest = read(packet/'pack.json')
            indexed={item['case']['case_id']:item for item in manifest['cases']}
            if len(indexed)!=len(manifest['cases']):raise ValueError('Duplicate existing packet cases')
            chosen=[indexed[case['case_id']] for case in records]
            if [item['case'] for item in chosen] != records:
                raise ValueError('Existing output differs from selected frozen cases')
            manifest=dict(manifest,cases=chosen)
            save(attempt/'selected-pack.json',manifest)
            provenance_path = packet/'candidate-provenance.json'
            provenance = read(provenance_path)
            candidate_identity(provenance)
            provenance = dict(provenance, execution_mode='compare-existing; original collector exit is not inferred',
                              observed_original_exit_files={key: (source/key).read_text().strip() if (source/key).exists() else None
                                                            for key in ('scorer.exit', 'supervisor.exit', 'comparison.exit','native-exit.json')})
        else:
            provenance = dict(request, model_revision=REVISION,model_repo='deepseek-ai/DeepSeek-V4.1-Flash-1017')
            reusable=None
            if args.resume:
                for earlier in sorted(panel.glob('attempt-*'),reverse=True):
                    receipt=earlier/'collection.json'
                    if not receipt.exists():continue
                    saved=read(receipt)
                    if saved.get('provenance')!=provenance:continue
                    from raw_score import stat_identity
                    earlier_pack=earlier/'pack'
                    if any(stat_identity(earlier_pack/raw)!=identity for raw,identity in saved['raw_stat_identities'].items()):
                        raise ValueError('Previously collected raw frames changed')
                    if read(earlier/'native-exit.json')['actual_exit']!=0:continue
                    reusable=earlier_pack;break
            packet = reusable or attempt/'pack'
            if reusable:
                manifest=read(packet/'pack.json')
                print(f'{name}: reusing successful native collection; retrying comparison only',flush=True)
            elif name == 'vision32':
                from vision import capture, summarize
                probe_out = attempt/'preprocessing'
                images = args.images or args.references/definitions['images_path']
                capture(records, images, args.probe, probe_out)
                assessment = summarize(probe_out)
                pack_vision(records, packet, images, assessment)
            elif not reusable:
                pack(records, packet)
            manifest = read(packet/'pack.json')
            if not reusable:
                save(packet/'candidate-provenance.json', provenance)
                print(f'{name}: collecting {len(records)} cases, context {manifest["required_context"]}', flush=True)
                collect(args.native, args.model, args.vision if name == 'vision32' else None,
                        attempt, manifest['required_context'], (run_lock.fileno(), gpu_lock.fileno()))
                from raw_score import stat_identity
                save(attempt/'collection.json',dict(provenance=provenance,
                     raw_stat_identities={item['raw']:stat_identity(packet/item['raw']) for item in manifest['cases']}))
            if file_stat(args.model) != request['model_stat'] or digest(args.native) != request['native_sha256']:
                raise ValueError('Model or native executable changed during collection')
            if name == 'vision32' and file_stat(args.vision) != request['vision_stat']:
                raise ValueError('Vision encoder changed during collection')
        result = score_panel(args.references/name, packet, attempt/'score', provenance,selected_manifest=manifest)
        if name == 'vision32':
            aligner = attempt/'ALIGNER-SCORE.json'
            extra = ['--allow-subset'] if args.limit else []
            subprocess.run([sys.executable, str(HERE/'compare_aligner_features.py'),
                            '--reference', str(args.references/name), '--pack', str(packet),
                            '--out', str(aligner), '--manifest',str(attempt/'selected-pack.json') if args.existing_runs else str(packet/'pack.json'), *extra], check=True)
            result['aligner'] = read(aligner)
            save(attempt/'score/SCORE.json', result)
        if not result['complete_selected_cases']:
            raise ValueError('Selected case comparison is incomplete')
        relative = str((attempt/'score/SCORE.json').relative_to(panel))
        receipt=dict(score=relative,score_sha256=digest(attempt/'score/SCORE.json'),raw_root=str(packet))
        if packet.is_relative_to(panel):receipt['raw_relative']=str(packet.relative_to(panel))
        save(done,receipt)
        summary['panels'][name] = result
        save(args.out/'summary.partial.json', summary)
        print(f'{name}: complete comparison ({result["overall"]["positions"]} positions)', flush=True)
    summary['complete_selected_suite'] = True
    summary['all_nine_panels_complete'] = (set(cases) == set(SUITES['all']) and
                                         all(row['complete_committed_collection'] for row in summary['panels'].values()))
    save(args.out/'summary.json', summary)
    (args.out/'report.md').write_text(render_report(summary))
    print(f'Complete selected comparison: {args.out / "report.md"}', flush=True)
    return 0


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted; output attempts are preserved.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f'deep scorer: {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)
