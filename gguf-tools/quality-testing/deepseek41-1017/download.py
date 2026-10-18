#!/usr/bin/env python3
"""Download selected frozen 1017 reference panels."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def extract_stream(stream, destination, expected_root, expected_bytes, expected_files):
    """Extract only regular files under the declared panel root."""
    destination = Path(destination)
    names, total = set(), 0
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (not path.parts or path.is_absolute() or '..' in path.parts
                    or path.parts[0] != expected_root or '\\' in member.name):
                raise ValueError('Archive contains a path outside its panel')
            if not member.isfile() or member.name in names:
                raise ValueError('Archive contains a link, duplicate or non-file entry')
            names.add(member.name)
            total += member.size
            if total > expected_bytes or len(names) > expected_files:
                raise ValueError('Archive exceeds its published size or file count')
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as output:
                remaining = member.size
                while remaining:
                    block = source.read(min(1 << 20, remaining))
                    if not block:
                        raise ValueError('Archive payload is truncated')
                    output.write(block)
                    remaining -= len(block)
    if total != expected_bytes or len(names) != expected_files:
        raise ValueError('Archive does not contain the complete published panel')


def validate_panel(directory, expected):
    collection = directory/'collection.json'
    if hashlib.sha256(collection.read_bytes()).hexdigest() != expected['collection_sha256']:
        raise ValueError('Reference collection differs from the pinned release')
    metadata = read(collection)
    ids = metadata['completed_cases']
    if (not metadata.get('collection_complete') or len(ids) != expected['cases']
            or ids != metadata['selected_case_ids'] or len(ids) != len(set(ids))):
        raise ValueError('Reference collection is incomplete')
    if {p.parent.name for p in directory.glob('*/case.json')} != set(ids):
        raise ValueError('Reference case directories are incomplete')


def install_archive(archive, output, name, entry, revision):
    marker = output/name/'.download.json'
    identity = {'dataset_revision': revision, 'archive_sha256': entry['sha256']}
    if marker.is_file() and read(marker).get('archive_sha256') == entry['sha256']:
        if 'collection_sha256' in entry:
            validate_panel(output/name, entry)
        print(f'{name}: already installed')
        return
    if (output/name).exists():
        raise FileExistsError(f'{output/name} already exists without this release receipt; choose a fresh directory')
    staging = output/'.downloads'/f'{name}.extract-{os.getpid()}'
    staging.mkdir(parents=True, exist_ok=False)
    try:
        import zstandard
        with Path(archive).open('rb') as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as stream:
                extract_stream(stream, staging, name, entry['unpacked_bytes'], entry['files'])
        if 'collection_sha256' in entry:
            validate_panel(staging/name, entry)
        (staging/name/'.download.json').write_text(json.dumps(identity) + '\n')
        (staging/name).rename(output/name)
    finally:
        # Only this invocation's temporary extraction directory is removed.
        shutil.rmtree(staging)


def selected_panels(suite, panels):
    from score import SUITES
    chosen = panels or SUITES[suite]
    if len(chosen) != len(set(chosen)):
        raise ValueError('Duplicate panels')
    return chosen


def install_diagnostics(output, panels, package, download):
    head = package['shared_head']
    paths = {}
    for entry in [head, *package.get('diagnostic_files', [])]:
        path = Path(download(package['repo_id'], entry['path'], repo_type='dataset',
                             revision=package['revision'], local_dir=output))
        if path.stat().st_size != entry['bytes']:
            raise ValueError('Diagnostic file has the wrong size')
        paths[entry['path']] = path
    for name in panels:
        target = output/name/'native-head.safetensors'
        if not target.exists():
            os.link(paths[head['path']], target)
        elif not os.path.samefile(paths[head['path']], target):
            raise FileExistsError(f'{target} is not the downloaded shared head')


def main():
    from score import SUITES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--suite', choices=SUITES, default='short')
    selection.add_argument('--panel', choices=SUITES['all'], action='append')
    parser.add_argument('--diagnostics', action='store_true', help='Also download the original output head and raw numerical controls')
    args = parser.parse_args()
    package = read(HERE/'dataset.json')
    panels = selected_panels(args.suite, args.panel)
    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('references.json', 'native-controls.json'):
        target = output/name
        source = (HERE/name).read_bytes()
        if target.exists() and target.read_bytes() != source:
            raise ValueError(f'{target} belongs to a different reference release')
        if not target.exists():
            target.write_bytes(source)
    downloads = panels + (['vision-assets'] if 'vision32' in panels else [])
    from huggingface_hub import hf_hub_download
    for name in downloads:
        entry = package['archives'][name]
        marker = output/name/'.download.json'
        if marker.is_file() and read(marker).get('archive_sha256') == entry['sha256']:
            if 'collection_sha256' in entry:
                validate_panel(output/name, entry)
            print(f'{name}: already installed')
            continue
        if (output/name).exists():
            raise FileExistsError(f'{output/name} already exists; choose a fresh directory')
        required = entry['archive_bytes'] + entry['unpacked_bytes'] + (1 << 30)
        if shutil.disk_usage(output).free < required:
            raise ValueError(f'{name}: need {required/1e9:.1f} GB free for download and extraction')
        print(f'Downloading {name}: {entry["archive_bytes"]/1e9:.2f} GB compressed', flush=True)
        archive = Path(hf_hub_download(package['repo_id'], entry['path'], repo_type='dataset',
                                      revision=package['revision'], local_dir=output/'.downloads'))
        install_archive(archive, output, name, entry, package['revision'])
        # The installed panel is complete; its temporary transfer archive is no longer needed.
        archive.unlink()
        print(f'{name}: installed', flush=True)
    if args.diagnostics:
        install_diagnostics(output, panels, package, hf_hub_download)
    print(f'References ready: {output}', flush=True)


if __name__ == '__main__':
    main()
