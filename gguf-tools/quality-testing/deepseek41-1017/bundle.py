#!/usr/bin/env python3
"""Assemble the saved1017 references without putting large payloads in Git."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing nine-panel native-reference directory')
    parser.add_argument('--images', type=Path, required=True, help='Existing evaluated image asset root')
    parser.add_argument('--controls', type=Path, required=True, help='Saved native-controls-v1.tar.gz')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.source, args.images = args.source.resolve(), args.images.resolve()
    manifest = json.loads((HERE/'references.json').read_text())
    if args.out.exists():
        raise FileExistsError('Choose a fresh bundle directory; existing data is never replaced')
    for name, panel in manifest['panels'].items():
        if digest(args.source/name/'collection.json') != panel['collection_sha256']:
            raise ValueError(f'{name}: frozen collection identity differs')
    if (args.controls.stat().st_size != 157106223 or
            digest(args.controls) != '3983e884a6483b1d9fcf287b5382d500c0512f464aaaa835a52e694949a081d5'):
        raise ValueError('Wrong saved native numerical-control archive')
    args.out.mkdir(parents=True)
    inodes, files, logical = {}, 0, 0

    def copy_file(source, destination):
        nonlocal files, logical
        if source.is_symlink() or not source.is_file():
            raise ValueError('Reference payload must be a regular file')
        destination.parent.mkdir(parents=True, exist_ok=True)
        info = source.stat()
        key = info.st_dev, info.st_ino
        existing = inodes.get(key, source)
        try:
            os.link(existing, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(source, destination)
        inodes[key] = destination
        if destination.stat().st_size != info.st_size:
            raise ValueError('Reference copy length differs')
        files += 1
        logical += info.st_size

    for name in manifest['panels']:
        source = args.source/name
        for directory, dirs, names in os.walk(source, followlinks=False):
            if any((Path(directory)/child).is_symlink() for child in dirs):
                raise ValueError('Reference directories may not escape through symlinks')
            for filename in names:
                path = Path(directory)/filename
                copy_file(path, args.out/name/path.relative_to(source))
        print(f'{name}: reference files staged', flush=True)
    image_files = {}
    for path in (args.source/'vision32').glob('*/case.json'):
        for image in json.loads(path.read_text())['image_identity']:
            previous = image_files.setdefault(image['file'], image['raw_sha256'])
            if previous != image['raw_sha256']:
                raise ValueError('Conflicting frozen image identities')
    for name, expected in image_files.items():
        path = (args.images/name).resolve()
        if not path.is_relative_to(args.images) or digest(path) != expected:
            raise ValueError('Frozen evaluation image differs')
        copy_file(path, args.out/manifest['images_path']/name)
    copy_file(args.controls, args.out/'native-controls-v1.tar.gz')
    shutil.copyfile(HERE/'references.json', args.out/'references.json')
    shutil.copyfile(HERE/'native-controls.json', args.out/'native-controls.json')
    receipt = dict(model_revision=manifest['model_revision'], files=files,
                   logical_bytes=logical, images=len(image_files),
                   case_occurrences=sum(x['cases'] for x in manifest['panels'].values()),
                   scored_positions=sum(x['positions'] for x in manifest['panels'].values()),
                   reference_payload_policy='Hardlinks on the same filesystem; copies across filesystems. No source files or permissions changed.',
                   model_weight_files_scanned=False)
    (args.out/'BUNDLE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
