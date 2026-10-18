import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from download import extract_stream, selected_panels, validate_panel, install_archive, install_diagnostics


def archive(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as writer:
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                info.linkname = '/tmp/elsewhere'
            writer.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    output.seek(0)
    return output


class DownloadTests(unittest.TestCase):
    def test_extract_preserves_regular_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract_stream(archive([('short100/case_000/logits.npy', b'abc', tarfile.REGTYPE)]), root, 'short100', 3, 1)
            self.assertEqual((root/'short100/case_000/logits.npy').read_bytes(), b'abc')

    def test_unsafe_paths_links_duplicates_and_wrong_root_are_rejected(self):
        for names in [('../outside',), ('/absolute',), ('vision32/wrong',), ('short100/../outside',), ('short100/duplicate', 'short100/duplicate')]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    extract_stream(archive([(name, b'a', tarfile.REGTYPE) for name in names]), Path(directory), 'short100', len(names), len(names))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            extract_stream(archive([('short100/link', b'', tarfile.SYMTYPE)]), Path(directory), 'short100', 0, 1)

    def test_missing_and_oversized_payloads_are_rejected(self):
        for size, count in [(4, 1), (2, 1), (3, 2), (3, 0)]:
            with self.subTest(size=size, count=count), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    extract_stream(archive([('short100/data', b'abc', tarfile.REGTYPE)]), Path(directory), 'short100', size, count)

    def test_collection_identity_and_case_coverage_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = json.dumps({'collection_complete': True, 'completed_cases': ['a'], 'selected_case_ids': ['a']}).encode()
            (root/'collection.json').write_bytes(data)
            expected = {'collection_sha256': hashlib.sha256(data).hexdigest(), 'cases': 1}
            with self.assertRaises(ValueError):
                validate_panel(root, expected)
            (root/'a').mkdir()
            (root/'a/case.json').write_text('{}')
            validate_panel(root, expected)
            (root/'collection.json').write_text('{}')
            with self.assertRaises(ValueError):
                validate_panel(root, expected)

    def test_diagnostics_download_once_and_share_the_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'short100').mkdir()
            package = {'repo_id': 'test/data', 'revision': 'pinned',
                       'shared_head': {'path': 'diagnostics/head', 'bytes': 3},
                       'diagnostic_files': [{'path': 'diagnostics/control', 'bytes': 2}]}
            calls = []
            def download(repo, path, **options):
                calls.append((repo, path, options))
                target = root/path
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    target.write_bytes(b'abc' if path.endswith('head') else b'ab')
                return target
            install_diagnostics(root, ['short100'], package, download)
            self.assertEqual(len(calls), 2)
            self.assertTrue((root/'short100/native-head.safetensors').samefile(root/'diagnostics/head'))
            install_diagnostics(root, ['short100'], package, download)
            package['shared_head']['bytes'] = 4
            with self.assertRaises(ValueError):
                install_diagnostics(root, ['short100'], package, download)

    def test_unchanged_archive_is_reused_across_metadata_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = root/'vision-assets'
            panel.mkdir()
            (panel/'.download.json').write_text(json.dumps({'dataset_revision': 'old', 'archive_sha256': 'same'}))
            install_archive(root/'not-needed.tar.zst', root, 'vision-assets', {'sha256': 'same'}, 'new')
            self.assertEqual(json.loads((panel/'.download.json').read_text())['dataset_revision'], 'old')

    def test_default_short_and_optional_suite_selection(self):
        self.assertEqual(selected_panels('short', None), ['short100'])
        self.assertEqual(selected_panels('vision', None), ['vision32'])
        self.assertEqual(len(selected_panels('all', None)), 9)
        self.assertEqual(selected_panels('short', ['deep36']), ['deep36'])
        with self.assertRaises(ValueError):
            selected_panels('short', ['short100', 'short100'])


if __name__ == '__main__':
    unittest.main()
