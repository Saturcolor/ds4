"""Protocol, coverage and identity checks; no GPU or model loading."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
from q2_packets import pack
from raw_score import raw_frames, score_panel
from reference_contract import MODEL_REPO, MODEL_REVISION, token_digest
from score import candidate_identity,existing_capture


def fixture(root, *, cases=1):
    reference = root/'reference'
    reference.mkdir()
    records = []
    logits = np.zeros((2, 129280), dtype='<f4')
    logits[0, 1] = 1
    logits[1, 2] = 1
    for number in range(cases):
        case_id = f'toy-{number}'
        directory = reference/case_id
        directory.mkdir()
        data = directory/'logits.npy'
        np.save(data, logits, allow_pickle=False)
        case = dict(case_id=case_id, group='short', model_repo=MODEL_REPO,
                    model_revision=MODEL_REVISION, vocab=129280, input_ids=[0, 1, 2],
                    target_ids=[1, 2], positions=[0, 1], tokenizer_sha256='0'*64,
                    thinking_mode='chat', outputs_collected=True,
                    native_outputs_collected=True, logit_dtype='float32',
                    logit_chunks=[dict(file='logits.npy', start_row=0, end_row=2,
                                       sha256=hashlib.sha256(data.read_bytes()).hexdigest())])
        case.update(input_ids_sha256=token_digest(case['input_ids']),
                    target_ids_sha256=token_digest(case['target_ids']))
        (directory/'case.json').write_text(json.dumps(case))
        records.append(case)
    ids = [case['case_id'] for case in records]
    (reference/'collection.json').write_text(json.dumps(dict(
        identity=dict(model_repo=MODEL_REPO, model_revision=MODEL_REVISION),
        completed_cases=ids, selected_case_ids=ids, collection_complete=True)))
    return reference, records, logits


def frames(path, case, logits):
    with path.open('wb') as stream:
        stream.write(b'DS41OUT1'+struct.pack('<II', 129280, len(case['positions'])))
        for position, target, row in zip(case['positions'], case['target_ids'], logits):
            stream.write(struct.pack('<II', position, target)+row.tobytes())


def gguf(path, revision=MODEL_REVISION, *, duplicate=False):
    entries = [('general.architecture', 'deepseek41'),
               ('general.source.revision', revision),
               ('general.source.url', 'https://huggingface.co/'+MODEL_REPO)]
    if duplicate:
        entries.append(entries[1])
    def string(value):
        value = value.encode()
        return struct.pack('<Q', len(value))+value
    with path.open('wb') as stream:
        stream.write(b'GGUF'+struct.pack('<IQQ', 3, 1046, len(entries)))
        for key, value in entries:
            stream.write(string(key)+struct.pack('<I', 8)+string(value))


class RawComparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference, self.cases, self.logits = fixture(self.root)
        self.packet = self.root/'pack'
        pack(self.cases, self.packet)
        self.raw = self.packet/'toy-0.raw'
        frames(self.raw, self.cases[0], self.logits)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_vocabulary_identical_distribution_and_analytic_nll(self):
        result = score_panel(self.reference, self.packet, self.root/'score', {'test': True})
        metrics = result['overall']['metrics']
        expected = math.log(129279+math.e)-1
        self.assertAlmostEqual(metrics['native_nll']['token_weighted_mean'], expected, places=9)
        self.assertEqual(metrics['forward_kl']['token_weighted_mean'], 0)
        self.assertEqual(metrics['top1_same']['token_weighted_mean'], 1)
        self.assertEqual(result['overall']['positions'], 2)
        self.assertTrue(result['complete_committed_collection'])

    def test_committed_capture_can_be_reanalyzed_after_relocation(self):
        panel=self.root/'saved/short100';attempt=panel/'attempt-1';packet=attempt/'pack'
        packet.mkdir(parents=True)
        (packet/'pack.json').write_text('{}')
        (attempt/'native-exit.json').write_text('{"actual_exit":0}')
        (panel/'done.json').write_text(json.dumps({'raw_root':'/old/mount/short100/attempt-1/pack','raw_relative':'attempt-1/pack'}))
        self.assertEqual(existing_capture(self.root/'saved','short100'),(attempt,packet))
        (attempt/'native-exit.json').write_text('{"actual_exit":1}')
        with self.assertRaisesRegex(ValueError,'did not exit successfully'):
            existing_capture(self.root/'saved','short100')

    def test_native_and_legacy_candidate_identities_match(self):
        stat={'bytes':365713686528,'mtime_ns':123}
        legacy=dict(model_repo=MODEL_REPO,model_revision=MODEL_REVISION,stat=stat)
        native=dict(model_revision=MODEL_REVISION,model_info={'architecture':'deepseek41','model_revision':MODEL_REVISION},model_stat=stat)
        self.assertEqual(candidate_identity(legacy),candidate_identity(native))
        with self.assertRaisesRegex(ValueError,'stat identity'):
            candidate_identity(dict(legacy,stat={}))

    def test_partial_file_cannot_be_a_completed_comparison(self):
        with self.raw.open('r+b') as stream:
            stream.truncate(16+8+4*129280)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            score_panel(self.reference, self.packet, self.root/'score', {})
        self.assertFalse((self.root/'score/SCORE.json').exists())

    def test_wrong_target_ids_are_rejected(self):
        bad = copy.deepcopy(self.cases[0])
        bad['target_ids'][0] = 99
        frames(self.raw, bad, self.logits)
        with self.assertRaisesRegex(ValueError, 'target IDs'):
            raw_frames(self.raw, self.cases[0])

    def test_nonfinite_tail_is_rejected(self):
        bad = self.logits.copy()
        bad[-1, -1] = np.nan
        frames(self.raw, self.cases[0], bad)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            score_panel(self.reference, self.packet, self.root/'score', {})
        self.assertFalse((self.root/'score/SCORE.json').exists())

    def test_subset_is_explicitly_not_complete_collection(self):
        other = self.root/'other'
        other.mkdir()
        reference, cases, _ = fixture(other, cases=2)
        packet = other/'pack'
        pack(cases[:1], packet)
        frames(packet/'toy-0.raw', cases[0], self.logits)
        result = score_panel(reference, packet, other/'score', {})
        self.assertTrue(result['complete_selected_cases'])
        self.assertFalse(result['complete_committed_collection'])
        self.assertEqual(result['reference_case_count'], 2)

    def test_wrong_reference_revision_is_rejected(self):
        bad = copy.deepcopy(self.cases[0])
        bad['model_revision'] = 'df42c109f1defefcbfcedbe7d905718a12266e40'
        with self.assertRaisesRegex(ValueError, 'exact DeepSeek'):
            pack([bad], self.root/'bad-packet')

    def test_wrong_input_history_is_rejected_even_with_matching_new_hash(self):
        manifest=json.loads((self.packet/'pack.json').read_text())
        path=self.packet/manifest['cases'][0]['packet']
        blob=bytearray(path.read_bytes())
        struct.pack_into('<I',blob,24,99)
        path.write_bytes(blob)
        manifest['cases'][0]['packet_sha256']=hashlib.sha256(blob).hexdigest()
        (self.packet/'pack.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'token history'):
            score_panel(self.reference,self.packet,self.root/'score',{})
        self.assertFalse((self.root/'score/SCORE.json').exists())

    def test_deep_context_and_position_kind_denominators(self):
        case=self.cases[0]
        case.update(context_bucket=4096,position_kinds=['prompt','continuation'])
        (self.reference/'toy-0/case.json').write_text(json.dumps(case))
        manifest=json.loads((self.packet/'pack.json').read_text())
        manifest['cases'][0]['case']=case
        (self.packet/'pack.json').write_text(json.dumps(manifest))
        result=score_panel(self.reference,self.packet,self.root/'score',{})
        self.assertEqual(result['groups']['context:4096']['positions'],2)
        self.assertEqual(result['groups']['position_kind:prompt']['positions'],1)
        self.assertEqual(result['groups']['position_kind:continuation']['positions'],1)


class NativeIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (HERE/'score_native').is_file():
            raise RuntimeError('Build the native helper before running the full test suite')

    def test_version_identifies_its_source(self):
        result=subprocess.run([str(HERE/'score_native'),'--version'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        parts=result.stdout.split()
        self.assertGreaterEqual(len(parts),3)
        self.assertEqual(parts[0],'score_native')
        self.assertEqual(parts[-1],'protocol1')

    def test_exact_revision_header_passes_without_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'header.gguf'
            gguf(path)
            result = subprocess.run([str(HERE/'score_native'), '--model-info', str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['model_revision'], MODEL_REVISION)

    def test_older_revision_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old-header.gguf'
            gguf(path, 'df42c109f1defefcbfcedbe7d905718a12266e40')
            result = subprocess.run([str(HERE/'score_native'), '--model-info', str(path)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('exact DeepSeek', result.stderr)

    def test_duplicate_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'duplicate.gguf'
            gguf(path, duplicate=True)
            result = subprocess.run([str(HERE/'score_native'), '--model-info', str(path)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('duplicate', result.stderr)


if __name__ == '__main__':
    unittest.main()
