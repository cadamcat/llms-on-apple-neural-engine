"""Exercise the portable G1-W accounting and the QDQ probe verifier against disposable copies."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from g1w.evidence import BASE, derive

REPRO = ROOT / 'findings/coreai-qdq-multiply-scale/repro'


class G1WEvidence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        self.bundle = self.temporary / 'bundle'
        shutil.copytree(ROOT / BASE, self.bundle)

    def change(self, name, edit):
        path = self.bundle / name
        if name.endswith('.gz'):
            with gzip.open(path, 'rt') as stream:
                value = json.load(stream)
            edit(value)
            path.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
        else:
            value = json.loads(path.read_text())
            edit(value)
            path.write_text(json.dumps(value))
        # Re-sign the product so the semantic check, not the hash check, has to catch it.
        manifest = self.bundle / 'provenance.json'
        record = json.loads(manifest.read_text())
        record['products'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(record))

    def test_recorded_bundle_recomputes(self):
        data = derive(self.bundle)
        self.assertGreater(data['depth']['a8_over_w4a16']['both-128'], 1)
        self.assertLess(data['depth']['a8_over_w4a16']['both-1'], 1)
        self.assertEqual(data['old_probe'], {'equal': 16, 'unequal': 4, 'reverse_order': 64, 'explicit_q': 16})

    def test_unidentifiable_probe_output_fails_by_name(self):
        root = self.temporary / 'root'
        (root / 'results/fresh').mkdir(parents=True)
        smoke = json.loads((ROOT / 'results/fresh/smoke.json').read_text())
        case = next(r for r in smoke['runs'] if r['case_id'] == 'coreai-qdq-unequal')
        case['numerical']['comparisons'][0]['output_sha256'] = '0' * 64
        (root / 'results/fresh/smoke.json').write_text(json.dumps(smoke))
        with self.assertRaisesRegex(ValueError, 'g1w_old_probe_value.coreai-qdq-unequal'):
            derive(self.bundle, root)

    def test_unsigned_change_fails_the_product_hash(self):
        path = self.bundle / 'evidence.json'
        path.write_text(path.read_text().replace('"positions": 64', '"positions": 65'))
        with self.assertRaisesRegex(ValueError, 'g1w_product_hash.evidence.json'):
            derive(self.bundle)

    def test_changed_timing_fails_against_the_recorded_summary(self):
        def slow(value):
            calls = value['density']['sparse'][0]
            calls[:] = [2 * v for v in calls]
        self.change('timings.json.gz', slow)
        with self.assertRaisesRegex(ValueError, 'g1w_density_p50.sparse'):
            derive(self.bundle)

    def test_changed_summary_ratio_fails(self):
        self.change('evidence.json', lambda v: v['reported']['boundary']['a8_over_w4a16']
                    .__setitem__('both_1', 1.2))
        with self.assertRaisesRegex(ValueError, 'g1w_boundary_ratio.both-1'):
            derive(self.bundle)

    def test_dropped_e4b_call_fails_the_block_check(self):
        self.change('timings.json.gz', lambda v: v['e4b']['0']['native']['0'].pop())
        with self.assertRaisesRegex(ValueError, 'g1w_e4b_block.0.native.0'):
            derive(self.bundle)


class QDQProbeRecords(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='qdq repro ')
        self.addCleanup(temporary.cleanup)
        self.copy = Path(temporary.name) / 'repro'
        shutil.copytree(REPRO, self.copy, ignore=shutil.ignore_patterns('__pycache__'))

    def verify(self, *args):
        return subprocess.run([sys.executable, '-B', str(self.copy / 'verify.py'), *args],
                              capture_output=True, text=True)

    def test_recorded_outputs_pass(self):
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        arms = json.loads(result.stdout)['arms']
        self.assertEqual(arms['s2']['observed'], [8.0])
        self.assertTrue(arms['s2_clip']['matches_reference'])

    def test_changed_output_fails_by_name(self):
        path = self.copy / 'recorded/s4/control-0.raw'
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('s4.control-0.sha256', result.stderr)

    def test_corrected_runtime_is_reported_not_failed(self):
        outputs = self.copy.parent / 'new outputs'
        outputs.mkdir()
        for arm in ('s16', 's16_clip', 's8', 's8_clip', 's4', 's4_clip', 's2', 's2_clip'):
            for index in range(3):
                source = self.copy / 'recorded/s16' / f'control-{index}.raw'
                shutil.copy(source, outputs / f'{arm}-control-{index}.raw')
        result = self.verify('--outputs', str(outputs))
        self.assertEqual(result.returncode, 0, result.stderr)
        arms = json.loads(result.stdout)['arms']
        self.assertTrue(all(row['matches_reference'] for row in arms.values()))
        self.assertFalse(arms['s2']['matches_scale_substitution'])


if __name__ == '__main__':
    unittest.main()
