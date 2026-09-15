"""Exercise portable G3 accounting against corrupted disposable bundles."""
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from g3.evidence import BASE, derive, implied, read_rows
from g3.power import energy_window
from g3.claims import check as check_quotes


class G3Evidence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.bundle = Path(temporary.name) / 'bundle'
        shutil.copytree(ROOT / BASE, self.bundle)

    def change(self, name, edit):
        path = self.bundle / name
        value = read_rows(path) if name.endswith('.gz') else json.loads(path.read_text())
        edit(value)
        if name.endswith('.gz'):
            path.write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in value)+'\n').encode(), mtime=0))
        else:
            path.write_text(json.dumps(value))

    def test_complete_bundle_and_equal_work(self):
        data = derive(self.bundle)
        self.assertEqual(len(data['pairs']), len(data['protocol']['lengths']) * 2)
        for pair in data['pairs']:
            self.assertEqual(pair['ane']['tokens'], pair['gpu']['tokens'])
            for arm in ('ane', 'gpu'):
                b = pair[arm]
                self.assertAlmostEqual(b['mean_W']['components']/b['rate'], b['J_per_token']['components'])

    def test_token_count_cannot_be_changed(self):
        self.change('requests.jsonl.gz', lambda r: r[0]['result'].__setitem__('input_tokens', 999))
        with self.assertRaisesRegex(ValueError, 'request_input_length'):
            derive(self.bundle)

    def test_weight_bytes_must_match_the_tensor_inventory(self):
        self.change('model-structure.json', lambda v: v.__setitem__('source_total_bytes', 8_000_000_000))
        with self.assertRaisesRegex(ValueError, 'g3_weight_bytes'):
            derive(self.bundle)

    def test_implied_rates_follow_the_measured_rates(self):
        data = derive(self.bundle)
        work = data['work']
        self.assertEqual(work['weight_bytes'], 2 * work['total_parameters'])
        for row in data['implied']:
            pair = next(p for p in data['pairs'] if p['mode'] == row['mode'] and p['context'] == row['context'])
            rate = pair[row['arm']]['rate']
            if row['mode'] == 'prefill':
                self.assertAlmostEqual(row['tera_flops_per_second'], rate * work['flops_per_token'] / 1e12)
            else:
                self.assertAlmostEqual(row['giga_bytes_per_second'], rate * row['step_bytes'] / 1e9)

    def test_decode_bandwidth_includes_each_KV_step(self):
        # Three forwards in six seconds: (100+20)+(100+30)+(100+40) = 390 bytes.
        work = {'weight_bytes': 100, 'kv_bytes_per_token': 10}
        pair = {'mode': 'decode', 'context': 2,
                'ane': {'tokens': 3, 'rate': 0.5}, 'gpu': {'tokens': 3, 'rate': 0.5}}
        for row in implied(work, [pair]):
            self.assertAlmostEqual(row['giga_bytes_per_second'] * 1e9, 65)
            self.assertEqual(row['block_bytes'], 390)

    def test_missing_power_sample_is_named(self):
        self.change('power.jsonl.gz', lambda r: r.pop(50))
        with self.assertRaisesRegex(ValueError, 'g3_power_sample_inventory'):
            derive(self.bundle)

    def test_changed_decoded_power_is_rejected_from_plist(self):
        self.change('power.jsonl.gz', lambda r: r[50]['recorded_decoded']['domains']['gpu'].__setitem__('energy_mj', 0))
        with self.assertRaisesRegex(ValueError, 'g3_power_decode'):
            derive(self.bundle)

    def test_capture_reference_cannot_be_swapped(self):
        self.change('blocks.json', lambda r: r[-1].__setitem__('capture', 'r6'))
        with self.assertRaisesRegex(ValueError, 'g3_block_identity'):
            derive(self.bundle)

    def test_block_work_cannot_be_shortened(self):
        self.change('blocks.json', lambda r: r[0].__setitem__('requested_count', 45))
        with self.assertRaisesRegex(ValueError, 'g3_block_work_count'):
            derive(self.bundle)

    def test_original_short_pair_cannot_be_admitted(self):
        self.change('protocol.json', lambda r: r.__setitem__('preferred_500_prefill_run', 'r5'))
        with self.assertRaisesRegex(ValueError, 'g3_preferred_energy_admission'):
            derive(self.bundle)

    def test_stage_clock_is_checked(self):
        self.change('requests.jsonl.gz', lambda r: r[0]['result'].__setitem__('first_token_ns', 1))
        with self.assertRaisesRegex(ValueError, 'phase_clock'):
            derive(self.bundle)

    def test_known_constant_power_integral(self):
        rows = [{'receipt': {'monotonic_after_ns': (i+1)*10**9}, 'decoded': {
            'elapsed_ns': 10**9, 'source_timestamp_bin_monotonic_ns': [i*10**9, i*10**9],
            'issues': [], 'domains': {k: {'energy_mj': v} for k,v in
                                     [('cpu', 1000), ('gpu', 2000), ('ane', 3000)]}}}
                for i in range(20)]
        result = energy_window(rows, 5e9, 10e9, 0)
        self.assertTrue(result['integrity_passed'])
        self.assertAlmostEqual(result['domains']['components']['estimate_J'], 30)


# The G3 finding page keeps its G3 references after the README moved to G4 A.
DOCUMENT = 'findings/qwen3-4b-prefill-decode/README.md'


class G3DocumentClaims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=derive()

    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        registry=json.loads((ROOT/'scripts/g3/claims.json').read_text())
        for name in [*registry, 'scripts/g3/claims.json', 'results/fresh/throughput.json']:
            target=self.root/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)

    def first_reference(self):
        """The registry decides which reference these tests tamper with."""
        registry = json.loads((self.root / 'scripts/g3/claims.json').read_text())[DOCUMENT]
        location = sorted(registry)[0]
        return location, registry[location], f'<!-- claim:{registry[location]}@{location} -->'

    def test_one_wrong_reference_fails_with_other_references_correct(self):
        p = self.root / DOCUMENT
        body = p.read_text()
        location, claim, marker = self.first_reference()
        start = body.index(marker) + len(marker)
        end = body.index('<!-- /claim -->', start)
        p.write_text(body[:start] + '99 token/s' + body[end:])
        errors = check_quotes(self.root, self.data)
        self.assertTrue(any(f'{claim}@{location}: expected' in e for e in errors), errors)

    def test_deleted_reference_fails_by_location(self):
        p = self.root / DOCUMENT
        location, claim, marker = self.first_reference()
        p.write_text(p.read_text().replace(marker, '', 1))
        errors = check_quotes(self.root, self.data)
        self.assertIn(f'{DOCUMENT}: missing claim {claim}@{location}', errors)

    def test_synthetic_reference_reads_its_source(self):
        p = self.root / 'results/fresh/throughput.json'
        source = json.loads(p.read_text())
        for run in source['runs']:
            if run['case_id'] == 'coreai-fp16-128' and run['p50_ms']:
                run['p50_ms'] *= 2
        p.write_text(json.dumps(source))
        errors = check_quotes(self.root, self.data)
        self.assertTrue(any('g3.synthetic-fp16@' in e and ': expected' in e for e in errors), errors)

if __name__ == '__main__':
    unittest.main()
