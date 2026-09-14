"""Exercise portable G6 accounting against corrupted disposable bundles."""
import gzip
import hashlib
import json
import plistlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from g3.evidence import read_rows
from g3.power import decode_power
from g6.evidence import BASE, derive


class G6Evidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = derive()

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
            path.write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in value) + '\n').encode(), mtime=0))
        else:
            path.write_text(json.dumps(value))
        manifest = self.bundle / 'provenance.json'
        record = json.loads(manifest.read_text())
        record['products'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(record))

    def fails(self, name):
        return self.assertRaisesRegex(ValueError, name)

    def test_complete_bundle(self):
        data = self.data
        self.assertEqual(len(data['widths']), 6)
        self.assertEqual(len(data['repeat_blocks']), 24)
        self.assertEqual([r['input_N'] for r in data['w4']], [1024, 4096])
        self.assertTrue(all(b['admitted'] for w in data['widths'] for b in (w['q8'], w['q4'])))
        self.assertTrue(all(not b['admitted'] and b['J_per_token']['ane'] == 0 for r in data['w4'] for b in r['w4'].values()))

    def test_product_hash(self):
        (self.bundle / 'sessions.json').write_text('{}')
        with self.fails('g6_product_identity:sessions.json'):
            derive(self.bundle, ROOT)

    def test_decode_query_matches_phase(self):
        def edit(rows):
            next(r for r in rows if r['phase'] == 'decode_q4')['command']['decode_query'] = 8
        self.change('requests.jsonl.gz', edit)
        with self.fails('g6_decode_query'):
            derive(self.bundle, ROOT)

    def test_w4_admission_placement(self):
        def edit(value):
            logs = value['admissions']['w4-c1280-reading_1024-switch']['placement_logs']
            next(iter(logs.values()))['ane_direct_request_success_rows'] = 3
        self.change('asset-identity.json', edit)
        with self.fails('g6_w4_admission_placement'):
            derive(self.bundle, ROOT)

    def test_fp16_admission_placement(self):
        def edit(value):
            logs = value['admissions']['q4-c4352-quality_short-switch']['placement_logs']
            next(iter(logs.values()))['ane_direct_request_success_rows'] = 0
        self.change('asset-identity.json', edit)
        with self.fails('g6_fp16_admission_placement'):
            derive(self.bundle, ROOT)

    def test_w4_block_with_ane_energy(self):
        def edit(rows):
            window = next(s for s in json.loads((self.bundle / 'sessions.json').read_text())['query_sessions']
                          if s['id'] == 'query-w4-1024')
            ids = set(window['requests']['decode_q8'])
            start = min(r['result']['request_start_ns'] for r in read_rows(self.bundle / 'requests.jsonl.gz')
                        if r['segment'] == 'query-w4-1024' and r['result']['id'] in ids)
            for row in rows:
                if row['capture'] == window['monitor'] and row['receipt']['monotonic_after_ns'] > start + 10e9:
                    fields = plistlib.loads(row['plist_fields'].encode())
                    processor = fields['processor']
                    watts = 50.0
                    processor['ane_power'] = watts
                    processor['ane_energy'] = round(watts * fields['elapsed_ns'] / 1e9)
                    processor['combined_power'] += watts
                    row['plist_fields'] = plistlib.dumps(fields, sort_keys=True).decode()
                    # Keep the decoded record consistent with the edited frame, so only the placement rule can object.
                    row['recorded_decoded'] = decode_power(row['plist_fields'].encode() + b'\0', row['receipt'])
                    assert not row['recorded_decoded']['issues']
                    break
        self.change('power.jsonl.gz', edit)
        with self.fails('g6_w4_placement:query-w4-1024:decode_q8'):
            derive(self.bundle, ROOT)

    def test_recorded_speed(self):
        def edit(value):
            entry = next(e for e in value['segments'] if e['id'] == 'query-fp16-2048')
            entry['phases']['decode_q4']['speed']['decode_tokens_per_second'] *= 1.001
        self.change('summary-recorded.json', edit)
        with self.fails('g6_recorded_speed'):
            derive(self.bundle, ROOT)

    def test_repeat_block_count(self):
        def edit(value):
            value['repeat_arms'][0]['requests']['prefill'].pop()
        self.change('sessions.json', edit)
        with self.fails('g6_repeat_work_count'):
            derive(self.bundle, ROOT)

    def test_idle_power(self):
        def edit(value):
            next(e for e in value['segments'] if e['id'] == 'idle-end')['power']['mean_W']['cpu'] += 0.5
        self.change('summary-recorded.json', edit)
        with self.fails('g6_recorded_idle_power'):
            derive(self.bundle, ROOT)

    def test_tier_graph_inventory(self):
        self.change('asset-identity.json', lambda v: v['tiers'][0]['tier']
                    ['kept_graphs_unchanged_before_serialization'].append('extend_768_1'))
        with self.fails('g6_tier_graphs:fp16:768'):
            derive(self.bundle, ROOT)


if __name__ == '__main__':
    unittest.main()
