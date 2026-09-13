"""Exercise portable G4 A accounting against corrupted disposable bundles."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from g3.evidence import read_rows
from g4a.disk import EVIDENCE, derive_disk
from g4a.evidence import BASE, derive

spec = importlib.util.spec_from_file_location('import_g4a', ROOT / 'results/historical/import_g4a.py')
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


class G4AEvidence(unittest.TestCase):
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
        # Re-sign the container to exercise semantic accounting, beyond its hash check.
        manifest = self.bundle / 'provenance.json'
        record = json.loads(manifest.read_text())
        record['products'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(record))

    def fails(self, name):
        return self.assertRaisesRegex(ValueError, name)

    def request(self, rows, phase, arm='ane', n=1024):
        return next(r for r in rows if r['phase'] == phase and r['arm'] == arm and r['input_N'] == n)

    def test_complete_bundle(self):
        data = self.data
        self.assertEqual(len(data['blocks']), 24)
        self.assertEqual(len(data['pairs']), 12)
        self.assertTrue(all(b['admitted'] for b in data['blocks']))
        for pair in data['pairs']:
            self.assertEqual(pair['ane']['tokens'], pair['gpu']['tokens'])

    def test_product_hash(self):
        (self.bundle / 'arms.json').write_text('[]')
        with self.fails('g4a_product_identity:arms.json'):
            derive(self.bundle, ROOT)

    def test_input_length(self):
        self.change('requests.jsonl.gz', lambda r: self.request(r, 'full')['result'].__setitem__('input_tokens', 999))
        with self.fails('request_input_length'):
            derive(self.bundle, ROOT)

    def test_forced_continuation(self):
        def edit(rows):
            item = self.request(rows, 'full')
            item['command']['forced_ids'] = [0] + item['command']['forced_ids'][1:]
            item['result']['generated_ids'] = item['command']['forced_ids']
        self.change('requests.jsonl.gz', edit)
        with self.fails('g4a_forced_continuation'):
            derive(self.bundle, ROOT)

    def test_timed_decode_query_matches_protocol(self):
        self.change('requests.jsonl.gz', lambda rows: self.request(rows, 'full')['command']
                    .__setitem__('decode_query', 1))
        with self.fails('g4a_decode_query'):
            derive(self.bundle, ROOT)

    def test_boundary_graph_capacity(self):
        def edit(rows):
            for event in self.request(rows, 'boundary')['result']['graphs']:
                event['graph'] = event['graph'].replace('_1280_', '_2304_')
        self.change('requests.jsonl.gz', edit)
        with self.fails('fixed_graph_capacity'):
            derive(self.bundle, ROOT)

    def test_tier_graph_inventory(self):
        self.change('asset-identity.json', lambda v: v['ane_tiers'][0]['tier']
                    ['kept_graphs_unchanged_before_serialization'].append('extend_33024_8'))
        with self.fails('g4a_tier_graphs:768'):
            derive(self.bundle, ROOT)

    def test_prefill_count(self):
        self.change('arms.json', lambda v: v[0]['requests']['prefill'].pop())
        with self.fails('g4a_block_work_count'):
            derive(self.bundle, ROOT)

    def test_recorded_speed(self):
        def edit(v):
            v['arms'][0]['phases']['decode']['speed']['decode_tokens_per_second'] *= 1.001
        self.change('summary-recorded.json', edit)
        with self.fails('g4a_recorded_speed'):
            derive(self.bundle, ROOT)

    def test_recorded_energy(self):
        def edit(v):
            v['arms'][0]['phases']['prefill_block']['energy']['counter_intervals']['2']['domains']['cpu']['estimate_J'] += 1
        self.change('summary-recorded.json', edit)
        with self.fails('g4a_recorded_energy'):
            derive(self.bundle, ROOT)

    def test_recorded_energy_normalization(self):
        self.change('summary-recorded.json', lambda v: v['arms'][0]['phases']['decode']['energy']
                    .__setitem__('component_J_per_token', 999))
        with self.fails('g4a_recorded_normalization'):
            derive(self.bundle, ROOT)

    def test_recorded_request_stats(self):
        self.change('summary-recorded.json', lambda v: v['arms'][0]['phases']['decode']['requests'][0]
                    ['stats'].__setitem__('decode_seconds', 999))
        with self.fails('g4a_recorded_request_stats'):
            derive(self.bundle, ROOT)

    def test_recorded_window(self):
        self.change('summary-recorded.json', lambda v: v['arms'][0]['phases']['decode']['energy']
                    ['counter_intervals']['2'].__setitem__('samples', 0))
        with self.fails('g4a_recorded_window'):
            derive(self.bundle, ROOT)

    def test_power_decode(self):
        def edit(rows):
            rows[100]['plist_fields'] = rows[100]['plist_fields'].replace('<key>cpu_energy</key>\n\t\t<integer>',
                                                                        '<key>cpu_energy</key>\n\t\t<integer>9')
        self.change('power.jsonl.gz', edit)
        with self.fails('g4a_power_decode'):
            derive(self.bundle, ROOT)

    def test_block_must_end_inside_power_stream(self):
        self.change('protocol.json', lambda v: v.__setitem__('energy_interior_margin_seconds', 10 ** 5))
        with self.fails('g4a_block_admission'):
            derive(self.bundle, ROOT)

    def test_every_tier_needs_a_direct_ane_admission(self):
        def edit(v):
            for check in v['flow_admissions'].values():
                if check['capacity'] == 4352:
                    check['placement']['short-q8']['target_PID_ANE_direct_request_log_rows'] = 0
        self.change('asset-identity.json', edit)
        with self.fails('g4a_ane_admission:4352'):
            derive(self.bundle, ROOT)

    def test_inputs_identity(self):
        self.change('protocol.json', lambda v: v['inputs_source'].__setitem__('sha256', '0' * 64))
        with self.fails('g4a_inputs_identity'):
            derive(self.bundle, ROOT)

    def test_model_source(self):
        self.change('asset-identity.json', lambda v: v['model_source']['config'].__setitem__('head_dim', 64))
        with self.fails('g4a_model_source'):
            derive(self.bundle, ROOT)

    def test_load_gate_order(self):
        def edit(v):
            v['load_gate'][0]['usable_bytes'] = 0
        self.change('disk.json', edit)
        with self.fails('g4a_load_gate_order'):
            derive(self.bundle, ROOT)

    def test_implied_rates_follow_measured_rates(self):
        work = self.data['work']
        for row in self.data['implied']:
            block = next(b for b in self.data['blocks'] if b['input_N'] == row['context'] and
                         b['arm'] == row['arm'] and b['mode'] == row['mode'])
            if row['mode'] == 'prefill':
                self.assertAlmostEqual(row['projection_TFLOP_per_second'], block['rate'] * work['flops_per_token'] / 1e12)
            else:
                self.assertAlmostEqual(row['kv_GB_per_second'],
                                       block['rate'] * (row['context'] + 127.5) * work['kv_bytes_per_token'] / 1e9)


class DiskObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g4a = derive()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        shutil.copytree(ROOT / EVIDENCE, self.repo / EVIDENCE)

    def change(self, edit):
        path = self.repo / EVIDENCE / 'observations.json'
        value = json.loads(path.read_text())
        edit(value)
        path.write_text(json.dumps(value))
        manifest = self.repo / EVIDENCE / 'provenance.json'
        record = json.loads(manifest.read_text())
        record['products']['observations.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(record))

    def test_recorded_observations(self):
        data = derive_disk(self.g4a, self.repo)
        self.assertEqual((data['held_files'], data['large_files'], data['reclaim_files']), (29, 26, 3))
        self.assertEqual(data['reclaim_signal'], 'KILL')
        self.assertTrue(all(-7.5 < x < -6.5 for x in data['ane_change_before_restart_GiB']))

    def test_product_hash(self):
        (self.repo / EVIDENCE / 'observations.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'disk_product_identity:observations.json'):
            derive_disk(self.g4a, self.repo)

    def test_only_deleted_compile_inputs_count(self):
        self.change(lambda v: v['listing_1335']['rows'][0].__setitem__('nlink', 1))
        with self.assertRaisesRegex(ValueError, 'disk_held_file_kind'):
            derive_disk(self.g4a, self.repo)

    def test_service_must_have_exited_before_release(self):
        self.change(lambda v: v['free_space'][2].__setitem__('service_1720_running', True))
        with self.assertRaisesRegex(ValueError, 'disk_service_state'):
            derive_disk(self.g4a, self.repo)

    def test_reclaim_log_accounts_for_its_files(self):
        self.change(lambda v: v['reclaim_log'].__setitem__(0, v['reclaim_log'][0].replace('held_bytes=21819799376', 'held_bytes=21819799377')))
        with self.assertRaisesRegex(ValueError, 'disk_reclaim_third_file'):
            derive_disk(self.g4a, self.repo)


class G4AClaims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from g4a.claims import quantities
        data = derive()
        cls.catalog = quantities(data, derive_disk(data), ROOT)
        cls.required = json.loads((ROOT / 'scripts/g4a/claims.json').read_text())

    def check(self, body, name='README.md'):
        from doc_claims import check_document
        body = re.sub(r'<!-- claim:(?!g4a\.)[^>]+-->.*?<!-- /claim -->', '', body, flags=re.DOTALL)
        return check_document(body, self.catalog, 'zh' if name.endswith('zh-CN.md') or '/zh/' in name else 'en',
                              required=self.required[name], name=name)

    def test_published_documents_match(self):
        for name in self.required:
            self.assertEqual(self.check((ROOT / name).read_text(), name), [], name)

    def test_changed_value_fails(self):
        body = (ROOT / 'README.md').read_text()
        marker = '<!-- claim:g4a.decode.1024.ane-rate@g4a-011 -->'
        self.assertIn(marker + '14.56 token/s', body)
        errors = self.check(body.replace(marker + '14.56 token/s', marker + '14.60 token/s'))
        self.assertTrue(any('g4a.decode.1024.ane-rate@g4a-011: expected' in e for e in errors), errors)

    def test_deleted_marker_fails(self):
        body = re.sub(r'<!-- claim:g4a\.decode\.1024\.ane-rate@g4a-011 -->(.*?)<!-- /claim -->', r'\1',
                      (ROOT / 'README.md').read_text())
        self.assertIn('missing claim g4a.decode.1024.ane-rate@g4a-011', ' '.join(self.check(body)))

    def test_component_check_ignores_other_families(self):
        from doc_claims import check_local_quotes
        from g2.evidence import derive as derive_g2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('README.md', 'README.zh-CN.md'):
                shutil.copy2(ROOT / name, root / name)
            shutil.copytree(ROOT / 'results/historical', root / 'results/historical')
            self.assertEqual(check_local_quotes(root, derive_g2()), [])


class G4AImportRedaction(unittest.TestCase):
    def test_personal_paths_are_refused(self):
        reader = importer.Reader(ROOT)
        for value in ('/Users/someone/file', '/private/var/folders/ab/cd/T/x'):
            with self.assertRaisesRegex(ValueError, 'personal_path'):
                reader.clean({'path': value})
        self.assertEqual(reader.clean(str(ROOT) + '/results/x'), 'results/x')

    def test_listing_redaction(self):
        row = ('g3-flow-h 53431 someone txt REG 1,13 777912320 0 36700836 '
               '/private/var/folders/xx/yy/T/payload-a49ceb.bin')
        parsed = importer.listing(row, 'nobody')
        self.assertEqual(parsed[0]['name'], '$TMPDIR/payload-a49ceb.bin')
        self.assertEqual(parsed[0]['user'], 'session user')
        with self.assertRaisesRegex(ValueError, 'listing_redaction'):
            importer.listing(row.replace('payload', 'someone'), 'someone')


if __name__ == '__main__':
    unittest.main()
