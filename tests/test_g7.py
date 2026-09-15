"""Exercise portable G7 accounting against corrupted disposable bundles."""
import gzip
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
from g7.evidence import BASE, derive


class G7Evidence(unittest.TestCase):
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

    def fails(self, name):
        return self.assertRaisesRegex(ValueError, name)

    def test_complete_bundle(self):
        data = self.data
        self.assertEqual(len(data['blocks']), 78)
        self.assertEqual(sorted(k for k, v in data['admission'].items() if not v and 'mlp' not in k), ['synthetic-d128-coreml-a8w4_int8_lut'])
        self.assertTrue(data['identical_original_output']['mlp-n1024-w8a8_same_codes'])
        self.assertFalse(data['identical_original_output']['mlp-n1024-a8w4_int8_lut'])

    def test_numeric_screen(self):
        def edit(value):
            value['gate-n1024-coreai-a8w4_int8_lut']['admission']['numeric']['comparisons']['benchmark']['ordinary']['relative_l2'] = 0.06
        self.change('configs.json', edit)
        with self.fails('g7_numeric_rule:gate-n1024-coreai-a8w4_int8_lut'):
            derive(self.bundle, ROOT)

    def test_zero_control(self):
        def edit(value):
            value[0]['admission']['control_output_sha256']['zero'] = value[0]['admission']['control_output_sha256']['original']
        self.change('blocks.json', edit)
        with self.fails('g7_numeric_rule:r0-gate-n64-coreai-a8w4_int8_lut'):
            derive(self.bundle, ROOT)

    def test_missing_ane_request(self):
        def edit(value):
            block = next(b for b in value if b['id'] == 'r1-mlp-n1024-coreai-w4a16')
            block['admission']['placement']['requests_per_control'][2] = 0
        self.change('blocks.json', edit)
        with self.fails('g7_placement_rule:r1-mlp-n1024-coreai-w4a16'):
            derive(self.bundle, ROOT)

    def test_coreml_operation_on_cpu(self):
        def edit(value):
            plan = value['gate-n1024-coreml-w4a16']['admission']['placement']['compute_plan']
            next(p for p in plan if p['operator'] == 'ios18.conv')['preferred'] = 'MLCPUComputeDevice'
        self.change('configs.json', edit)
        with self.fails('g7_placement_rule:gate-n1024-coreml-w4a16'):
            derive(self.bundle, ROOT)

    def test_coreml_other_operation_on_cpu(self):
        def edit(value):
            plan = value['gate-n64-coreml-a8w4_int8_lut']['admission']['placement']['compute_plan']
            next(p for p in plan if p['operator'] == 'ios18.quantize')['preferred'] = 'MLCPUComputeDevice'
        self.change('configs.json', edit)
        with self.fails('g7_coreml_operations_not_all_ane:gate-n64-coreml-a8w4_int8_lut'):
            derive(self.bundle, ROOT)

    def test_block_energy(self):
        blocks = json.loads((self.bundle / 'blocks.json').read_text())
        block = next(b for b in blocks if b['id'] == 'r2-gate-n1024-coreml-w4a16')

        def edit(rows):
            middle = (block['block']['start_ns'] + block['block']['end_ns']) // 2
            row = next(r for r in rows if r['capture'] == block['capture'] and r['receipt']['monotonic_after_ns'] > middle)
            fields = plistlib.loads(row['plist_fields'].encode())
            fields['processor']['ane_energy'] += 5000
            added = 5000 * 1e9 / fields['elapsed_ns']
            fields['processor']['ane_power'] += added
            fields['processor']['combined_power'] += added
            row['plist_fields'] = plistlib.dumps(fields, sort_keys=True).decode()
            # Keep the decoded record consistent with the edited frame, so only the energy recomputation can object.
            row['recorded_decoded'] = decode_power(row['plist_fields'].encode() + b'\0', row['receipt'])
        self.change('power.jsonl.gz', edit)
        with self.fails('g7_recorded_energy:0:r2-gate-n1024-coreml-w4a16'):
            derive(self.bundle, ROOT)

    def test_recorded_speed(self):
        def edit(value):
            next(b for b in value if b['id'] == 'r0-gate-n1024-coreai-a8w4_int8_lut')['recorded_measurement']['positions_per_second'] *= 1.001
        self.change('blocks.json', edit)
        with self.fails('g7_recorded_speed:r0-gate-n1024-coreai-a8w4_int8_lut'):
            derive(self.bundle, ROOT)

    def test_idle_power(self):
        self.change('idle.json', lambda v: v['idle-end']['recorded']['mean_W'].__setitem__('cpu', 0.5))
        with self.fails('g7_recorded_idle_power:idle-end'):
            derive(self.bundle, ROOT)


if __name__ == '__main__':
    unittest.main()
