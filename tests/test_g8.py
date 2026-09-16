"""Exercise portable G8 accounting against corrupted disposable bundles."""
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
from g8.evidence import BASE, derive


class G8Evidence(unittest.TestCase):
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

    @staticmethod
    def block(value, name):
        return next(b for b in value if b['id'] == name)

    def test_complete_bundle(self):
        data = self.data
        self.assertEqual(len(data['blocks']), 60)
        self.assertEqual(sum(1 for b in data['blocks'].values() if b['phase'] == 'follow-up'), 9)
        self.assertEqual(data['classes'][('mlp', 1024, 'original')], [['coreai', 'int4'], ['fp16', 'fp16_lut', 'int8_lut']])

    def test_decoded_weights(self):
        def edit(value):
            value['gate-n1024-coreml-fp16_lut']['asset']['audit']['weights'][0]['decoded_fp16_sha256'] = '0' * 64
        self.change('configs.json', edit)
        with self.fails('g8_decoded_weights:gate-1024'):
            derive(self.bundle, ROOT)

    def test_saved_representation(self):
        def edit(value):
            value['mlp-n64-coreml-int4']['asset']['audit']['counts']['constexpr_lut_to_dense'] = 3
        self.change('configs.json', edit)
        with self.fails('g8_saved_representation:mlp-n64-coreml-int4'):
            derive(self.bundle, ROOT)

    def test_numeric_screen(self):
        def edit(value):
            value['gate-n64-coreml-int4']['admission']['numeric']['comparisons']['negative']['all']['relative_l2'] = 0.006
        self.change('configs.json', edit)
        with self.fails('g8_numeric_rule:gate-n64-coreml-int4'):
            derive(self.bundle, ROOT)

    def test_numeric_inventory(self):
        original = json.loads((self.bundle / 'configs.json').read_text())
        for missing in ('control', 'empty', 'all', 'ordinary'):
            with self.subTest(missing=missing):
                value = json.loads(json.dumps(original))
                comparisons = value['gate-n64-coreml-int4']['admission']['numeric']['comparisons']
                if missing == 'control':
                    del comparisons['negative']
                elif missing == 'empty':
                    comparisons['negative'] = {}
                else:
                    del comparisons['negative'][missing]
                (self.bundle / 'configs.json').write_text(json.dumps(value))
                with self.fails('g8_numeric_inventory:gate-n64-coreml-int4'):
                    derive(self.bundle, ROOT)

    def test_mach_inventory(self):
        original = json.loads((self.bundle / 'configs.json').read_text())
        for counts in (None, [], [1] * 5, [True] * 6):
            with self.subTest(counts=counts):
                value = json.loads(json.dumps(original))
                mappings = value['gate-n64-coreml-int4']['admission']['requests_per_control_by_mapping']
                if counts is None:
                    del mappings['mach']
                else:
                    mappings['mach'] = counts
                (self.bundle / 'configs.json').write_text(json.dumps(value))
                with self.fails('g8_mach_inventory:gate-n64-coreml-int4'):
                    derive(self.bundle, ROOT)

    def test_recount_under_phase_mapping(self):
        def edit(value):
            self.block(value, 'r2-gate-n64-coreai-int8_lut')['admission']['requests_per_control_by_mapping']['wall'][4] = 2
        self.change('blocks.json', edit)
        with self.fails('g8_clock_mapping:r2-gate-n64-coreai-int8_lut'):
            derive(self.bundle, ROOT)

    def test_mach_recount(self):
        def edit(value):
            self.block(value, 'r1-gate-n1024-coreml-fp16_lut')['admission']['requests_per_control_by_mapping']['mach'][0] = 0
        self.change('blocks.json', edit)
        with self.fails('g8_mach_requests:r1-gate-n1024-coreml-fp16_lut'):
            derive(self.bundle, ROOT)

    def test_follow_up_mapping(self):
        def edit(value):
            self.block(value, 'r1-mlp-n1024-coreml-int4')['admission']['placement']['clock_mapping'] = 'wall'
        self.change('blocks.json', edit)
        with self.fails('g8_phase_mapping:r1-mlp-n1024-coreml-int4'):
            derive(self.bundle, ROOT)

    def test_rejection_record(self):
        def edit(value):
            value['rejected_first_round'][0]['admission']['placement']['requests_per_control'] = [1] * 6
            value['rejected_first_round'][0]['admission']['requests_per_control_by_mapping']['wall'] = [1] * 6
        self.change('rejections.json', edit)
        with self.fails('g8_rejection_record:r0-mlp-n1024-coreml-fp16'):
            derive(self.bundle, ROOT)

    def test_follow_up_inventory(self):
        self.change('rejections.json', lambda value: value['cancelled_later_rounds'].pop())
        with self.fails('g8_follow_up_inventory'):
            derive(self.bundle, ROOT)

    def test_block_energy(self):
        blocks = json.loads((self.bundle / 'blocks.json').read_text())
        block = self.block(blocks, 'r0-mlp-n1024-coreml-int4')

        def edit(rows):
            middle = (block['block']['start_ns'] + block['block']['end_ns']) // 2
            row = next(r for r in rows if r['capture'] == block['capture'] and r['receipt']['monotonic_after_ns'] > middle)
            fields = plistlib.loads(row['plist_fields'].encode())
            fields['processor']['cpu_energy'] += 5000
            added = 5000 * 1e9 / fields['elapsed_ns']
            fields['processor']['cpu_power'] += added
            fields['processor']['combined_power'] += added
            row['plist_fields'] = plistlib.dumps(fields, sort_keys=True).decode()
            # Keep the decoded record consistent with the edited frame, so only the energy recomputation can object.
            row['recorded_decoded'] = decode_power(row['plist_fields'].encode() + b'\0', row['receipt'])
        self.change('power.jsonl.gz', edit)
        with self.fails('g8_recorded_energy:0:r0-mlp-n1024-coreml-int4'):
            derive(self.bundle, ROOT)

    def test_recorded_speed(self):
        def edit(value):
            self.block(value, 'r2-mlp-n64-coreml-fp16')['recorded_measurement']['positions_per_second'] *= 1.001
        self.change('blocks.json', edit)
        with self.fails('g8_recorded_speed:r2-mlp-n64-coreml-fp16'):
            derive(self.bundle, ROOT)

    def test_idle_power(self):
        self.change('idle.json', lambda v: v['follow-up/idle-start']['recorded_mean_W'].__setitem__('gpu', 0.01))
        with self.fails('g8_recorded_idle_power:follow-up/idle-start'):
            derive(self.bundle, ROOT)


if __name__ == '__main__':
    unittest.main()
