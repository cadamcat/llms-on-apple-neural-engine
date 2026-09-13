"""Exercise the portable G5 timing accounting against disposable bundles."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from g5.evidence import BASE, derive, quoted_values


class G5Evidence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.bundle = Path(temporary.name) / 'bundle'
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
        manifest = self.bundle / 'provenance.json'
        record = json.loads(manifest.read_text())
        record['products'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(record))

    def test_recorded_bundle_recomputes(self):
        data = derive(self.bundle)
        self.assertLess(data['speed']['kv_unrolled_vs_dense.resident'], 1)
        self.assertIn('findings/attention-product-precision/README.md', quoted_values(data))

    def test_changed_block_fails_the_paired_speed(self):
        def slow(value):
            for block in value['post_pv_b4096_vs_b1024']:
                if block['arm'] == 'post1024' and block['mode'] == 'resident':
                    block['operation_ns'][:] = [3 * v for v in block['operation_ns']]
        self.change('timings.json.gz', slow)
        with self.assertRaisesRegex(ValueError, 'g5_block_speed.resident'):
            derive(self.bundle)

    def test_uniform_timing_scale_fails_absolute_accounting(self):
        def double(value):
            for block in value['dense_vs_post_pv_b1024']:
                block['operation_ns'] = [2 * n for n in block['operation_ns']]
        self.change('timings.json.gz', double)
        with self.assertRaisesRegex(ValueError, 'g5_dense_block_ms'):
            derive(self.bundle)

    def test_recorded_block_mean_fails_by_name(self):
        self.change('evidence.json', lambda v: v['reported']['dense_round_block_means'][0]
                    .__setitem__('ms_per_operation', 999))
        with self.assertRaisesRegex(ValueError, 'g5_dense_block_ms'):
            derive(self.bundle)

    def test_duplicated_block_fails(self):
        self.change('timings.json.gz', lambda v: v['dense_vs_post_pv_b1024'].append(v['dense_vs_post_pv_b1024'][0]))
        with self.assertRaisesRegex(ValueError, 'g5_duplicate_block'):
            derive(self.bundle)

    def test_nonzero_normalized_output_fails_the_quote(self):
        self.change('evidence.json', lambda v: v['numerics']['constant_v_pv']
                    ['weighted_value/full-normalized'].__setitem__('output_max', 1e-4))
        with self.assertRaisesRegex(ValueError, 'g5_normalized_zero_output'):
            quoted_values(derive(self.bundle))


if __name__ == '__main__':
    unittest.main()
