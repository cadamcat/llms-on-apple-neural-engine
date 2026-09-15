"""The Core ML QDQ probe verifier fails by name on changed recorded outputs, placement or graphs."""
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / 'findings/coreai-qdq-multiply-scale/repro/coreml/recorded'
spec = importlib.util.spec_from_file_location('qdq_coreml_verify', RECORDED.parent / 'verify.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class QdqCoreML(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.copy = Path(temporary.name) / 'recorded'
        shutil.copytree(RECORDED, self.copy)

    def edit_results(self, change):
        path = self.copy / 'results.json'
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def test_recorded(self):
        _, rows = verify.outputs()
        wrong = {r['arm'] for r in rows if not r['matches_reference']}
        self.assertEqual(wrong, {'s8', 's4', 's2'})

    def test_ane_output_equal_to_reference(self):
        # A corrected runtime returning 1 on the unclamped ANE arm no longer matches the recorded substitution.
        size = 1024 * 1024 // 2
        for phase in ('original', 'repeat'):
            name = f'outputs/s8-cpuAndNeuralEngine-{phase}.raw.gz'
            (self.copy / name).write_bytes(gzip.compress(struct.pack('<e', 1.0) * size, mtime=0))
        with self.assertRaisesRegex(ValueError, 'qdq_coreml_values:s8-cpuAndNeuralEngine'):
            verify.outputs(self.copy)

    def test_missing_ane_request(self):
        def change(data):
            next(r for r in data['runs'] if r['arm'] == 's4' and r['compute_units'] == 'cpuAndNeuralEngine')['ane_requests_per_control'][0] = 0
        self.edit_results(change)
        with self.assertRaisesRegex(ValueError, 'qdq_coreml_ane_requests:s4-cpuAndNeuralEngine'):
            verify.outputs(self.copy)

    def test_graph_dequantize_scale(self):
        path = self.copy / 'graphs/s2.mil'
        path.write_text(path.read_text().replace(
            'dequantize(input=%quantize_2, scale=0.0625',
            'dequantize(input=%quantize_2, scale=0.5'))
        with self.assertRaisesRegex(ValueError, 'qdq_coreml_graph:s2'):
            verify.outputs(self.copy)

    def test_graph_qdq_connection(self):
        path = self.copy / 'graphs/s2.mil'
        path.write_text(path.read_text().replace(
            'dequantize(input=%quantize_3,', 'dequantize(input=%quantize_2,'))
        with self.assertRaisesRegex(ValueError, 'qdq_coreml_graph:s2'):
            verify.outputs(self.copy)

    def test_graph_bypasses_product(self):
        path = self.copy / 'graphs/s2.mil'
        path.write_text(path.read_text().replace('quantize(input=%mul_0,', 'quantize(input=%slice_by_index_0,'))
        with self.assertRaisesRegex(ValueError, 'qdq_coreml_graph:s2'):
            verify.outputs(self.copy)

    def test_graph_clamp_bounds(self):
        path = self.copy / 'graphs/s2_clip.mil'
        path.write_text(path.read_text().replace('alpha=-64.0', 'alpha=0.0'))
        with self.assertRaisesRegex(ValueError, 'qdq_coreml_graph:s2_clip'):
            verify.outputs(self.copy)

    def test_graph_variable_names(self):
        # The graph's connections matter; generated symbol names do not.
        path = self.copy / 'graphs/s2.mil'
        path.write_text(path.read_text().replace('quantize_2', 'quantize_renamed'))
        verify.outputs(self.copy)

    def test_graph_without_clamp(self):
        path = self.copy / 'graphs/s2_clip.mil'
        path.write_text('\n'.join(line for line in path.read_text().splitlines() if '= clip(' not in line))
        with self.assertRaisesRegex(ValueError, 'qdq_coreml_graph:s2_clip'):
            verify.outputs(self.copy)


if __name__ == '__main__':
    unittest.main()
