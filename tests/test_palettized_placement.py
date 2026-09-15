"""The palettized-placement verifier fails by name on changed recorded logs."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / 'findings/coreai-palettized-weights-gpu/repro/recorded'
spec = importlib.util.spec_from_file_location('palettized_verify', RECORDED.parent / 'verify.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class PalettizedPlacement(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.copy = Path(temporary.name) / 'recorded'
        shutil.copytree(RECORDED, self.copy)

    def edit(self, change):
        path = self.copy / 'observations.json'
        data = json.loads(path.read_text())
        change(data['pair'])
        path.write_text(json.dumps(data))

    def test_recorded_pair(self):
        rows = verify.placement()
        self.assertFalse(rows['w4']['compile_success'])
        self.assertTrue(rows['fp16']['compile_success'])

    def test_w4_with_ane_requests(self):
        self.edit(lambda pair: pair['w4']['log_from_before_host_start'].__setitem__('ane_direct_request_success_rows', 4))
        with self.assertRaisesRegex(ValueError, 'palettized_w4_not_on_ane'):
            verify.placement(self.copy)

    def test_mismatched_export(self):
        self.edit(lambda pair: pair['w4']['export'].__setitem__('capacities', [768]))
        with self.assertRaisesRegex(ValueError, 'palettized_matched_export:capacities'):
            verify.placement(self.copy)


if __name__ == '__main__':
    unittest.main()
