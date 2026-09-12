"""Verify the archived grouped-LUT evidence and its standalone ZIP, without a device."""
from pathlib import Path
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / 'findings/coreai-flattened-scale/repro'


class LUTReproduction(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='lut repro with spaces ')
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        self.copy = self.temporary / 'repro'
        shutil.copytree(REPRO, self.copy, ignore=shutil.ignore_patterns('__pycache__', '.venv', 'run-*'))

    def verify(self, directory=None, *args):
        return subprocess.run([sys.executable, '-B', str((directory or self.copy) / 'verify.py'), *args],
                              cwd=self.temporary, capture_output=True, text=True)

    def assert_failure(self, name):
        result = self.verify()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(name, result.stderr)

    def test_recorded_arrays_and_public_smoke_identity(self):
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        path = ROOT / 'results/fresh/smoke.json'
        smoke = json.loads(path.read_text())
        record = json.loads((self.copy / 'recorded/results.json').read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['source_file_sha256'])
        for field in ('environment', 'source_identity'):
            self.assertEqual(record[field], smoke[field], field)
        cases = {'coreai-group-native64', 'coreai-group-split32'}
        self.assertEqual(record['runs'], [r for r in smoke['runs'] if r['case_id'] in cases])

    def test_changed_output_fails_by_name(self):
        path = self.copy / 'recorded/coreai-group-native64/control_original.raw'
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        self.assert_failure('coreai-group-native64.control_original.sha256')

    def test_changed_comparison_fails_by_name(self):
        path = self.copy / 'recorded/results.json'
        record = json.loads(path.read_text())
        record['runs'][0]['numerical']['comparisons'][0]['numeric_mismatches'] += 1
        path.write_text(json.dumps(record))
        self.assert_failure('coreai-group-native64.control_original.intended.mismatches')

    def test_missing_case_fails_by_name(self):
        path = self.copy / 'recorded/results.json'
        record = json.loads(path.read_text())
        record['runs'].pop()
        path.write_text(json.dumps(record))
        self.assert_failure('recorded.cases')

    def test_fixed_native_output_can_be_reported(self):
        outputs = self.temporary / 'new outputs'
        for case in ('coreai-group-native64', 'coreai-group-split32'):
            shutil.copytree(self.copy / 'recorded/coreai-group-split32', outputs / case)
        result = self.verify(None, '--outputs', str(outputs))
        self.assertEqual(result.returncode, 0, result.stderr)
        native = json.loads(result.stdout)['comparisons']['coreai-group-native64'][0]
        self.assertEqual(native['intended']['numeric_mismatches'], 0)

    def test_archive_is_reproducible_and_excludes_local_outputs(self):
        create = runpy.run_path(str(self.copy / 'archive.py'))['archive']
        first, second = self.temporary / 'first.zip', self.temporary / 'second.zip'
        create(first, self.copy)
        # Exercise file selection with unrelated files inside and outside fixture/.
        for name in ('run-01/host.log', '.venv/local-config', 'fixture/private-note.txt'):
            path = self.copy / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('local-only')
        create(second, self.copy)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        extracted = self.temporary / 'extracted'
        with zipfile.ZipFile(second) as archive:
            archive.extractall(extracted)
        result = self.verify(extracted / 'coreai-grouped-lut-repro')
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
