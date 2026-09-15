"""The decode-query-width verifier fails by name on changed recorded outcomes."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / 'findings/ane-short-decode-query/repro'
spec = importlib.util.spec_from_file_location('short_query_verify', REPRO / 'verify.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class ShortQuery(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.copy = Path(temporary.name) / 'repro'
        shutil.copytree(REPRO, self.copy, ignore=shutil.ignore_patterns('__pycache__'))

    def edit(self, change):
        path = self.copy / 'recorded/observations.json'
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def case(self, data, name):
        return next(c for c in data['cases'] if c['case'] == name)

    def test_recorded_outcomes(self):
        _, rows = verify.outcomes()
        self.assertEqual({r['query'] for r in rows if r['outcome'] == 'fails'}, {1, 2})

    def test_query_four_failure_is_rejected(self):
        self.edit(lambda d: self.case(d, 'l1-c768-q12483-t8-q8-q4')['results']['4'].__setitem__('host_returncode', 130))
        with self.assertRaisesRegex(ValueError, 'short_query_success:l1-c768-q12483-t8-q8-q4:4'):
            verify.outcomes(self.copy / 'recorded')

    def test_query_one_without_driver_status_is_rejected(self):
        self.edit(lambda d: self.case(d, 'c1280-n1024-r1')['results']['1']['log'].__setitem__('failure_status_lines', []))
        with self.assertRaisesRegex(ValueError, 'short_query_failure:c1280-n1024-r1:1'):
            verify.outcomes(self.copy / 'recorded')


if __name__ == '__main__':
    unittest.main()
