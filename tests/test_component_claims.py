"""A repeated number cannot hide a changed or deleted G1-W/G5 reference."""
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from component_claims import check
from g1w.evidence import derive as derive_g1w, quoted_values as g1w_quotes
from g1w.claims import quantities as g1w_quantities
from g5.evidence import derive as derive_g5, quoted_values as g5_quotes
from g5.claims import quantities as g5_quantities


class ComponentClaims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogs = {'g1w': g1w_quantities(derive_g1w()), 'g5': g5_quantities(derive_g5())}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for family in self.catalogs:
            registry = Path('scripts') / family / 'claims.json'
            for name in [str(registry), *json.loads((ROOT / registry).read_text())]:
                target = self.root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)

    def check(self, family):
        return check(self.root, family, self.catalogs[family])

    def target(self, family):
        name = ('findings/quantized-speedup-conditions/README.md' if family == 'g1w'
                else 'findings/attention-product-precision/README.md')
        value = '1.88×' if family == 'g1w' else '3.95%'
        path = self.root / name
        body = path.read_text()
        matches = list(re.finditer(r'<!-- claim:(' + family + r'\.[^>]+) -->(.*?)<!-- /claim -->', body))
        target = next(m for m in matches if m.group(2) == value)
        self.assertGreater(sum(m.group(2) == value for m in matches), 1)
        return path, body, target

    def test_registered_documents(self):
        for family in self.catalogs:
            with self.subTest(family=family):
                self.assertEqual(self.check(family), [])

    def test_previous_numeric_inventory_has_bound_locations(self):
        for family, data, quotes in [('g1w', derive_g1w(), g1w_quotes), ('g5', derive_g5(), g5_quotes)]:
            for name, values in quotes(data).items():
                body = (self.root / name).read_text()
                registered = {v.replace('−', '-') for v in re.findall(
                    r'<!-- claim:' + family + r'\.[^>]+-->(.*?)<!-- /claim -->', body)}
                self.assertLessEqual(set(values), registered, (family, name))

    def test_one_wrong_occurrence_fails_by_name(self):
        for family in self.catalogs:
            with self.subTest(family=family):
                path, body, target = self.target(family)
                path.write_text(body[:target.start(2)] + '999' + body[target.end(2):])
                self.assertTrue(any(target.group(1) + ': expected' in error for error in self.check(family)))

    def test_one_deleted_marker_is_missing(self):
        for family in self.catalogs:
            with self.subTest(family=family):
                path, body, target = self.target(family)
                path.write_text(body[:target.start()] + target.group(2) + body[target.end():])
                self.assertTrue(any('missing claim ' + target.group(1) in error for error in self.check(family)))

    def test_registered_span_can_move(self):
        for family in self.catalogs:
            with self.subTest(family=family):
                path, body, target = self.target(family)
                path.write_text(body[:target.start()] + body[target.end():] + '\n' + target.group() + '\n')
                self.assertEqual(self.check(family), [])


if __name__ == '__main__':
    unittest.main()
