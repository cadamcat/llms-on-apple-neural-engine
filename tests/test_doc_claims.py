"""Mutate disposable document copies to exercise numeric-reference failures."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from doc_claims import check_document, contains_number, quantities
from g2.evidence import derive


class DocumentClaims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = quantities(ROOT, derive())

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ('README.md', 'README.zh-CN.md'):
            shutil.copy2(ROOT / name, self.root / name)

    def edit(self, old, new, name='README.md'):
        path = self.root / name
        body = path.read_text()
        self.assertIn(old, body)
        path.write_text(body.replace(old, new, 1))

    def check(self, name='README.md'):
        return check_document((self.root / name).read_text(), self.catalog,
                              'zh' if name.endswith('zh-CN.md') else 'en', name=name)

    def test_sources_exist_and_both_readmes_match(self):
        for quantity in self.catalog.values():
            self.assertTrue((ROOT / quantity.source.split(':', 1)[0]).is_file(), quantity.source)
        self.assertEqual(self.check(), [])
        self.assertEqual(self.check('README.zh-CN.md'), [])

    def test_wrong_card_fails_cli_with_correct_repeat_elsewhere(self):
        marker = '<!-- claim:g2.temperature-gap@temperature-card -->'
        correct = self.catalog['g2.temperature-gap'].display('en')
        self.edit(marker + correct, marker + '99 °C')
        # The later temperature paragraph still satisfies the old presence check.
        self.assertTrue(contains_number((self.root / 'README.md').read_text(), correct))
        for name in ('scripts', 'results', 'docs', 'articles', 'findings', 'workarounds'):
            shutil.copytree(ROOT / name, self.root / name, ignore=shutil.ignore_patterns('__pycache__'))
        result = subprocess.run([sys.executable, '-B', str(self.root / 'scripts/summarize.py')],
                                cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('g2.temperature-gap@temperature-card: expected', result.stderr)

    def test_wrong_unit_fails_at_its_location(self):
        value = self.catalog['g2.ane-rate'].display('en')
        self.edit(value + '<!-- /claim -->', value.replace('positions/s', 'requests/s') + '<!-- /claim -->')
        self.assertTrue(any('g2.ane-rate@ane-rate: expected' in error for error in self.check()))

    def test_short_count_cannot_hide_in_a_larger_number(self):
        marker = '<!-- claim:arithmetic.q8-summary@arithmetic-card -->'
        value = self.catalog['arithmetic.q8-summary'].display('en')
        self.edit(marker + value, marker + '1' + value)
        self.assertTrue(any('arithmetic.q8-summary@arithmetic-card: expected' in error for error in self.check()))
        for body in ('113', '13.50', '1,313', 'Q13', '13%', '13e2', '<!-- 13 -->'):
            self.assertFalse(contains_number(body, '13'), body)
        for body in ('13 outputs', '13.', '13, with more before QDQ'):
            self.assertTrue(contains_number(body, '13'), body)

    def test_deleting_a_reference_is_reported_by_name(self):
        marker = '<!-- claim:g2.ane-share@speed-card -->'
        value = self.catalog['g2.ane-share'].display('en')
        self.edit(marker + value + '<!-- /claim -->', value)
        self.assertIn('README.md: missing claim g2.ane-share@speed-card', self.check())

    def test_duplicate_location_or_changed_source_is_rejected(self):
        self.edit('g2.temperature-gap@temperature-detail', 'g2.temperature-gap@temperature-card')
        self.assertTrue(any('duplicate claim location temperature-card' in error for error in self.check()))
        self.edit('g2.ane-rate@ane-rate', 'g2.gpu-rate@ane-rate')
        self.assertTrue(any('unknown or changed claim g2.gpu-rate@ane-rate' in error for error in self.check()))

    def test_malformed_and_unclosed_markers_are_rejected(self):
        self.edit('claim:g2.ane-share@speed-card', 'claim:g2.ane-share')
        self.assertIn('README.md: malformed claim marker', self.check())
        path = self.root / 'README.md'
        path.write_text(path.read_text() + '\n<!-- claim:g2.ane-share@speed-card -->')
        self.assertIn('README.md: unclosed claim g2.ane-share@speed-card', self.check())

    def test_rewording_and_reordering_do_not_pin_the_sentence(self):
        path = self.root / 'README.md'
        body = path.read_text()
        marker = '<!-- claim:g2.ane-share@speed-card -->'
        value = self.catalog['g2.ane-share'].display('en')
        reference = marker + value + '<!-- /claim -->'
        body = body.replace(reference, '', 1)
        path.write_text('A rewritten introduction: ' + reference + '\n' + body)
        self.assertEqual(self.check(), [])


if __name__ == '__main__':
    unittest.main()
