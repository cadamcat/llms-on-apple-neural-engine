"""Evidence must reject corrupted bytes and tell the sign controls apart.

These run anywhere: they build a tiny fixture on disk and never touch a device,
a model or a framework beyond NumPy.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ane_scope.common import sha
from ane_scope.evidence import verify_outputs

PHASES = ('control_original', 'control_zero', 'control_negative', 'control_original_repeat')


class EvidenceTests(unittest.TestCase):
    def fixture(self, root, parity='odd'):
        """A four-control fixture whose outputs match its reference exactly."""
        reference = np.array([[[[2., -3.]]]], np.float16)
        np.save(root / 'reference.npy', reference)
        records = []
        for phase in PHASES:
            if phase == 'control_zero':
                value = np.zeros_like(reference)
            elif phase == 'control_negative' and parity == 'odd':
                value = -reference
            else:
                value = reference
            path = root / (phase + '.raw')
            value.tofile(path)
            records.append({'phase': phase, 'output_file': path.name,
                            'output_sha256': sha(path), 'output_count': 2})
        metadata = {'reference_file': 'reference.npy', 'reference_repeat_columns': 1,
                    'output_shape': [1, 1, 1, 2], 'negative_parity': parity, 'l2_limit': 0}
        return metadata, records

    def test_even_function_negative_input_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, calls = self.fixture(root, 'even')
            self.assertTrue(verify_outputs(root, metadata, root, calls)['passed'])

    def test_stale_output_hash_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, calls = self.fixture(root)
            (root / calls[0]['output_file']).write_bytes(b'\0' * 4)
            with self.assertRaisesRegex(ValueError, 'identity'):
                verify_outputs(root, metadata, root, calls)

    def test_nonzero_zero_control_is_a_finding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, calls = self.fixture(root)
            path = root / calls[1]['output_file']
            np.array([1., 0.], np.float16).tofile(path)
            calls[1]['output_sha256'] = sha(path)
            result = verify_outputs(root, metadata, root, calls)
            self.assertFalse(result['passed'])
            self.assertFalse(result['comparisons'][1]['passed'])
