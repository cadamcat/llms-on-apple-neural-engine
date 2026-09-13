"""Dry-run decisions of the ANECompilerService reclaim script; no service is signalled."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'workarounds/ane-compiler-reclaim/ane-compiler-reclaim.sh'
OBSERVATIONS = ROOT / 'findings/ane-compiler-service-disk/evidence/observations.json'
DEAD = 9000000  # above kern.pid_max, so ps -p never finds it
SCRATCH = '/private/var/folders/xx/yy/T/com.apple.MetalPerformanceShadersGraph'


def fixture(files, service=1720):
    lines = [f'p{service}']
    for owner, links, size, *rest in files:
        name = rest[0] if rest else f'{SCRATCH}/mpsgraph-{owner}-2026-09-13_13_16_03-1/extend_ANE_region_0_0.bc.mlir'
        lines += ['ftxt', f's{size}', f'k{links}', f'n{name}']
    return '\n'.join(lines) + '\n'


@unittest.skipUnless(os.path.exists('/bin/sh') and os.path.exists('/usr/bin/pgrep'), 'needs a POSIX shell and pgrep')
class ReclaimDecisions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.live = subprocess.Popen(['sleep', '60'])
        self.addCleanup(self.live.wait)
        self.addCleanup(self.live.kill)

    def decide(self, listing, busy='no-such-process-g4', scratch=()):
        (self.root / 'lsof.txt').write_text(listing)
        for pid in scratch:
            (self.root / f'mpsgraph-{pid}-2026-09-13_13_16_03-1').mkdir()
        env = dict(os.environ, ANE_RECLAIM_DRY_RUN='1', ANE_RECLAIM_LSOF_INPUT=str(self.root / 'lsof.txt'),
                   ANE_RECLAIM_BUSY_PROCESS=busy, ANE_RECLAIM_SCRATCH_GLOB=str(self.root / 'mpsgraph-*'),
                   ANE_RECLAIM_LOG=str(self.root / 'log'))
        out = subprocess.run(['/bin/sh', str(SCRIPT)], env=env, capture_output=True, text=True, check=True)
        self.assertFalse((self.root / 'log').exists())
        return out.stdout.strip()

    def test_recorded_listing_is_reclaimed(self):
        rows = json.loads(OBSERVATIONS.read_text())['listing_1335']['rows']
        files = []
        for i, r in enumerate(rows):
            # Keep each recorded file name but give it an owner PID that cannot be alive.
            tail = r['name'].split('/mpsgraph-', 1)[1].split('-', 1)[1]
            files.append((DEAD + i, r['nlink'], r['size_bytes'], f'{SCRATCH}/mpsgraph-{DEAD + i}-{tail}'))
        total = sum(r['size_bytes'] for r in rows)
        self.assertEqual(self.decide(fixture(files)), f'decision=reclaim services=1720 held_files={len(rows)} held_bytes={total}')

    def test_linked_and_unrelated_files_are_not_counted(self):
        listing = fixture([(DEAD, 0, 3 << 30), (DEAD + 1, 1, 7 << 30), (DEAD + 2, 0, 9 << 30, '/private/var/db/other.bin')])
        self.assertEqual(self.decide(listing), f'decision=reclaim services=1720 held_files=1 held_bytes={3 << 30}')

    def test_live_input_owner_blocks(self):
        listing = fixture([(DEAD, 0, 7 << 30), (self.live.pid, 1, 7 << 30)])
        self.assertEqual(self.decide(listing), f'decision=skip reason=input_owner_alive pid={self.live.pid}')

    def test_live_scratch_owner_blocks(self):
        self.assertEqual(self.decide(fixture([(DEAD, 0, 7 << 30)]), scratch=[DEAD + 5, self.live.pid]),
                         f'decision=skip reason=scratch_owner_alive pid={self.live.pid}')

    def test_running_measurement_blocks(self):
        self.assertEqual(self.decide(fixture([(DEAD, 0, 7 << 30)]), busy='sleep'), 'decision=skip reason=measurement_running')

    def test_small_hold_is_left(self):
        self.assertEqual(self.decide(fixture([(DEAD, 0, 1 << 30)])), f'decision=skip reason=below_min held_bytes={1 << 30}')

    def test_no_scratch_files(self):
        self.assertEqual(self.decide('p1720\nftxt\ns10\nk1\nn/usr/lib/dyld\n'), 'decision=skip reason=no_held_scratch')


if __name__ == '__main__':
    unittest.main()
