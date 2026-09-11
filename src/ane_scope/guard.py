"""Serial child execution behind observed memory, swap, disk and time limits.

The guard samples the process tree it owns.  Compiler and runtime services that
live outside that tree are not fully counted, and sampling can miss a transient
peak, so these are conservative experiment boundaries rather than a measurement
of total memory use.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .common import dump, read

RSS_LIMIT_BYTES = 4 * 1024 ** 3
MIN_FREE_PERCENT = 17
MIN_DISK_FREE_BYTES = 30 * 1024 ** 3
MAX_RESULT_BYTES = 2 * 1024 ** 3
SWAP_GROWTH_MiB = 64
MAX_SWAP_GROWTHS = 3
RESOURCE_INTERVAL_SECONDS = 15


def run_child(command, output, *, result_root, seconds=300, deadline=None):
    """Run one child to completion, or stop the suite when a limit is reached."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(result_root)
    record = {'command': list(map(str, command)), 'started_epoch': time.time(),
              'rss_limit_bytes': RSS_LIMIT_BYTES, 'status': 'started'}
    rss, resources = [], []
    child, code = None, 125

    ledger = root / 'resources.json'
    history = read(ledger) if ledger.exists() else []
    growth = history[-1]['consecutive_swap_growth'] if history else 0

    def sample_resource():
        nonlocal growth
        pressure = subprocess.run(['/usr/bin/memory_pressure', '-Q'], text=True,
                                  capture_output=True, timeout=10, check=True)
        swapusage = subprocess.run(['/usr/sbin/sysctl', 'vm.swapusage'], text=True,
                                   capture_output=True, timeout=10, check=True)
        free = re.search(r'free percentage:\s*(\d+)%', pressure.stdout)
        used = re.search(r'used =\s*([\d.]+)M', swapusage.stdout)
        if not free or not used:
            raise RuntimeError('memory resource observation unavailable')
        previous = (resources or history)[-1] if resources or history else None
        swap = float(used[1])
        growth = growth + 1 if previous and swap - previous['swap_used_MiB'] > SWAP_GROWTH_MiB else 0
        row = {'epoch': time.time(), 'job': output.name, 'free_percent': int(free[1]),
               'swap_used_MiB': swap, 'consecutive_swap_growth': growth,
               'disk_free_bytes': shutil.disk_usage(root).free,
               'result_bytes': sum(x.stat().st_size for x in root.rglob('*') if x.is_file())}
        resources.append(row)
        dump(output / 'resources.json', resources)
        dump(ledger, history + resources)
        if (row['free_percent'] < MIN_FREE_PERCENT
                or growth >= MAX_SWAP_GROWTHS
                or row['disk_free_bytes'] < MIN_DISK_FREE_BYTES
                or row['result_bytes'] > MAX_RESULT_BYTES):
            raise RuntimeError('resource stop threshold reached')

    def sample_rss():
        listing = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,rss='], text=True,
                                 capture_output=True, timeout=5, check=True)
        rows = [tuple(map(int, s.split())) for s in listing.stdout.splitlines() if s.strip()]
        owned = {child.pid}
        while True:                                   # transitive closure of the owned tree
            grown = owned | {pid for pid, ppid, size in rows if ppid in owned}
            if grown == owned:
                break
            owned = grown
        points = [{'pid': pid, 'rss_bytes': size * 1024}
                  for pid, ppid, size in rows if pid in owned]
        row = {'epoch': time.time(), 'processes': points,
               'total_rss_bytes': sum(x['rss_bytes'] for x in points)}
        rss.append(row)
        dump(output / 'rss.json', rss)
        if row['total_rss_bytes'] > record['rss_limit_bytes']:
            raise RuntimeError('owned process tree RSS exceeded 4 GiB')

    def terminated(signum, frame):
        raise KeyboardInterrupt('termination requested')

    old_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, terminated)
    try:
        sample_resource()
        end = min(time.time() + seconds, deadline or float('inf'))
        if end <= time.time():
            raise TimeoutError('experiment cutoff reached')
        with (output / 'output.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            record['pid'] = child.pid
            dump(output / 'command.json', record)
            sample_rss()
            next_system = time.time() + RESOURCE_INTERVAL_SECONDS
            while child.poll() is None:
                if time.time() > end:
                    raise TimeoutError('bounded job timed out')
                time.sleep(.1)
                if time.time() - rss[-1]['epoch'] >= 1:
                    sample_rss()
                if time.time() >= next_system:
                    sample_resource()
                    next_system = time.time() + RESOURCE_INTERVAL_SECONDS
            code = child.wait()
            record['status'] = 'completed' if code == 0 else 'failed'
            sample_resource()
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = repr(error)
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        if child:
            # Kill any descendant left in the owned session, even after the leader exits.
            _killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _killpg(child.pid, signal.SIGKILL)
                child.wait()
            _killpg(child.pid, signal.SIGKILL)
        record.update(returncode=code, finished_epoch=time.time(),
                      sampled_rss_peak_bytes=max((x['total_rss_bytes'] for x in rss), default=0))
        dump(output / 'command.json', record)
    if record['status'] != 'completed':
        raise RuntimeError(f"{output.name} failed; see {output / 'output.log'}: "
                           f"{record.get('error', code)}")
    return record


def _killpg(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
