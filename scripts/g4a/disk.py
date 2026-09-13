"""Recompute the disk space held by ANECompilerService from the redacted observations and the G4 A load gate."""
import hashlib
import json
import re
from pathlib import Path

from g3.protocol import require
from .evidence import GiB, ROOT

EVIDENCE = 'findings/ane-compiler-service-disk/evidence'
STAMP = re.compile(r'mpsgraph-\d+-(\d{4}-\d\d-\d\d)_(\d\d)_(\d\d)_\d\d-')


def held(rows, pid):
    files = [r for r in rows if r['pid'] == pid and r['command'].startswith('ANECompil')]
    require(files and all(r['user'] == 'root' and r['nlink'] == 0 and r['fd'] == 'txt' and
                          '/com.apple.MetalPerformanceShadersGraph/mpsgraph-' in r['name'] and
                          r['name'].endswith('.bc.mlir') for r in files), 'disk_held_file_kind')
    return files


def derive_disk(g4a, repo=None):
    root = (Path(repo) if repo is not None else ROOT) / EVIDENCE
    provenance = json.loads((root / 'provenance.json').read_text())
    for name, digest in provenance['products'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, 'disk_product_identity:' + name)
    data = json.loads((root / 'observations.json').read_text())

    first = held(data['listing_1335']['rows'], 1720)
    require(len(first) == len(data['listing_1335']['rows']), 'disk_other_holders_1335')
    total = sum(r['size_bytes'] for r in first)
    large = [r['size_bytes'] for r in first if r['size_bytes'] > 5 * GiB]
    stamps = sorted(STAMP.search(r['name']).groups() for r in first)

    second_rows = data['listing_1400']['rows']
    second = held(second_rows, 51145)
    others = [r for r in second_rows if r not in second]
    require(all(r['user'] == 'session user' and '/payload-' in r['name'] for r in others), 'disk_other_holders_1400')

    before, stuck, after = data['free_space']
    require(before['service_1720_running'] and stuck['service_1720_running'] and not after['service_1720_running'],
            'disk_service_state')
    released = (after['available_kib'] - before['available_kib']) / 2 ** 20

    match = re.fullmatch(r'\S+ reclaimed signals=(\d+):(\w+) held_files=(\d+) held_bytes=(\d+) '
                         r'free_kib_before=(\d+) free_kib_after=(\d+)', data['reclaim_log'][0])
    require(match, 'disk_reclaim_log')
    pid, signal, files, held_bytes, free_before, free_after = match.groups()
    require(int(pid) == 51145 and int(files) == 3, 'disk_reclaim_identity')
    # The daemon held the two files listed at 14:00 and one written later by the ANE 500 host (capacity 768).
    third = int(held_bytes) - sum(r['size_bytes'] for r in second)
    c768 = {r['size_bytes'] for r in first if '/extend_768_' in r['name']}
    require(third in c768, 'disk_reclaim_third_file')

    arms = g4a['disk']['arms']
    restart = next(i for i, s in enumerate(arms) if s['waits'])
    ane_before = [s['statfs_change_GiB'] for s in arms[:restart - 1] if s['arm'] == 'ane']
    ane_after = [s['statfs_change_GiB'] for s in arms[restart:] if s['arm'] == 'ane' and 'statfs_change_GiB' in s]
    gpu = [s['statfs_change_GiB'] for s in arms if s['arm'] == 'gpu' and 'statfs_change_GiB' in s]
    return {'macos': data['macos'], 'held_files': len(first), 'held_GiB': total / GiB,
            'large_files': len(large), 'large_GiB': (min(large) / GiB, max(large) / GiB),
            'oldest': stamps[0], 'released_GiB': released, 'free_before_GiB': before['available_kib'] / 2 ** 20,
            'free_after_GiB': after['available_kib'] / 2 ** 20,
            'restart_held': len(second), 'reclaim_signal': signal, 'reclaim_files': int(files),
            'reclaim_GiB': int(held_bytes) / GiB, 'reclaim_free_GiB': (int(free_after) - int(free_before)) / 2 ** 20,
            'ane_change_before_restart_GiB': ane_before, 'ane_change_after_restart_GiB': ane_after,
            'gpu_change_GiB': gpu, 'gate_waits': arms[restart]['waits'], 'gate_wait_arm': (arms[restart]['input_N'], arms[restart]['arm']),
            'gate_waiting_usable_GiB': max(arms[restart]['waiting_usable_bytes']) / GiB,
            'gate_required_GiB': arms[restart]['gate']['required_bytes'] / GiB}
