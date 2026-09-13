"""Recompute G5 attention timings and expose its imported relative L2 values without a device."""
import argparse
from g5.evidence import derive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    data = derive(parser.parse_args().bundle)
    speed = data['speed']
    print(f"G5 passed: {data['timed_operations']:,} timed operations recomputed against the recorded summaries. "
          f"Accurate blocks {speed['kv_unrolled_vs_dense.resident']:.3f}× dense; "
          f"B1024/B4096 {speed['b1024_vs_b4096.resident']:.3f}× resident. No device execution.")
