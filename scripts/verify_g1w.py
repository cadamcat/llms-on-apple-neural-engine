"""Recompute G1-W quantized speed conditions and the E4B QAT MLP results without a device."""
import argparse
from g1w.evidence import derive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    data = derive(parser.parse_args().bundle)
    depth = data['depth']['a8_over_w4a16']
    print(f"G1-W passed: {data['measured_calls']:,} measured calls recomputed against the recorded summaries. "
          f"A8W4/W4A16 {depth['both-1']:.2f}× at 1 layer, {depth['both-128']:.2f}× at 128; "
          f"exact-zero FP16 {data['density']['over_sparse']['old']:.2f}× faster; "
          f"E4B stack A8W4 {data['e4b']['speed_vs_w4a16']['native']:.3f}× W4A16. No device execution.")
