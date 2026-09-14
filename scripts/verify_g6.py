"""Recompute G6 decode-query, W4 and G4 A repeat speed and component energy without a device."""
import argparse
from g6.evidence import derive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    data = derive(parser.parse_args().bundle)
    print(f"G6 passed: {data['request_count']} raw requests, {data['power_samples']} parsed power samples in {data['captures']} captures, "
          f"{data['blocks']} blocks matching the recorded summary. Software component energy; no device execution.")
