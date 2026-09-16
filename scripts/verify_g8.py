"""Recompute G8 four-bit representation speed, component energy, admission and weight identity without a device."""
import argparse
from g8.evidence import derive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    data = derive(parser.parse_args().bundle)
    print(f"G8 passed: {len(data['configs'])} configurations re-admitted with identical decoded weights across the Core ML forms, "
          f"{len(data['blocks'])} blocks and {data['calls']:,} calls, {data['power_samples']:,} parsed power samples in "
          f"{data['captures']} captures, matching the recorded measurements. Software component energy; no device execution.")
