"""Recompute G7 Core ML / Core AI same-code speed, component energy and admission without a device."""
import argparse
from g7.evidence import derive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    data = derive(parser.parse_args().bundle)
    print(f"G7 passed: {len(data['configs'])} configurations re-admitted, {len(data['blocks'])} blocks and {data['calls']:,} calls, "
          f"{data['power_samples']:,} parsed power samples in {data['captures']} captures, matching the recorded measurements. "
          "Software component energy; no device execution.")
