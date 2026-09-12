"""Recompute complete-model G3 speed and component energy without a device."""
import argparse
from g3.evidence import derive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    data = derive(parser.parse_args().bundle)
    print(f"G3 passed: {data['request_count']} raw requests, {len(data['coverage'])} coverage points, "
          f"{data['power_samples']} parsed power samples, {len(data['pairs'])} stage-energy pairs. "
          'Software component energy; no device execution.')
