"""Recompute G4 A matched-graph speed and component energy without a device."""
import argparse
from g4a.evidence import derive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle')
    data = derive(parser.parse_args().bundle)
    print(f"G4 A passed: {data['request_count']} raw requests, {data['power_samples']} parsed power samples, "
          f"{len(data['blocks'])} admitted blocks matching the recorded summary, {len(data['comparison'])} G3 comparisons. "
          'Software component energy; no device execution.')
