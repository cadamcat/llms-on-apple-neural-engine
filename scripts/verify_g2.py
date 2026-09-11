"""Check the imported G2 records without an Apple device or model assets."""
from g2.evidence import derive

if __name__ == '__main__':
    result = derive()
    print(f"G2 portable recomputation passed: {len(result['p2'])} P2 cells, "
          f"{len(result['slots'])} slots, {result['events_checked']:,} event rows, "
          f"{len(result['starts'])} thermal groups; energy undetermined. "
          'Original device/asset audit remains a separate imported record.')
