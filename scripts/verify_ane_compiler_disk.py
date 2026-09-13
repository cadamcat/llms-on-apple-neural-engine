"""Recompute the disk space held by ANECompilerService from redacted observations and the G4 A load gate."""
from g4a.disk import derive_disk
from g4a.evidence import derive

if __name__ == '__main__':
    data = derive_disk(derive())
    print(f"ANECompilerService disk observations passed: {data['held_files']} deleted inputs held "
          f"({data['held_GiB']:.1f} GiB), {data['released_GiB']:.1f} GiB free after the service exited, "
          f"{data['reclaim_files']} files ({data['reclaim_GiB']:.1f} GiB) in the reclaim log. No device execution.")
