"""Run the complete study end-to-end, then build the PDF report.

    python run_all.py            # everything (Part 1, Part 2, E6, report)
    python run_all.py --report   # rebuild the report from existing results/ only
    python run_all.py --resume   # keep cached Part 2 / E6 folds (results/cache) from an interrupted run
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = ["part1_reproduce.py", "part2_proposed.py", "part2_estimation_validity.py", "make_report.py", "make_video_script.py"]

if __name__ == "__main__":
    steps = STEPS[-2:] if "--report" in sys.argv else STEPS
    if "--report" not in sys.argv and "--resume" not in sys.argv:
        shutil.rmtree(SRC.parent / "results" / "cache", ignore_errors=True)   # fresh run
    for s in steps:
        print(f"\n===== running {s} =====", flush=True)
        subprocess.run([sys.executable, s], cwd=SRC, check=True)
    print("\nAll done. Results in results/, report in report/.")
