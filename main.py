"""Run all analysis problems in sequence."""
import subprocess
import sys

PROBLEMS = [
    "problem1a", "problem1b", "problem1c",
    "problem2a", "problem2b",
    "problem3",
    "problem4",
]

if __name__ == "__main__":
    for p in PROBLEMS:
        print(f"\n{'=' * 60}\n  Running {p}\n{'=' * 60}")
        result = subprocess.run([sys.executable, f"src/{p}.py"])
        if result.returncode != 0:
            print(f"  {p} FAILED with code {result.returncode}")
            sys.exit(result.returncode)
    print(f"\n{'=' * 60}\n  All {len(PROBLEMS)} problems completed.\n{'=' * 60}")
