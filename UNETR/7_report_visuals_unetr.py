from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acdc_framework.visualization import create_best_worst_report, create_case_grid


if __name__ == "__main__":
    print(create_case_grid("transformer"))
    print(create_best_worst_report("transformer"))
