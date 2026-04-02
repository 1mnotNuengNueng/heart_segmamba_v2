from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acdc_framework.evaluate import evaluate_predictions


if __name__ == "__main__":
    print(evaluate_predictions("transformer"))
