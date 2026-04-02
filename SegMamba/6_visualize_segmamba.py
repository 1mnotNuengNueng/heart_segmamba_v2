from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acdc_framework.visualization import create_prediction_figure


if __name__ == "__main__":
    print(create_prediction_figure("segmamba"))
