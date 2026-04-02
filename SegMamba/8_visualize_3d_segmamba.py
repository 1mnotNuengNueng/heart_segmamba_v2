from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acdc_framework.visualization_3d import create_3d_mesh_html, create_triplanar_html


if __name__ == "__main__":
    print(create_3d_mesh_html("segmamba"))
    print(create_triplanar_html("segmamba"))
