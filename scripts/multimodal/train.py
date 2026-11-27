"""Run the packaged native multimodal training command from a checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya_multimodal.train_cli import main, validate_visual_training as validate_visual_training


if __name__ == "__main__":
    main()
