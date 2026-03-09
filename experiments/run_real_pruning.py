"""Public CLI entrypoint for the GRA-CNN pruning workflow.

The public release keeps this familiar command name while delegating execution
to ``experiments.run_chip_worker``, which contains the portable experiment
implementation for GRA-CNN and the main structured-pruning baselines.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_chip_worker import main


if __name__ == "__main__":
    main()
