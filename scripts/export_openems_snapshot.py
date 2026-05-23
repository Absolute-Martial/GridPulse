"""Generate a GridPulse JSON snapshot for local backend integration."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gridpulse.engine import GridPulseEngine
from gridpulse.openems_bridge import export_snapshot


def main() -> None:
    engine = GridPulseEngine()
    path = export_snapshot(engine)
    print(path)


if __name__ == "__main__":
    main()
