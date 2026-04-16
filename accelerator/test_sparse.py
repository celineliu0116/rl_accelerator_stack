import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path.cwd() / "compiler"))
sys.path.insert(0, str(Path.cwd() / "auto_tuner" / "env"))

import json
override = {
    "M": 128,
    "N": 10,
    "K": 128,
    "tile_m": 4,
    "tile_n": 8,
    "burst_size": 16,
    "prefetch_depth": 1,
    "tile_b": 1,
    "hardware_dataflow_mode": 1
}
with open("auto_tuner/rl_override.json", "w") as f:
    json.dump(override, f)

print("Created rl_override.json for debugging")
