import json
override = {
  "M": 128,
  "N": 128,
  "K": 784,
  "tile_m": 4,
  "tile_n": 8,
  "burst_size": 16,
  "prefetch_depth": 1,
  "tile_b": 1,
  "hardware_dataflow_mode": 1
}
with open("auto_tuner/rl_override.json", "w") as f:
    json.dump(override, f)
