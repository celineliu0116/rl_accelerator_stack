#!/usr/bin/env python3
"""MNIST MLP export wrapper for the generic model runtime."""

from __future__ import annotations

import sys
from pathlib import Path
import struct
import numpy as np
import torch

# Add compiler/ directory to path so we can find export_model
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "compiler"))

from export_model import ACT_NONE, ACT_RELU, build_model_blob, write_model_blob_header
from train_and_export import (
    train,
    load_hidden_test_set,
    quantize_weights,
    verify_quantized_accuracy,
    quantize_input,
    export_to_c,
)


def _scaled_biases(quantized):
  s_w1 = quantized["fc1.weight"]["scale"]
  s_b1 = quantized["fc1.bias"]["scale"]
  s_w2 = quantized["fc2.weight"]["scale"]
  s_b2 = quantized["fc2.bias"]["scale"]

  b1_q = quantized["fc1.bias"]["data"].astype(np.int64)
  b2_q = quantized["fc2.bias"]["data"].astype(np.int64)

  b1_scaled = np.round(b1_q * s_w1 * 127.0 / s_b1).astype(np.int32)
  b2_scaled = np.round(b2_q * s_w2 * 127.0 / s_b2).astype(np.int32)
  return b1_scaled, b2_scaled


def _write_test_images_header(test_data, out_path: Path, num_test_images: int = 100) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)

  lines = [
      "// Auto-generated test images from MNIST",
      "#ifndef TEST_IMAGES_H",
      "#define TEST_IMAGES_H",
      "",
      "#include <stdint.h>",
      "",
      f"#define NUM_TEST_IMAGES {num_test_images}",
      "",
  ]

  labels = []
  for i in range(num_test_images):
    image, label = test_data[i]
    labels.append(int(label))
    q_image = quantize_input(image)

    lines.append(f"// Test image {i}: label = {label}")
    lines.append(f"const int8_t test_image_{i}[784] = {{")

    flat = q_image.flatten()
    for j in range(0, len(flat), 16):
      chunk = flat[j : j + 16]
      lines.append("  " + ", ".join(str(int(v)) for v in chunk) + ",")
    lines.append("};")
    lines.append("")

  lines.append(f"const int8_t *test_images[NUM_TEST_IMAGES] = {{")
  for i in range(num_test_images):
    lines.append(f"  test_image_{i},")
  lines.append("};")
  lines.append("")

  lines.append(f"const int expected_labels[NUM_TEST_IMAGES] = {{")
  lines.append("  " + ", ".join(str(l) for l in labels))
  lines.append("};")
  lines.append("")

  lines.append("#endif // TEST_IMAGES_H")
  lines.append("")

  out_path.write_text("\n".join(lines))


def main() -> None:
  repo_root = Path(__file__).resolve().parents[2]
  fw_include = repo_root / "firmware" / "include"

  print("=" * 60)
  print("MNIST MLP export for generic runtime")
  print("=" * 60)

  import torch
  cache_path = Path("mnist_mlp_quantized.pth")
  if cache_path.exists():
      model = torch.load(cache_path, map_location='cpu', weights_only=False)
  else:
      print("Training model from scratch to build cache...")
      model = train()
      torch.save(model, cache_path)
      print(f"Saved cache to '{cache_path}'")
  quantized = quantize_weights(model)
  test_data = load_hidden_test_set()
  verify_quantized_accuracy(model, quantized, test_data)

  b1_scaled, b2_scaled = _scaled_biases(quantized)

  layers = [
      {
          "M": 128,
          "N": 784,
          "K": 128,
          "W": quantized["fc1.weight"]["data"],
          "B": b1_scaled,
          "activation": ACT_RELU,
          "weight_scale": int(quantized["fc1.weight"]["scale"] * (1 << 16)),
      },
      {
          "M": 10,
          "N": 128,
          "K": 10,
          "W": quantized["fc2.weight"]["data"],
          "B": b2_scaled,
          "activation": ACT_NONE,
          "weight_scale": int(quantized["fc2.weight"]["scale"] * (1 << 16)),
      },
  ]

  blob = build_model_blob(layers=layers, input_size=784, output_size=10)

  model_bin_path = fw_include / "model.bin"
  model_header_path = fw_include / "model_blob.h"
  test_header_path = fw_include / "test_images.h"

  with open(model_bin_path, "wb") as f:
    f.write(blob)
  write_model_blob_header(blob, model_header_path)
  
  # Also re-export weights.h and test_images.h via the original script so inference_bare stays synced
  export_to_c(quantized, test_data, num_test_images=100, out_dir=fw_include)

  print(f"Wrote {model_bin_path}")
  print(f"Wrote {model_header_path}")
  print(f"Wrote {test_header_path}")
  print("\nRun with: make run INFERENCE_SRC=inference_generic.c")


if __name__ == "__main__":
  main()

