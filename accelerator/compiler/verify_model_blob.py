#!/usr/bin/env python3
"""Verify that the model_blob.h produces correct inference results in Python.

Reads model_blob.h, parses the byte array, and runs reference inference
to check accuracy against test_images.h labels.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

MODEL_MAGIC = 0xACCE1E27
MAX_LAYERS = 16


def parse_c_byte_array(header_path: Path) -> bytes:
    """Extract the blob bytes from model_blob.h."""
    text = header_path.read_text()
    # Find the array body between { and };
    m = re.search(r'g_model_blob\[\]\s*=\s*\{(.*?)\};', text, re.DOTALL)
    if not m:
        raise ValueError("Cannot find g_model_blob array in header")
    body = m.group(1)
    tokens = re.findall(r'0x[0-9a-fA-F]+', body)
    return bytes(int(t, 16) for t in tokens)


def parse_test_labels(header_path: Path) -> list[int]:
    """Extract expected_labels from test_images.h."""
    text = header_path.read_text()
    m = re.search(r'expected_labels\[.*?\]\s*=\s*\{(.*?)\}', text, re.DOTALL)
    if not m:
        raise ValueError("Cannot find expected_labels in header")
    return [int(x.strip()) for x in m.group(1).split(',') if x.strip()]


def parse_test_images(header_path: Path, num_images: int) -> list[np.ndarray]:
    """Extract test images from test_images.h."""
    text = header_path.read_text()
    images = []
    for i in range(num_images):
        pattern = rf'test_image_{i}\[784\]\s*=\s*\{{(.*?)\}}'
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            raise ValueError(f"Cannot find test_image_{i}")
        vals = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
        arr = np.array(vals, dtype=np.int32)
        # Convert from uint8-stored-as-int to signed int8 range
        arr = np.where(arr > 127, arr - 256, arr)
        images.append(arr)
    return images


def parse_model_blob(blob: bytes):
    """Parse the binary model blob into header + weights + biases."""
    magic, num_layers, input_size, output_size = struct.unpack_from('<IIII', blob, 0)
    assert magic == MODEL_MAGIC, f"Bad magic: 0x{magic:08x}"
    print(f"Model: {num_layers} layers, input={input_size}, output={output_size}")

    layers = []
    off = 16
    for i in range(MAX_LAYERS):
        M, N, K, act, w_off, b_off, w_scale = struct.unpack_from('<7I', blob, off)
        off += 28
        if i < num_layers:
            layers.append({
                'M': M, 'N': N, 'K': K, 'activation': act,
                'weight_offset': w_off, 'bias_offset': b_off,
                'weight_scale': w_scale,
            })
            print(f"  Layer {i}: M={M} N={N} K={K} act={act} w_off={w_off} b_off={b_off}")

    header_size = 16 + MAX_LAYERS * 28  # 464 bytes

    # Compute weight blob size
    max_w_end = 0
    for l in layers:
        end = l['weight_offset'] + l['N'] * l['K']
        if end > max_w_end:
            max_w_end = end
    weight_blob_size = max_w_end

    weights_start = header_size
    bias_start = header_size + weight_blob_size

    return {
        'num_layers': num_layers,
        'input_size': input_size,
        'output_size': output_size,
        'layers': layers,
        'weights_blob': blob[weights_start:],
        'bias_blob': blob[bias_start:],
    }


def soft_div(numer: int, denom: int) -> int:
    """Match the C soft_div exactly."""
    if denom == 0:
        return 0
    neg = (numer < 0) ^ (denom < 0)
    a = abs(numer)
    b = abs(denom)
    q = 0
    for i in range(31, -1, -1):
        shifted = b << i
        if shifted <= a:
            a -= shifted
            q |= (1 << i)
    return -q if neg else q


def clamp_int8(v: int) -> int:
    if v > 127:
        return 127
    if v < -128:
        return -128
    return v


def rescale_to_int8(vals: np.ndarray) -> np.ndarray:
    """Match the C rescale_to_int8 exactly."""
    size = len(vals)
    max_abs = int(np.max(np.abs(vals)))

    if max_abs == 0:
        return np.zeros(size, dtype=np.int8)

    shift = 0
    scaled_max = max_abs
    while scaled_max > 0x7fff and shift < 15:
        scaled_max >>= 1
        shift += 1
    if scaled_max == 0:
        scaled_max = 1

    recip = soft_div(127 << 16, scaled_max)
    out = np.zeros(size, dtype=np.int8)
    for i in range(size):
        v = int(vals[i]) >> shift
        scaled = (v * recip) >> 16
        out[i] = clamp_int8(scaled)
    return out


def run_reference_inference(model, image: np.ndarray) -> np.ndarray:
    """Run inference matching the C code exactly."""
    layers = model['layers']
    weights_blob = model['weights_blob']
    bias_blob = model['bias_blob']

    buf = image.copy().astype(np.int8)

    for li, layer in enumerate(layers):
        N, K = layer['N'], layer['K']
        w_off = layer['weight_offset']
        b_off = layer['bias_offset']
        act = layer['activation']

        # Extract weights (N x K, row-major)
        w = np.frombuffer(weights_blob, dtype=np.int8, count=N * K, offset=w_off).reshape(N, K)
        # Extract bias (N int32 values)
        b = np.frombuffer(bias_blob, dtype=np.int32, count=N, offset=b_off)

        # Matmul: accum[n] = sum_k w[n,k] * buf[k]
        accum = np.zeros(N, dtype=np.int64)
        for n in range(N):
            for k in range(K):
                accum[n] += int(w[n, k]) * int(buf[k])

        # Add bias + activation
        for n in range(N):
            v = int(accum[n]) + int(b[n])
            if act == 1 and v < 0:  # ReLU
                v = 0
            accum[n] = v

        if li == len(layers) - 1:
            return accum.astype(np.int32)

        # Rescale to int8
        buf = rescale_to_int8(accum)

    return np.zeros(0, dtype=np.int32)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    runtime_dir = repo_root / "riscv-ml-inference" / "runtime"

    blob_path = runtime_dir / "model_blob.h"
    test_path = runtime_dir / "test_images.h"

    if not blob_path.exists():
        print(f"ERROR: {blob_path} not found. Run 'make export_mnist_model' first.")
        sys.exit(1)

    blob = parse_c_byte_array(blob_path)
    print(f"Parsed blob: {len(blob)} bytes")

    model = parse_model_blob(blob)

    labels = parse_test_labels(test_path)
    num_images = len(labels)
    images = parse_test_images(test_path, num_images)

    print(f"\nRunning reference inference on {num_images} images...")
    correct = 0
    for i in range(num_images):
        output = run_reference_inference(model, images[i])
        pred = int(np.argmax(output))
        expected = labels[i]
        ok = "✓" if pred == expected else "✗"
        if pred != expected:
            print(f"  Image {i}: pred={pred} expected={expected} {ok}  out={output[:5]}...")
        if pred == expected:
            correct += 1

    wrong = num_images - correct
    print(f"\nResult: {correct}/{num_images} correct, {wrong} wrong")
    print(f"Expected tohost = {wrong + 1 if wrong > 0 else 1}")


if __name__ == "__main__":
    main()
