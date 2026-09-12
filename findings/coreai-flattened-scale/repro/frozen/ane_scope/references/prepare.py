"""Generate the bounded ANE Scope v0.1 data unit.

Origins: the frozen Hadamard/sign/permutation construction in
``scripts/g1ic_export.py``; the independent Q8 RZA recipe in
``scripts/g1ic_heldout.py``; and the K32/RZA FP16 split arithmetic in
``scripts/g1w4_split_math.py``.  This module is deliberately model-free and
has no import-time filesystem, RNG, or device side effects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _array(a: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    x = np.asarray(a)
    if shape is not None and x.shape != shape:
        raise ValueError(f"expected shape {shape}, got {x.shape}")
    if not np.isfinite(x.astype(np.float64, copy=False)).all():
        raise ValueError("array contains non-finite values")
    return x


def fp16_rza(x: Any) -> np.ndarray:
    """Cast float values to binary16, choosing the away endpoint at ties."""
    x = _array(x).astype(np.float64, copy=False)
    nearest = x.astype(np.float16)
    n = nearest.astype(np.float64)
    low = np.where(n <= x, nearest, np.nextafter(nearest, np.float16(-np.inf)))
    high = np.where(n >= x, nearest, np.nextafter(nearest, np.float16(np.inf)))
    tie = (low != high) & (x == (low.astype(np.float64) + high.astype(np.float64)) / 2)
    return np.where(tie, np.where(x < 0, low, high), nearest).astype(np.float16)


def q8_rza(x: Any, scale: float = 0.125) -> np.ndarray:
    """Nearest Q8 with midpoint ties away from zero, followed by saturation."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    x = _array(x).astype(np.float64, copy=False) / float(scale)
    q = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return np.clip(q, -128, 127).astype(np.int8)


def q8_rne(x: Any, scale: float = 0.125) -> np.ndarray:
    """Nearest Q8 using NumPy's ties-to-even rule, followed by saturation."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    x = _array(x).astype(np.float64, copy=False) / float(scale)
    return np.clip(np.rint(x), -128, 127).astype(np.int8)


def _hadamard_weights(depth=128) -> np.ndarray:
    h = np.ones((1, 1), dtype=np.int8)
    for _ in range(9):
        h = np.block([[h, h], [h, -h]])
    if not np.array_equal(h.astype(np.int32) @ h.astype(np.int32).T,
                          np.eye(512, dtype=np.int32) * 512):
        raise ValueError("invalid Hadamard construction")
    rng = np.random.default_rng(20260910)
    out = np.empty((depth, 512, 512), dtype=np.int8)
    for i in range(depth):
        out[i] = (h[rng.permutation(512)][:, rng.permutation(512)]
                  * rng.choice(np.array([-1, 1], dtype=np.int8), (512, 1))
                  * rng.choice(np.array([-1, 1], dtype=np.int8), (1, 512)))
    return out


def _compact_fp16(weights: np.ndarray, vectors: np.ndarray, depth: int) -> np.ndarray:
    y = vectors.astype(np.float16, copy=True)
    scale = np.float16(1 / np.sqrt(512))
    for i in range(depth):
        y = ((weights[i].astype(np.float16) * scale).astype(np.float32)
             @ y.astype(np.float32)).astype(np.float16)
    return y


def _compact_w8a8(weights: np.ndarray, vectors: np.ndarray, depth: int,
                  quantizer) -> np.ndarray:
    codes = np.rint(vectors.astype(np.float64) / .125).astype(np.int8)
    scale = np.float16(1 / np.sqrt(512))
    y = codes
    for i in range(depth):
        dots = weights[i].astype(np.int32) @ y.astype(np.int32)
        z = (dots.astype(np.float64) * float(scale) * .125).astype(np.float16)
        if i < depth - 1:
            y = quantizer(z)
        else:
            y = z
    return np.asarray(y, dtype=np.float16)


def _split2(weights: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    scale = np.float16(1 / np.sqrt(512))
    x = np.rint(vectors.astype(np.float64) / .125).astype(np.int8)

    def layer(w: np.ndarray, codes: np.ndarray) -> np.ndarray:
        parts = []
        for j in range(0, 512, 32):
            dots = w[:, j:j + 32].astype(np.int64) @ codes[j:j + 32].astype(np.int64)
            parts.append(fp16_rza(dots.astype(np.float64) * float(scale) * .125))
        while len(parts) > 1:
            parts = [fp16_rza(parts[j].astype(np.float64) + parts[j + 1].astype(np.float64))
                     for j in range(0, len(parts), 2)]
        return parts[0]

    first = layer(weights[0], x)
    return layer(weights[1], q8_rza(first))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data(out: Path, profile: str = "smoke") -> dict:
    """Write a new smoke/full fixture directory and return its manifest."""
    out = Path(out)
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    weights = _hadamard_weights(128 if profile == "full" else 2)
    codes = np.random.default_rng(20260912).integers(-8, 9, (512, 16), dtype=np.int8)
    vectors = codes.astype(np.float16) * np.float16(.125)
    full = np.tile(vectors, (1, 256)).reshape(1, 512, 64, 64)
    np.save(out / "weights.npy", weights)
    np.save(out / "vectors.npy", vectors)
    full.tofile(out / "input.raw")
    depths = (2, 128) if profile == "full" else (2,)
    for depth in depths:
        np.save(out / f"reference-fp16-{depth}.npy", _compact_fp16(weights, vectors, depth))
        np.save(out / f"reference-w8a8-{depth}.npy", _compact_w8a8(weights, vectors, depth, q8_rza))
        np.save(out / f"reference-w8a8-{depth}-rne.npy", _compact_w8a8(weights, vectors, depth, q8_rne))
    np.save(out / "reference-a8w4-split32-2.npy", _split2(weights, vectors))

    # Model-free signed INT4 group fixture: each output row has two exact power-of-two scales.
    grng = np.random.default_rng(20260910)
    group_codes = grng.integers(-8, 8, (64, 64, 1, 1), dtype=np.int8)
    scales = np.empty((64, 2, 1, 1), dtype=np.float16)
    for row in range(64):
        scales[row, :, 0, 0] = (.125, .25) if row % 2 == 0 else (.25, .125)
    group_input_scales = np.r_[np.full(32, .25, np.float16), np.full(32, .5, np.float16)]
    identity = np.eye(64, dtype=np.float16).reshape(1, 64, 1, 64)
    group_y = np.empty((1, 64, 1, 64), dtype=np.float16)
    for row in range(64):
        decoded = group_codes[row, :, 0, 0].astype(np.float16)
        decoded = decoded * np.repeat(scales[row, :, 0, 0], 32)
        group_y[0, row, 0] = (decoded.astype(np.float32) @ identity[0, :, 0].astype(np.float32)).astype(np.float16)
    # Wrong candidate intentionally flattens the two K-group scales to one row scale.
    wrong = np.empty_like(group_y)
    for row in range(64):
        scalar = scales.reshape(-1)[:64][row]
        wrong[0, row, 0] = (group_codes[row, :, 0, 0].astype(np.float16) * scalar).astype(np.float16)
    np.save(out / "group_codes.npy", group_codes); np.save(out / "group_scales.npy", scales)
    np.save(out / "group_input_scales.npy", group_input_scales)
    identity.tofile(out / "group_input.raw")
    np.save(out / "reference-group.npy", group_y); np.save(out / "reference-group-wrong.npy", wrong)
    mul = np.stack([np.full((64, 1, 16), 2, np.float16), np.full((64, 1, 16), 8, np.float16)])
    mul.tofile(out / "mul_input.raw"); np.save(out / "reference-mul.npy", (mul[0] * mul[1])[None, ...])

    files = {p.name: _sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    manifest = {"profile": profile, "seed_weights": 20260910, "seed_inputs": 20260912,
                "shape": [1, 512, 64, 64], "weight_scale": .044189453125,
                "activation_scale": .125, "depths": list(depths), "files": files,
                "reference_rules": {"q8": "nearest ties away from zero, clamp int8",
                                    "q8_rne": "nearest ties to even, clamp int8",
                                    "final_fp16": "binary16 RNE", "split_fp16": "binary16 RZA",
                                    "group": "model-free signed int4, two K32 scales"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
