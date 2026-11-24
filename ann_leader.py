#!/usr/bin/env python3
"""ANN-accelerated fixed-Leader clustering for large photo embedding arrays.

Standalone implementation: NumPy + Faiss (HNSW or Flat). Run --help for CLI.
Original row IDs are retained. All radius acceptance and final audits use
float64 distances on normalized float32 stored embeddings. See README.md
for calibration semantics, memory requirements, and center policies.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class IndexConfig:
    backend: str = "hnsw"
    hnsw_m: int = 32
    ef_search: int = 64
    ef_construction: int = 200
    candidates: int = 8
    threads: int = 1


@dataclass
class LeaderResult:
    centers: np.ndarray
    leader_ids: np.ndarray
    labels: np.ndarray
    counts: np.ndarray
    max_radius: np.ndarray
    complete: bool
    n_processed: int
    proposed_centers: int
    radius: float
    elapsed_seconds: float


def _validate_index_config(config: IndexConfig) -> None:
    if config.backend not in {"hnsw", "flat", "numpy"}:
        raise ValueError("backend must be 'hnsw', 'flat', or explicit debug backend 'numpy'")
    for name in ("hnsw_m", "ef_search", "ef_construction", "candidates", "threads"):
        value = getattr(config, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def sqdist_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared L2 distances over the last axis, using float64 differences.

    Inputs may be rows or broadcast-compatible arrays. This deliberately does
    not use 2 - 2 * dot(a, b): the stored float32 vectors need not have norm
    exactly one, and subtraction avoids cancellation for almost equal vectors.
    """
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sum(delta * delta, axis=-1, dtype=np.float64)


class _NumpyExactIndex:
    """Explicit small-data debug index; never selected as an implicit fallback."""

    def __init__(self, dimension: int) -> None:
        self.d = int(dimension)
        self.ntotal = 0
        self._vectors = np.empty((0, self.d), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        data = np.ascontiguousarray(vectors, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.d:
            raise ValueError("added vectors have the wrong shape")
        needed = self.ntotal + len(data)
        if needed > len(self._vectors):
            capacity = max(16, needed, 2 * len(self._vectors))
            enlarged = np.empty((capacity, self.d), dtype=np.float32)
            enlarged[: self.ntotal] = self._vectors[: self.ntotal]
            self._vectors = enlarged
        self._vectors[self.ntotal : needed] = data
        self.ntotal = needed

    def search_ids(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Exact float64 search, with bounded temporary distance storage."""
        result = np.full((len(queries), k), -1, dtype=np.int64)
        if k == 0 or self.ntotal == 0:
            return result
        take = min(k, self.ntotal)
        chunk_size = max(1, min(4096, 262144 // max(1, self.d)))
        for row, query in enumerate(queries):
            best_ids = np.empty(0, dtype=np.int64)
            best_dist = np.empty(0, dtype=np.float64)
            for start in range(0, self.ntotal, chunk_size):
