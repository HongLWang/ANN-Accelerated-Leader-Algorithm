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
                stop = min(start + chunk_size, self.ntotal)
                ids = np.arange(start, stop, dtype=np.int64)
                dist = sqdist_rows(self._vectors[start:stop], query)
                ids = np.concatenate((best_ids, ids))
                dist = np.concatenate((best_dist, dist))
                chosen = np.lexsort((ids, dist))[:take]
                best_ids = ids[chosen]
                best_dist = dist[chosen]
            result[row, : len(best_ids)] = best_ids
        return result


def make_index(vectors: np.ndarray, config: IndexConfig) -> Any:
    """Build an index with sequential IDs equal to row positions.

    Accepts an empty array of shape (0, p). The returned object exposes add()
    and ntotal. Faiss is required for hnsw and flat; missing Faiss is an error.
    """
    _validate_index_config(config)
    data = np.ascontiguousarray(vectors, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("index vectors must have shape (M, p), with p > 0")
    dimension = int(data.shape[1])
    if config.backend == "numpy":
        index = _NumpyExactIndex(dimension)
    else:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                f"backend={config.backend!r} requires Faiss. Install faiss-cpu "
                "(or a compatible Faiss build). Use --backend numpy only for "
                "an explicitly requested small-data exact debug run."
            ) from exc
        faiss.omp_set_num_threads(int(config.threads))
        if config.backend == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, int(config.hnsw_m), faiss.METRIC_L2)
            index.hnsw.efSearch = int(config.ef_search)
            index.hnsw.efConstruction = int(config.ef_construction)
        else:
            index = faiss.IndexFlatL2(dimension)
    if len(data):
        index.add(data)
    return index


def query_candidates(index: Any, queries: np.ndarray, k: int, config: IndexConfig) -> np.ndarray:
    """Return int64 candidate IDs of shape (n_queries, k), padding with -1.

    IDs, not distances, are returned. Callers must verify original-vector
    distances in float64 before making a hard radius acceptance decision.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    data = np.ascontiguousarray(queries, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != int(index.d):
        raise ValueError("query vectors have the wrong shape")
    output = np.full((len(data), int(k)), -1, dtype=np.int64)
    if len(data) == 0 or k == 0 or int(index.ntotal) == 0:
        return output
    actual_k = min(int(k), int(index.ntotal))
    if config.backend == "numpy":
        if not isinstance(index, _NumpyExactIndex):
            raise TypeError("numpy backend requires its explicit debug index")
        output[:, :actual_k] = index.search_ids(data, actual_k)
    else:
        _, ids = index.search(data, actual_k)
        output[:, :actual_k] = np.asarray(ids, dtype=np.int64)
    output[(output < 0) | (output >= int(index.ntotal))] = -1
    return output


def _core_row_batches(n: int, batch_size: int, order: str, seed: int) -> Iterator[np.ndarray]:
    """Yield original row IDs; block_shuffle shuffles blocks, not rows inside them."""
    rng = np.random.default_rng(seed)
    if order == "shuffle":
        row_order = rng.permutation(n)
        for start in range(0, n, batch_size):
            yield row_order[start : start + batch_size]
        return
    block_order = np.arange((n + batch_size - 1) // batch_size, dtype=np.int64)
    if order == "block_shuffle":
        rng.shuffle(block_order)
    for block in block_order:
        start = int(block) * batch_size
        yield np.arange(start, min(n, start + batch_size), dtype=np.int64)


def _core_exact_candidate_best(
    X: np.ndarray,
    block64: np.ndarray,
    leader_ids: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Verify all returned candidates without a (batch, L, p) allocation."""
    rows, columns = np.nonzero(candidates >= 0)
    distances = np.full(candidates.shape, np.inf, dtype=np.float64)
    pair_chunk = max(1, 131072 // max(1, X.shape[1]))
    for start in range(0, len(rows), pair_chunk):
        rr = rows[start : start + pair_chunk]
        cc = columns[start : start + pair_chunk]
        center_ids = candidates[rr, cc]
        anchors = X[leader_ids[center_ids]]
        distances[rr, cc] = sqdist_rows(block64[rr], anchors)
    best_dist = np.min(distances, axis=1)
    sentinel = np.iinfo(np.int64).max
    tied_ids = np.where(
        (candidates >= 0) & (distances == best_dist[:, None]),
        candidates,
        sentinel,
    )
    best_ids = np.min(tied_ids, axis=1)
    best_ids[best_ids == sentinel] = -1
    return best_ids, best_dist


def _core_empty_result(
    dimension: int,
    radius: float,
    started: float,
    *,
    complete: bool,
    n_processed: int,
    proposed_centers: int,
) -> LeaderResult:
    return LeaderResult(
        centers=np.empty((0, dimension), dtype=np.float32),
        leader_ids=np.empty(0, dtype=np.int64),
        labels=np.empty(0, dtype=np.int64),
        counts=np.empty(0, dtype=np.int64),
        max_radius=np.empty(0, dtype=np.float64),
        complete=complete,
        n_processed=n_processed,
        proposed_centers=proposed_centers,
        radius=radius,
        elapsed_seconds=time.perf_counter() - started,
    )


def leader_cluster(
    X: np.ndarray,
    radius: float,
    index_config: IndexConfig,
    seed: int = 42,
    batch_size: int = 128,
    order: str = "block_shuffle",
    max_centers: int | None = None,
    log_every: int = 100000,
) -> LeaderResult:
    """Create fixed-anchor radius clusters from an initially empty index.

    X must already be a normalized float32 (N, p) array or memmap. Input
    normalization and nonfinite/zero-vector validation belong to preprocessing.
    Acceptance uses coordinate differences in float64, with no hidden radius
    tolerance. Returned centers are original stored rows, never moved means.

    Global ANN queries are batched. New leaders created within a batch are
    checked exactly against every later point of that batch, then inserted
    into the global index together. Thus a stale batch query cannot miss a
    newly created within-batch leader. Existing global leaders may still be
    missed by ANN; such misses can create extra clusters but cannot invalidate
    the radius bound. Local checks can cost O(N * batch_size * p) in the worst
    case, so batch_size is a memory/throughput tradeoff, not a complexity claim.

    Labels use original row order. When a prospective (max_centers + 1)-st
    leader is encountered, complete=False is returned with ALL arrays empty.
    n_processed includes that triggering point and proposed_centers is the
    prospective count. No partial assignment is exposed after early stopping.
    """
    started = time.perf_counter()
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("X must be an ndarray/memmap with shape (N, p), p > 0")
    if X.dtype != np.dtype(np.float32):
        raise TypeError("X must be normalized float32; normalize it in preprocessing")
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    radius_sq = radius * radius
    if not math.isfinite(radius_sq):
        raise ValueError("radius is too large to square safely")
    _validate_index_config(index_config)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if order not in {"sequential", "block_shuffle", "shuffle"}:
        raise ValueError("order must be 'sequential', 'block_shuffle', or 'shuffle'")
    if max_centers is not None and (
        isinstance(max_centers, (bool, np.bool_))
        or not isinstance(max_centers, (int, np.integer))
        or max_centers < 0
    ):
        raise ValueError("max_centers must be a nonnegative integer or None")
    if isinstance(log_every, (bool, np.bool_)) or not isinstance(log_every, (int, np.integer)) or log_every < 0:
        raise ValueError("log_every must be a nonnegative integer")

    n, dimension = X.shape
    if n == 0:
        return _core_empty_result(
            dimension, radius, started, complete=True, n_processed=0, proposed_centers=0
        )
    index = make_index(np.empty((0, dimension), dtype=np.float32), index_config)
    labels = np.full(n, -1, dtype=np.int64)
    capacity = min(n, max(16, int(batch_size)))
    leader_ids = np.empty(capacity, dtype=np.int64)
    counts = np.zeros(capacity, dtype=np.int64)
    max_radius_sq = np.zeros(capacity, dtype=np.float64)
    n_centers = 0
    processed = 0
    next_log = int(log_every) if log_every else n + 1

    def grow_arrays() -> None:
        nonlocal leader_ids, counts, max_radius_sq, capacity
        if n_centers < capacity:
            return
        new_capacity = min(n, max(capacity + 1, 2 * capacity))
        new_ids = np.empty(new_capacity, dtype=np.int64)
        new_counts = np.zeros(new_capacity, dtype=np.int64)
        new_max_sq = np.zeros(new_capacity, dtype=np.float64)
        new_ids[:n_centers] = leader_ids[:n_centers]
        new_counts[:n_centers] = counts[:n_centers]
        new_max_sq[:n_centers] = max_radius_sq[:n_centers]
        leader_ids, counts, max_radius_sq = new_ids, new_counts, new_max_sq
        capacity = new_capacity

    for row_ids in _core_row_batches(n, int(batch_size), order, seed):
        block = np.ascontiguousarray(X[row_ids], dtype=np.float32)
        block64 = np.asarray(block, dtype=np.float64)
        published_centers = n_centers
        candidate_ids = query_candidates(index, block, int(index_config.candidates), index_config)
        global_ids, global_dist = _core_exact_candidate_best(
            X, block64, leader_ids, candidate_ids
        )
        pending_ids = np.full(len(block), -1, dtype=np.int64)
        pending_dist = np.full(len(block), np.inf, dtype=np.float64)

        for local_row, original_row in enumerate(row_ids):
            processed += 1
            chosen = int(global_ids[local_row])
            chosen_dist = float(global_dist[local_row])
            local_id = int(pending_ids[local_row])
            local_dist = float(pending_dist[local_row])
            if local_id >= 0 and (
                local_dist < chosen_dist
                or (local_dist == chosen_dist and (chosen < 0 or local_id < chosen))
            ):
                chosen = local_id
                chosen_dist = local_dist

            if chosen < 0 or chosen_dist > radius_sq:
                proposed = n_centers + 1
                if max_centers is not None and proposed > max_centers:
                    if log_every:
                        print(
                            f"[leader] aborted processed={processed}/{n} "
                            f"proposed_centers={proposed} budget={max_centers} "
                            f"radius={radius:.9g} elapsed={time.perf_counter() - started:.3f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                    return _core_empty_result(
                        dimension,
                        radius,
                        started,
                        complete=False,
                        n_processed=processed,
                        proposed_centers=proposed,
                    )
                grow_arrays()
                chosen = n_centers
                leader_ids[chosen] = int(original_row)
                n_centers += 1
                chosen_dist = 0.0

                # Every later row sees this new anchor before it can create a
                # leader. Temporary storage is O(batch_size * p), not B^2.
                if local_row + 1 < len(block):
                    later = slice(local_row + 1, len(block))
                    d2 = sqdist_rows(block64[later], block64[local_row])
                    improve = d2 < pending_dist[later]
                    pending_dist[later][improve] = d2[improve]
                    pending_ids[later][improve] = chosen

            labels[int(original_row)] = chosen
            counts[chosen] += 1
            if chosen_dist > max_radius_sq[chosen]:
                max_radius_sq[chosen] = chosen_dist
            if log_every and processed >= next_log:
                print(
                    f"[leader] processed={processed}/{n} centers={n_centers} "
                    f"radius={radius:.9g} elapsed={time.perf_counter() - started:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
                next_log += int(log_every)

        if n_centers > published_centers:
            index.add(np.ascontiguousarray(X[leader_ids[published_centers:n_centers]], dtype=np.float32))

    result_ids = leader_ids[:n_centers].copy()
    result_centers = np.ascontiguousarray(X[result_ids], dtype=np.float32)
    result_counts = counts[:n_centers].copy()
    result_radius = np.sqrt(max_radius_sq[:n_centers])
    elapsed = time.perf_counter() - started
    if log_every:
        print(
            f"[leader] complete processed={processed}/{n} centers={n_centers} "
            f"radius={radius:.9g} elapsed={elapsed:.3f}s",
            file=sys.stderr,
            flush=True,
        )
    return LeaderResult(
        centers=result_centers,
        leader_ids=result_ids,
        labels=labels,
        counts=result_counts,
        max_radius=result_radius,
        complete=True,
        n_processed=processed,
        proposed_centers=n_centers,
        radius=radius,
        elapsed_seconds=elapsed,
    )


"""Optional fixed-center reassignment and audited spherical recentering.

This section is appended after the core LeaderResult / IndexConfig definitions.
It deliberately does not import another project module.
"""

import dataclasses as _refine_dataclasses
import os as _refine_os
import sys as _refine_sys
import tempfile as _refine_tempfile
import time as _refine_time

import numpy as np


def _validate_refinement_input(X, result, batch_size, log_every):
    """Validate metadata without copying the full embedding matrix."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a two-dimensional ndarray or memmap")
    if X.dtype != np.float32:
        raise ValueError("X must contain the already-normalized float32 embeddings")
    n, dim = X.shape
    if dim == 0:
        raise ValueError("X must have a positive embedding dimension")
    if not result.complete:
        raise ValueError("Refinement requires a complete leader result")
    if int(result.n_processed) != n:
        raise ValueError("A complete result must have n_processed == len(X)")
    if int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if int(log_every) != log_every or log_every < 0:
        raise ValueError("log_every must be a nonnegative integer (0 disables logs)")
    radius = float(result.radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("result.radius must be finite and nonnegative")
    centers = np.asarray(result.centers)
    if centers.ndim != 2 or centers.shape[1] != dim:
        raise ValueError("result.centers must have shape (number_of_centers, X.shape[1])")
    if centers.dtype != np.float32:
        raise ValueError("result.centers must be float32; silently rounding them is unsafe")
    m = len(centers)
    if n and not m:
        raise ValueError("A nonempty complete result must have at least one center")
    labels = np.asarray(result.labels)
    if labels.shape != (n,) or labels.dtype.kind not in "iu":
        raise ValueError("result.labels must be a one-dimensional integer array of length N")
    for start in range(0, n, int(batch_size)):
        lab = labels[start:start + int(batch_size)]
        if np.any(lab < 0) or np.any(lab >= m):
            raise ValueError("result.labels contains an invalid center id")
    leader_ids = np.asarray(result.leader_ids)
    if leader_ids.shape != (m,) or leader_ids.dtype.kind not in "iu":
        raise ValueError("result.leader_ids must be a one-dimensional integer array of length M")
    if m and (np.any(leader_ids < 0) or np.any(leader_ids >= n)):
        raise ValueError("result.leader_ids contains an invalid original-row id")
    if np.asarray(result.counts).shape != (m,):
        raise ValueError("result.counts must have length M")
    if np.asarray(result.max_radius).shape != (m,):
        raise ValueError("result.max_radius must have length M")
    for start in range(0, m, int(batch_size)):
        if not np.all(np.isfinite(centers[start:start + int(batch_size)])):
            raise ValueError("result.centers contains NaN or infinity")
    return centers, labels, leader_ids, radius


def _refinement_sqdist_rows(a, b):
    """Distances of the actual stored float32 coordinates, evaluated in float64."""
    delta = np.subtract(a, b, dtype=np.float64)
    # Match the core verifier's reduction order at the hard-radius boundary.
    return np.sum(delta * delta, axis=-1, dtype=np.float64)


def _refinement_log(stage, done, total, next_log, log_every, started):
    if log_every and (done >= next_log or done == total):
        elapsed = _refine_time.perf_counter() - started
        print(
            f"[{stage}] processed={done:,}/{total:,} elapsed={elapsed:.2f}s",
            file=_refine_sys.stderr,
            flush=True,
        )
        return (done // log_every + 1) * log_every
    return next_log


def reassign_to_leaders(
    X,
    result: "LeaderResult",
    index_config: "IndexConfig",
    batch_size=4096,
    log_every=1_000_000,
) -> "LeaderResult":
    """Reassign to the final fixed leaders, without increasing any point distance.

    ANN proposals are verified in the original float32-coordinate geometry using
    float64 arithmetic. The previous assigned center is always a fallback, even
    when ANN does not return it. Equal distances retain the previous assignment;
    ties between strictly better new candidates choose the smaller center id.

    Empty clusters are removed in increasing old-center order. ``leader_ids``,
    ``centers``, ``counts``, ``max_radius``, and all labels are compacted together.
    ``proposed_centers`` remains the historical construction counter. Input arrays
    are not mutated. If no centers become empty, their arrays may be shared with
    the input result. Partial results are rejected because construction may have
    processed rows in a shuffled order.
    """
    started = _refine_time.perf_counter()
    centers, labels, leader_ids, radius = _validate_refinement_input(
        X, result, batch_size, log_every
    )
    n, _ = X.shape
    m = len(centers)
    k = int(index_config.candidates)
    if k <= 0:
        raise ValueError("index_config.candidates must be positive")
    new_labels = labels.copy()
    counts = np.zeros(m, dtype=np.int64)
    max_sq_radius = np.zeros(m, dtype=np.float64)
    next_log = int(log_every)
    index = make_index(centers, index_config) if n else None

    for start in range(0, n, int(batch_size)):
        stop = min(n, start + int(batch_size))
        points = np.ascontiguousarray(X[start:stop])
        if not np.all(np.isfinite(points)):
            raise ValueError(f"X contains NaN or infinity in rows [{start}, {stop})")
        old_ids = labels[start:stop].astype(np.int64, copy=False)
        best_ids = old_ids.copy()
        best_sq = _refinement_sqdist_rows(points, centers[old_ids])
        candidate_ids = np.asarray(query_candidates(index, points, k, index_config))
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != len(points):
            raise ValueError("query_candidates must return a two-dimensional array of ids")
        if candidate_ids.dtype.kind not in "iu":
            raise ValueError("query_candidates returned noninteger ids")
        if np.any(candidate_ids < -1) or np.any(candidate_ids >= m):
            raise ValueError("query_candidates returned an invalid center id")
        # Sorting makes ties among new candidates deterministic without dislodging
        # an equally good original assignment. -1 padding is skipped below.
        candidate_ids = np.sort(candidate_ids, axis=1)
        for column in range(candidate_ids.shape[1]):
            ids = candidate_ids[:, column]
            valid_rows = np.flatnonzero(ids >= 0)
            if not len(valid_rows):
                continue
            valid_ids = ids[valid_rows].astype(np.int64, copy=False)
            d2 = _refinement_sqdist_rows(points[valid_rows], centers[valid_ids])
            improves = d2 < best_sq[valid_rows]
            changed_rows = valid_rows[improves]
            best_sq[changed_rows] = d2[improves]
            best_ids[changed_rows] = valid_ids[improves]
        new_labels[start:stop] = best_ids
        np.add.at(counts, best_ids, 1)
        np.maximum.at(max_sq_radius, best_ids, best_sq)
        next_log = _refinement_log(
            "reassign", stop, n, next_log, int(log_every), started
        )

    if np.any(max_sq_radius > radius * radius):
        raise ValueError(
            "Reassignment still contains a point outside result.radius; "
            "the input assignments were not a valid radius-bounded cover"
        )
    keep = np.flatnonzero(counts > 0)
    if len(keep) != m:
        old_to_new = np.full(m, -1, dtype=np.int64)
        old_to_new[keep] = np.arange(len(keep), dtype=np.int64)
        new_labels = old_to_new[new_labels].astype(labels.dtype, copy=False)
        centers = centers[keep]
        leader_ids = leader_ids[keep]
        counts = counts[keep]
        max_sq_radius = max_sq_radius[keep]
    return _refine_dataclasses.replace(
        result,
        centers=centers,
        leader_ids=leader_ids,
        labels=new_labels,
        counts=counts,
        max_radius=np.sqrt(max_sq_radius),
        elapsed_seconds=float(result.elapsed_seconds)
        + (_refine_time.perf_counter() - started),
    )


def recenter_clusters(
    X,
    result: "LeaderResult",
    work_dir,
    batch_size=8192,
    policy="safe_mean",
    log_every=1_000_000,
) -> dict:
    """Compute every spherical mean, audit all members, and keep assignments fixed.

    ``safe_mean`` adopts a cluster's normalized sum only when every original member
    remains within ``result.radius`` of that *stored float32* mean. Otherwise the
    old legal leader is retained. ``mean`` adopts all defined means and explicitly
    reports whether the radius bound survived; it never claims the bound by fiat.

    A zero sum has no spherical mean. Its ``mean_centers`` row is the old center,
    a finite placeholder marked by ``mean_defined=False``. Empty clusters are also
    marked undefined. Thus ``mean_defined`` MUST accompany saved mean centers.

    The float64 M-by-d accumulator is a temporary memmap in ``work_dir`` and is
    closed and removed even if accumulation or auditing fails. Returned matrices
    are independent float32 ndarrays, not views of this temporary file. No labels
    are changed and no second reassignment is implicit.
    """
    started = _refine_time.perf_counter()
    centers, labels, _, radius = _validate_refinement_input(
        X, result, batch_size, log_every
    )
    if policy not in ("safe_mean", "mean"):
        raise ValueError("policy must be 'safe_mean' or 'mean'")
    n, dim = X.shape
    m = len(centers)
    counts = np.zeros(m, dtype=np.int64)
    mean_centers = np.array(centers, dtype=np.float32, copy=True)
    mean_defined = np.zeros(m, dtype=bool)
    mean_max_sq = np.zeros(m, dtype=np.float64)
    old_max_sq = np.zeros(m, dtype=np.float64)
    sum_path = None
    sums = None
    block_sums = None
    try:
        if m:
            _refine_os.makedirs(_refine_os.fspath(work_dir), exist_ok=True)
            descriptor, sum_path = _refine_tempfile.mkstemp(
                prefix="leader-sums-", suffix=".f64", dir=_refine_os.fspath(work_dir)
            )
            _refine_os.close(descriptor)
            sums = np.memmap(sum_path, mode="w+", dtype=np.float64, shape=(m, dim))
            sums[:] = 0.0
            next_log = int(log_every)
            for start in range(0, n, int(batch_size)):
                stop = min(n, start + int(batch_size))
                points = X[start:stop]
                if not np.all(np.isfinite(points)):
                    raise ValueError(f"X contains NaN or infinity in rows [{start}, {stop})")
                ids = labels[start:stop].astype(np.int64, copy=False)
                np.add.at(sums, ids, np.asarray(points, dtype=np.float64))
                np.add.at(counts, ids, 1)
                next_log = _refinement_log(
                    "mean-sums", stop, n, next_log, int(log_every), started
                )
            for start in range(0, m, int(batch_size)):
                stop = min(m, start + int(batch_size))
                block_sums = np.asarray(sums[start:stop])
                norm = np.sqrt(np.einsum("ij,ij->i", block_sums, block_sums, optimize=False))
                if not np.all(np.isfinite(norm)):
                    raise ValueError("A cluster's float64 embedding sum is not finite")
                defined = (counts[start:stop] > 0) & (norm > 0.0)
                local_rows = np.flatnonzero(defined)
                if len(local_rows):
                    mean_centers[start + local_rows] = (
                        block_sums[local_rows] / norm[local_rows, None]
                    ).astype(np.float32)
                mean_defined[start:stop] = defined
            block_sums = None
    finally:
        # No returned array references sums. Clear the local slice before unmapping.
        block_sums = None
        if sums is not None:
            mapped_file = getattr(sums, "_mmap", None)
            if mapped_file is not None:
                mapped_file.close()
            sums = None
        if sum_path is not None:
            try:
                _refine_os.unlink(sum_path)
            except FileNotFoundError:
                pass

    next_log = int(log_every)
    for start in range(0, n, int(batch_size)):
        stop = min(n, start + int(batch_size))
        points = X[start:stop]
        if not np.all(np.isfinite(points)):
            raise ValueError(f"X contains NaN or infinity in rows [{start}, {stop})")
        ids = labels[start:stop].astype(np.int64, copy=False)
        np.maximum.at(old_max_sq, ids, _refinement_sqdist_rows(points, centers[ids]))
        np.maximum.at(mean_max_sq, ids, _refinement_sqdist_rows(points, mean_centers[ids]))
        next_log = _refinement_log(
            "mean-audit", stop, n, next_log, int(log_every), started
        )

    radius_sq = radius * radius
    violating_means = mean_defined & (mean_max_sq > radius_sq)
    center_is_mean = mean_defined.copy()
    if policy == "safe_mean":
        center_is_mean &= ~violating_means
    final_centers = np.array(centers, dtype=np.float32, copy=True)
    final_centers[center_is_mean] = mean_centers[center_is_mean]
    final_max_sq = np.where(center_is_mean, mean_max_sq, old_max_sq)
    radius_guaranteed = bool(np.all(final_max_sq <= radius_sq))
    if policy == "safe_mean" and not radius_guaranteed:
        raise ValueError(
            "safe_mean cannot preserve the radius: an old fallback leader already "
            "has members outside result.radius"
        )
    # Undefined means use a finite old-center placeholder. Its radius is also a
    # placeholder; consumers must retain mean_defined when saving these arrays.
    mean_max_radius = np.sqrt(mean_max_sq)
    nonempty = counts > 0
    return {
        "centers": final_centers,
        "mean_centers": mean_centers,
        "center_is_mean": center_is_mean,
        "mean_defined": mean_defined,
        "counts": counts,
        "max_radius": np.sqrt(final_max_sq),
        "mean_max_radius": mean_max_radius,
        "old_max_radius": np.sqrt(old_max_sq),
        "radius": radius,
        "policy": policy,
        "radius_guaranteed": radius_guaranteed,
        "radius_violating_mean_count": int(np.count_nonzero(violating_means)),
        "zero_mean_count": int(np.count_nonzero(nonempty & ~mean_defined)),
        "empty_cluster_count": int(np.count_nonzero(~nonempty)),
        "mean_adopted_count": int(np.count_nonzero(center_is_mean)),
        "fallback_count": int(np.count_nonzero(~center_is_mean)),
        "n_processed": n,
        "elapsed_seconds": _refine_time.perf_counter() - started,
    }



# ---------------------------------------------------------------------------
# Input preparation, bounded automatic radius calibration, and CLI.
# This section is assembled after leader_core_part.py and leader_refine_part.py.
# ---------------------------------------------------------------------------
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional


def _pipeline_log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def prepare_embeddings(input_path, output_path, batch_size=16384,
                       assume_normalized=False, log_every=100000):
    """Validate .npy input; normalize once into a float32 memory-mapped .npy.

    No rows are silently dropped. Labels always refer to original input rows.
    ``assume_normalized`` still validates every row and requires float32 input.
    All later geometry is computed on these stored float32 vectors, rechecked
    in float64. This is a numerical, not symbolic, unit-sphere representation.
    """
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    input_path = Path(input_path)
    if input_path.suffix.lower() != '.npy':
        raise ValueError('Input must be a 2-D .npy array; .npz cannot be memory-mapped.')
    X = np.load(input_path, mmap_mode='r', allow_pickle=False)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError('Input embeddings must be a nonempty N x dimension matrix.')
    if X.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Input embeddings must have dtype float32 or float64.')
    if assume_normalized and X.dtype != np.float32:
        raise ValueError('--normalized-input requires float32 input.')
    output_path = Path(output_path)
    if not assume_normalized and input_path.resolve() == output_path.resolve():
        raise ValueError('Normalization output must differ from the input path.')
    if not assume_normalized and output_path.exists():
        raise FileExistsError('Normalization output already exists: ' + str(output_path))
    temporary = output_path.with_name(output_path.name + '.partial')
    Y = None
    next_log = log_every if log_every else None
    try:
        if not assume_normalized:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Y = np.lib.format.open_memmap(
                temporary, mode='w+', dtype=np.float32, shape=X.shape)
        for start in range(0, len(X), batch_size):
            stop = min(len(X), start + batch_size)
            block = np.asarray(X[start:stop], dtype=np.float64)
            if not np.all(np.isfinite(block)):
                bad = np.flatnonzero(~np.all(np.isfinite(block), axis=1))[0] + start
                raise ValueError('Nonfinite embedding at input row %d.' % bad)
            # Scaling by each row's largest coordinate avoids norm overflow.
            scale = np.max(np.abs(block), axis=1)
            if np.any(scale == 0):
                bad = np.flatnonzero(scale == 0)[0] + start
                raise ValueError('Zero embedding at input row %d; cosine is undefined.' % bad)
            scaled = block / scale[:, None]
            scaled_norms = np.sqrt(np.einsum('ij,ij->i', scaled, scaled))
            if assume_normalized:
                norms = scale * scaled_norms
                if not np.all(np.abs(norms - 1.0) <= 2e-6):
                    bad = np.flatnonzero(np.abs(norms - 1.0) > 2e-6)[0] + start
                    raise ValueError('--normalized-input: row %d is not unit length.' % bad)
            else:
                Y[start:stop] = (scaled / scaled_norms[:, None]).astype(np.float32)
            if next_log is not None and stop >= next_log:
                _pipeline_log('[normalize] %d/%d' % (stop, len(X)))
                next_log = stop + log_every
        if assume_normalized:
            return X
        Y.flush()
        del Y
        Y = None
        os.replace(temporary, output_path)
        return np.load(output_path, mmap_mode='r', allow_pickle=False)
    except Exception:
        if Y is not None:
            try:
                Y.flush()
            finally:
                del Y
        if temporary.exists():
            temporary.unlink()
        raise


_LEADER_ARRAYS = ('centers', 'leader_ids', 'labels', 'counts', 'max_radius')


def save_leader_result(result, directory):
    if not result.complete:
        raise ValueError('Cannot save an aborted trial as a complete partition.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in _LEADER_ARRAYS:
        np.save(directory / (name + '.npy'), getattr(result, name), allow_pickle=False)
    _write_json(directory / 'leader_meta.json', {
        'complete': True,
        'n_processed': int(result.n_processed),
        'proposed_centers': int(result.proposed_centers),
        'radius': float(result.radius),
        'elapsed_seconds': float(result.elapsed_seconds),
    })


def load_leader_result(directory, mmap_mode='r'):
    directory = Path(directory)
    with (directory / 'leader_meta.json').open(encoding='utf-8') as stream:
        meta = json.load(stream)
    arrays = {name: np.load(directory / (name + '.npy'), mmap_mode=mmap_mode,
                            allow_pickle=False) for name in _LEADER_ARRAYS}
    return LeaderResult(**arrays, **meta)


@dataclass
class AutoRadiusResult:
    best_result_dir: str
    chosen_radius: float
    trial_history: List[Dict[str, Any]]
    within_target_range: bool
    history_path: str


class NoFeasibleRadiusError(RuntimeError):
    def __init__(self, message, trial_history, history_path):
        super().__init__(message)
        self.trial_history = trial_history
        self.history_path = history_path


def automatic_radius_search(X, target_clusters, index_config, work_dir,
                            seed=42, batch_size=128, order='block_shuffle',
                            r_min=0.0001, r_max=2.0, max_trials=10,
                            target_tolerance=0.05, log_every=100000):
    """Coarse radius sweep followed by bounded adaptive refinement.

    All trials use the FULL data, fixed order/config, and a center-count cap.
    A capped trial stops at the first proposed (target_clusters + 1)-st center.
    We keep the smallest TESTED feasible radius. This is NOT an exact binary
    search, nor a proof that a smaller untested radius is infeasible. A greedy
    Leader center count need not be monotone as radius changes.

    The selected complete partition is persisted, so it need not be rerun.
    Exact cluster count is not guaranteed: target_clusters is an upper budget.
    """
    if X.ndim != 2 or len(X) == 0:
        raise ValueError('X must be a nonempty 2-D array.')
    if not isinstance(target_clusters, (int, np.integer)) or target_clusters < 1:
        raise ValueError('target_clusters must be a positive integer.')
    if max_trials < 1:
        raise ValueError('max_trials must be positive.')
    if not (math.isfinite(r_min) and math.isfinite(r_max) and 0 <= r_min <= r_max):
        raise ValueError('Require finite 0 <= r_min <= r_max.')
    if not (0 <= target_tolerance < 1):
        raise ValueError('target_tolerance must be in [0, 1).')
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    search_dir = Path(tempfile.mkdtemp(prefix='radius_search_', dir=work_dir))
    history_path = search_dir / 'radius_search.json'
    best_dir = search_dir / 'best_leader'
    history = []
    tested = set()
    best_radius = None
    best_count = None
    lower_count = math.ceil(min(target_clusters, len(X)) * (1.0 - target_tolerance))
    verbose = bool(log_every)

    def run_trial(radius, phase):
        nonlocal best_radius, best_count
        radius = float(radius)
        if radius in tested or len(history) >= max_trials:
            return
        tested.add(radius)
        _pipeline_log('[radius] trial %d/%d: r=%.9g, budget=%d' %
                      (len(history) + 1, max_trials, radius, target_clusters), verbose)
        result = leader_cluster(
            X, radius, index_config, seed=seed, batch_size=batch_size,
            order=order, max_centers=int(target_clusters), log_every=log_every)
        number = int(len(result.centers)) if result.complete else int(result.proposed_centers)
        item = {
            'trial': len(history) + 1, 'phase': phase, 'radius': radius,
            'complete': bool(result.complete),
            'cluster_count': number if result.complete else None,
            'cluster_count_lower_bound': number,
            'n_processed': int(result.n_processed), 'n_total': int(len(X)),
            'elapsed_seconds': float(result.elapsed_seconds),
            'max_observed_radius': float(np.max(result.max_radius))
                if result.complete and len(result.max_radius) else None,
        }
        history.append(item)
        if result.complete and number <= target_clusters:
            if best_radius is None or radius < best_radius:
                save_leader_result(result, best_dir)
                best_radius, best_count = radius, number
        _write_json(history_path, {
            'target_clusters': int(target_clusters), 'target_is_upper_bound': True,
            'chosen_radius': best_radius, 'chosen_cluster_count': best_count,
            'trials': history,
            'search_guarantee': 'smallest tested feasible radius, not global minimum',
        })
        _pipeline_log('[radius] %s: centers=%s, processed=%d/%d, %.2fs' %
                      ('complete' if result.complete else 'budget exceeded',
                       str(number) if result.complete else '>=%d' % number,
                       result.n_processed, len(X), result.elapsed_seconds), verbose)
        del result
        gc.collect()

    coarse_count = min(5, max_trials)
    if r_min == r_max:
        coarse = [r_max]
    elif coarse_count == 1:
        coarse = [r_max]
    elif r_min > 0:
        coarse = np.geomspace(r_min, r_max, coarse_count).tolist()
    else:
        if coarse_count == 2:
            coarse = [0.0, r_max]
        else:
            coarse = [0.0] + np.geomspace(r_max * 1e-4, r_max, coarse_count - 1).tolist()
    for radius in coarse:
        run_trial(radius, 'coarse')
        if best_radius is not None:
            break
    # Usually refine the interval immediately below the best feasible radius.
    # Every fourth refinement also explores a wider lower interval, including
    # ones whose endpoints both failed. This is a bounded search heuristic,
    # not a monotonicity assumption or a proof about untested radii.
    refine_steps = 0
    while best_radius is not None and len(history) < max_trials:
        if best_count >= lower_count or best_radius <= r_min:
            break
        points = sorted(set([float(r_min), float(best_radius)] +
                            [v for v in tested if r_min <= v <= best_radius]))
        gaps = []
        floor = max(r_max * 1e-12, 1e-15)
        for left, right in zip(points[:-1], points[1:]):
            if right - left <= max(1e-12, abs(right) * 1e-9):
                continue
            mid = math.sqrt(left * right) if left > 0 else right / 2.0
            if mid in tested or not (left < mid < right):
                continue
            score = math.log((right + floor) / (left + floor))
            gaps.append((score, right, mid))
        if not gaps:
            break
        refine_steps += 1
        if refine_steps % 4 == 0:
            selected_gap = max(gaps)
            phase = 'explore_lower_gap'
        else:
            selected_gap = max(gaps, key=lambda gap: gap[1])
            phase = 'refine_near_best'
        run_trial(selected_gap[2], phase)
    if best_radius is None:
        raise NoFeasibleRadiusError(
            'No TESTED radius in the requested range met the center budget. '
            'This is not a proof that no feasible clustering exists. '
            'Inspect the trial log; consider a larger allowed radius, stronger '
            'ANN search, or a different center budget.', history, str(history_path))
    within = lower_count <= best_count <= target_clusters
    return AutoRadiusResult(str(best_dir), float(best_radius), history,
                            bool(within), str(history_path))


def _summary(values):
    values = np.asarray(values)
    return {
        'min': float(np.min(values)), 'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p95': float(np.quantile(values, 0.95)),
        'p99': float(np.quantile(values, 0.99)), 'max': float(np.max(values)),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description='ANN Leader clustering with radius calibration and audited spherical recentering.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=True, help='N x dimension float32/float64 .npy embeddings')
    parser.add_argument('--output', required=True, help='New or empty output directory')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--target-clusters', type=int, help='Upper budget; automatically calibrate radius')
    mode.add_argument('--radius', type=float, help='Use a fixed chord/Euclidean radius')
    parser.add_argument('--radius-min', type=float, default=0.0001)
    parser.add_argument('--radius-max', type=float, default=2.0)
    parser.add_argument('--max-trials', type=int, default=10)
    parser.add_argument('--target-tolerance', type=float, default=0.05,
                        help='Stop tuning when count is in [(1-tol)*budget, budget]')
    parser.add_argument('--backend', choices=['hnsw', 'flat', 'numpy'], default='hnsw',
                        help='flat/numpy are exact small-data reference backends')
    parser.add_argument('--hnsw-links', type=int, default=32)
    parser.add_argument('--ef-search', type=int, default=64)
    parser.add_argument('--ef-construction', type=int, default=200)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--prepare-batch-size', type=int, default=16384)
    parser.add_argument('--refine-batch-size', type=int, default=4096)
    parser.add_argument('--order', choices=['sequential', 'shuffle', 'block_shuffle'],
                        default='block_shuffle')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--normalized-input', action='store_true',
                        help='Validate and reuse unit float32 .npy without making a normalized copy')
    parser.add_argument('--no-reassign', action='store_true',
                        help='Skip the optional second assignment pass before final center recomputation')
    parser.add_argument('--center-policy', choices=['safe_mean', 'mean'], default='safe_mean',
                        help='safe_mean retains old centers for clusters whose mean violates radius')
    parser.add_argument('--log-every', type=int, default=100000)
    parser.add_argument('--keep-normalized', action='store_true',
                        help='Keep the prepared normalized.npy for subsequent runs')
    parser.add_argument('--work-dir', help='Temporary workspace root, preferably local fast disk')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Output must be a new or empty directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    if args.log_every < 0:
        raise ValueError('--log-every must be nonnegative.')
    config = IndexConfig(
        backend=args.backend, hnsw_m=args.hnsw_links, ef_search=args.ef_search,
        ef_construction=args.ef_construction, candidates=args.candidates,
        threads=args.threads)
    if args.backend != 'numpy':
        try:
            import faiss
        except ImportError as exc:
            raise ImportError('Faiss is required: install a compatible faiss-cpu package. '
                              'Use --backend numpy only for small-data debugging.') from exc
    work_root = Path(args.work_dir).resolve() if args.work_dir else output
    work_root.mkdir(parents=True, exist_ok=True)
    normalized_path = output / 'normalized.npy'
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='ann_leader_work_', dir=work_root) as tmp:
        try:
            X = prepare_embeddings(
                args.input, normalized_path, batch_size=args.prepare_batch_size,
                assume_normalized=args.normalized_input, log_every=args.log_every)
            prepared_seconds = time.perf_counter() - started
            if args.target_clusters is not None:
                auto = automatic_radius_search(
                    X, args.target_clusters, config, tmp, seed=args.seed,
                    batch_size=args.batch_size, order=args.order,
                    r_min=args.radius_min, r_max=args.radius_max,
                    max_trials=args.max_trials, target_tolerance=args.target_tolerance,
                    log_every=args.log_every)
                result = load_leader_result(auto.best_result_dir)
                trial_history = auto.trial_history
                within_target = auto.within_target_range
            else:
                result = leader_cluster(
                    X, args.radius, config, seed=args.seed,
                    batch_size=args.batch_size, order=args.order,
                    log_every=args.log_every)
                trial_history, within_target = [], None
            _write_json(output / 'radius_search.json', {
                'target_clusters': args.target_clusters, 'chosen_radius': result.radius,
                'within_target_range': within_target, 'trials': trial_history,
                'guarantee': 'upper budget and smallest tested feasible radius; exact count not guaranteed',
            })
            initial_count = len(result.centers)
            if not args.no_reassign:
                result = reassign_to_leaders(
                    X, result, config, batch_size=args.refine_batch_size,
                    log_every=args.log_every)
            # Every cluster mean is computed. Only the policy decides which
            # of those newly computed means become the final geometric centers.
            refined = recenter_clusters(
                X, result, tmp, batch_size=args.refine_batch_size,
                policy=args.center_policy, log_every=args.log_every)
            arrays = {
                'centers': refined['centers'], 'mean_centers': refined['mean_centers'],
                'labels': result.labels, 'counts': refined['counts'],
                'max_radius': refined['max_radius'],
                'mean_max_radius': refined['mean_max_radius'],
                'leader_ids': result.leader_ids, 'leader_centers': result.centers,
                'center_is_mean': refined['center_is_mean'],
                'mean_defined': refined['mean_defined'],
            }
            for name, array in arrays.items():
                np.save(output / (name + '.npy'), array, allow_pickle=False)
            actual_max = float(np.max(refined['max_radius']))
            report = {
                'algorithm': 'fixed-leader radius cover with ANN proposal and exact distance acceptance',
                'input': str(Path(args.input).resolve()), 'input_shape': list(X.shape),
                'target_clusters': args.target_clusters, 'target_is_upper_bound': True,
                'chosen_radius': float(result.radius),
                'within_target_range_before_reassignment': within_target,
                'initial_leader_count': int(initial_count),
                'final_cluster_count': int(len(refined['centers'])),
                'mean_cluster_size': float(len(X) / len(refined['centers'])),
                'singleton_cluster_count': int(np.count_nonzero(refined['counts'] == 1)),
                'center_policy': args.center_policy,
                'center_is_mean_count': int(np.count_nonzero(refined['center_is_mean'])),
                'radius_violating_mean_count': int(refined['radius_violating_mean_count']),
                'zero_mean_count': int(refined['zero_mean_count']),
                'radius_preserved': bool(refined['radius_guaranteed']),
                'actual_max_radius': actual_max,
                'radius_audit_absolute_tolerance': 0.0,
                'cluster_radius_summary': _summary(refined['max_radius']),
                'cluster_size_summary': _summary(refined['counts']),
                'preparation_seconds': prepared_seconds,
                'total_seconds': time.perf_counter() - started,
                'index_config': dict(config.__dict__),
                'seed': args.seed, 'order': args.order, 'batch_size': args.batch_size,
                'reassignment_enabled': not args.no_reassign,
                'geometry': 'Euclidean distance on stored float32 L2-normalized embeddings',
                'labels_semantics': 'labels[i] is the cluster of ORIGINAL input row i',
                'leader_ids_semantics': 'original leader row IDs; mean centers need not be real photos',
                'files': {key: key + '.npy' for key in arrays},
            }
            if args.backend != 'numpy':
                report['faiss_version'] = getattr(faiss, '__version__', 'unknown')
            _write_json(output / 'report.json', report)
            print(json.dumps({
                'output': str(output), 'radius': report['chosen_radius'],
                'clusters': report['final_cluster_count'],
                'mean_centers_used': report['center_is_mean_count'],
                'radius_preserved': report['radius_preserved'],
                'seconds': report['total_seconds'],
            }, ensure_ascii=False, indent=2))
            del arrays, refined, result, X
            gc.collect()
            if not args.normalized_input and not args.keep_normalized:
                normalized_path.unlink()
            return 0
        except NoFeasibleRadiusError as exc:
            _write_json(output / 'radius_search.json', {
                'target_clusters': args.target_clusters, 'chosen_radius': None,
                'trials': exc.trial_history, 'error': str(exc),
            })
            raise


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, ImportError,
            NoFeasibleRadiusError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
