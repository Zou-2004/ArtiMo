#!/usr/bin/env python3
"""Regression tests for sparse multi-base GPU request coalescing."""
from __future__ import annotations

import concurrent.futures
import threading
import unittest

from solve_artimo_placement import _SparseBatchIKProxy


class _Backend:
    allow_bullet_fallback = False
    environment_collision = True
    self_collision = True

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.lock = threading.Lock()

    def solve_paths_batch(self, rows):
        with self.lock:
            self.batch_sizes.append(len(rows))
        return [
            {"success": True, "candidate": row["robot_base_position_world"][0]}
            for row in rows
        ]


class SparseGpuBatchingTest(unittest.TestCase):
    def test_concurrent_base_requests_share_bounded_gpu_batches(self) -> None:
        backend = _Backend()
        proxy = _SparseBatchIKProxy(backend, 4)

        def solve(index: int):
            return proxy.solve_path(
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 1.0]],
                [float(index), 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0] * 7,
                1.2,
                False,
                [[]],
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
                answers = list(executor.map(solve, range(7)))
        finally:
            proxy.close()
        self.assertEqual([row["candidate"] for row in answers], list(map(float, range(7))))
        self.assertEqual(sum(backend.batch_sizes), 7)
        self.assertTrue(all(size <= 4 for size in backend.batch_sizes))
        self.assertLess(len(backend.batch_sizes), 7)


if __name__ == "__main__":
    unittest.main()
