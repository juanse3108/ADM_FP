# src/lsh.py

"""
LSH banding for minhash signatures.

This module takes a minhash signature matrix and:
- Splits it into bands.
- Hashes each band to buckets.
- Produces candidate user pairs that share at least one identical band.
"""

from typing import Dict, List, Set, Tuple
import numpy as np


def _band_key_to_bytes(band: np.ndarray) -> bytes:
    
    #Convert a 1D array of integers (a band of the signature) into a bytes object suitable for use as a dictionary key.
    
    # Ensure contiguous int32, then view as raw bytes
    band = np.asarray(band, dtype=np.int32)
    return band.tobytes()


def lsh_candidate_pairs(
    signatures: np.ndarray,
    rows_per_band: int,
    max_bucket_size: int = 100,
    verbose: bool = True,
) -> Set[Tuple[int, int]]:
    
    # Perform LSH banding on a minhash signature matrix and return candidate pairs.
    
    if signatures.ndim != 2:
        raise ValueError("signatures must be a 2D array")

    n_users, num_hashes = signatures.shape

    if rows_per_band <= 0:
        raise ValueError("rows_per_band must be positive")

    num_bands = num_hashes // rows_per_band
    if num_bands == 0:
        raise ValueError(
            f"rows_per_band={rows_per_band} is too large for num_hashes={num_hashes}"
        )

    if verbose:
        print(f"LSH banding with:")
        print(f"  n_users      = {n_users}")
        print(f"  num_hashes   = {num_hashes}")
        print(f"  rows_per_band= {rows_per_band}")
        print(f"  num_bands    = {num_bands}")
        leftover = num_hashes - num_bands * rows_per_band
        if leftover > 0:
            print(f"  Warning: ignoring leftover {leftover} hash components at the end.")

    # Dictionary: (band_index, band_bytes) -> list of user indices
    buckets: Dict[Tuple[int, bytes], List[int]] = {}

    # Build buckets
    for band_idx in range(num_bands):
        start = band_idx * rows_per_band
        end = start + rows_per_band

        if verbose:
            print(f"Processing band {band_idx + 1}/{num_bands} (columns {start}:{end})")

        for u in range(n_users):
            band = signatures[u, start:end]  # shape: (rows_per_band,)
            key_bytes = _band_key_to_bytes(band)
            bucket_key = (band_idx, key_bytes)

            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(u)

    if verbose:
        print("Finished filling buckets. Now generating candidate pairs...")

    # Generate candidate pairs from buckets
    candidates: Set[Tuple[int, int]] = set()
    num_large_buckets = 0

    for (band_idx, band_key), users in buckets.items():
        k = len(users)
        if k < 2:
            continue  # no pairs here

        if k > max_bucket_size:
            # Skip very large buckets to avoid O(k^2) explosion
            num_large_buckets += 1
            continue

        # Generate all pairs (u1, u2) with u1 < u2
        # Simple double loop; k is limited by max_bucket_size
        for i in range(k):
            u1 = users[i]
            for j in range(i + 1, k):
                u2 = users[j]
                if u1 < u2:
                    pair = (u1, u2)
                else:
                    pair = (u2, u1)
                candidates.add(pair)

    if verbose:
        print(f"Number of candidate pairs: {len(candidates)}")
        print(f"Number of skipped large buckets (size > {max_bucket_size}): {num_large_buckets}")

    return candidates
