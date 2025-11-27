# src/minhash.py

"""
Minhash signature computation for the Netflix LSH assignment.

This module provides:
- A function to generate random hash functions of the form h(m) = (a*m + b) % p.
- A function to compute minhash signatures for all users in a User x Movie CSR matrix.
"""

from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix


def generate_hash_functions(
    num_hashes: int,
    max_value: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    # Generate random hash functions of the form h(x) = (a * x + b) % p.
    
    # Choose a fixed large prime > max_value. We can keep this simple.
    # 2_000_003 is a prime larger than typical n_movies (~17k).
    p = 2_000_003

    a = rng.integers(1, p, size=num_hashes, dtype=np.int64)
    b = rng.integers(0, p, size=num_hashes, dtype=np.int64)

    return a, b, p


def compute_minhash_signatures(
    R: csr_matrix,
    num_hashes: int,
    seed: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute minhash signatures for all users in a User x Movie CSR matrix.

    For each user u and each hash function h_i, we compute:
        signature[u, i] = min_{m in movies rated by u} h_i(m)

    """
    
    if not isinstance(R, csr_matrix):
        raise TypeError("R must be a csr_matrix")

    n_users, n_movies = R.shape

    rng = np.random.default_rng(seed)
    a, b, p = generate_hash_functions(num_hashes, n_movies, rng)

    # Precompute hash values for all movies and all hash functions.
    # movie_ids: 0, 1, ..., n_movies-1
    movie_ids = np.arange(n_movies, dtype=np.int64)
    # h_vals shape: (num_hashes, n_movies)
    # h_vals[i, m] = h_i(m)
    h_vals = (a[:, None] * movie_ids[None, :] + b[:, None]) % p

    # Prepare output array: each row is a user, each column is a hash function.
    # We'll use int32 to save some memory (values are < p ~ 2e6).
    signatures = np.empty((n_users, num_hashes), dtype=np.int32)

    # For convenience, keep references to CSR internals
    indptr = R.indptr
    indices = R.indices

    for u in range(n_users):
        start = indptr[u]
        end = indptr[u + 1]
        user_movies = indices[start:end]  # 0-based movie indices rated by user u

        if user_movies.size == 0:
            # This case should not happen in this dataset (each user rated >= 300 movies),
            # but we handle it defensively.
            signatures[u, :] = p + 1  # some large value
        else:
            # Take the precomputed hash values for these movies and compute the min.
            # user_hashes shape: (num_hashes, k_u)
            user_hashes = h_vals[:, user_movies]
            # min over movies (axis=1) -> shape (num_hashes,)
            minhash_values = user_hashes.min(axis=1)
            signatures[u, :] = minhash_values.astype(np.int32)

        if verbose and (u + 1) % 10_000 == 0:
            print(f"  Processed {u + 1}/{n_users} users for minhash signatures")

    return signatures
