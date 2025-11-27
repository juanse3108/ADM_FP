# src/data_loader.py

"""
Data loading and sparse matrix construction.

This module:
- Loads the raw (user_id, movie_id, rating) data from a .npy file.
- Builds a User x Movie sparse matrix in CSR format where:
    R[u, m] = 1 if user u rated movie m (rating is ignored for similarity).
- Prints basic info about the data and the built sparse matrix.
"""

from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix


def load_ratings_npy(path: str) -> np.ndarray:
    
    ratings = np.load(path)
    if ratings.ndim != 2 or ratings.shape[1] != 3:
        raise ValueError(
            f"Expected ratings array of shape (N, 3), got {ratings.shape}"
        )

    # Ensure integer type for ids (rating can be anything, we ignore it later)
    ratings = ratings.astype(np.int32, copy=False)
    return ratings


def infer_num_users_movies(ratings: np.ndarray) -> Tuple[int, int]:
    
    user_ids = ratings[:, 0]
    movie_ids = ratings[:, 1]

    n_users = int(user_ids.max())
    n_movies = int(movie_ids.max())

    return n_users, n_movies


def build_user_movie_matrix(
    ratings: np.ndarray,
    n_users: int,
    n_movies: int,
    dtype=np.int8,
) -> csr_matrix:
    
    user_ids = ratings[:, 0]
    movie_ids = ratings[:, 1]

    # Convert 1-based ids to 0-based indices for Python
    row_indices = user_ids - 1
    col_indices = movie_ids - 1

    # Every (user, movie) pair is a 1 in our incidence matrix
    data = np.ones_like(row_indices, dtype=dtype)

    # Build CSR matrix
    R = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_users, n_movies),
        dtype=dtype,
    )

    return R


def load_user_movie_matrix(path: str, dtype=np.int8) -> Tuple[csr_matrix, int, int]:
    
    ratings = load_ratings_npy(path)
    n_users, n_movies = infer_num_users_movies(ratings)
    R = build_user_movie_matrix(ratings, n_users, n_movies, dtype=dtype)
    return R, n_users, n_movies


def describe_matrix(R: csr_matrix, n_users: int, n_movies: int) -> None:
    
    nnz = R.nnz  # number of non-zero entries
    density = nnz / (n_users * n_movies)

    print(f"User x Movie matrix shape: {R.shape}")
    print(f"  Number of users  : {n_users}")
    print(f"  Number of movies : {n_movies}")
    print(f"  Non-zero entries : {nnz}")
    print(f"  Density          : {density:.6e} (fraction of non-zeros)")
