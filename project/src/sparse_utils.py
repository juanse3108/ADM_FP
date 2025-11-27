# src/sparse_utils.py

"""
Utilities for working with sparse User x Movie matrices in CSR format.

These helpers provide convenient ways to:
- Get the list of movies rated by a given user.
- Compute basic statistics like number of movies per user.
"""

from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix


def get_user_movies(R: csr_matrix, user_index: int) -> np.ndarray:
    
    # Return the list of movie indices rated by a given user (0-based indexing)!!!
    
    if not isinstance(R, csr_matrix):
        raise TypeError("R must be a csr_matrix")

    if user_index < 0 or user_index >= R.shape[0]:
        raise IndexError(
            f"user_index {user_index} out of bounds for matrix with {R.shape[0]} users"
        )

    start = R.indptr[user_index]
    end = R.indptr[user_index + 1]
    movies = R.indices[start:end]  # 0-based column indices
    return movies


def user_movie_counts(R: csr_matrix) -> np.ndarray:
    
    # Compute the number of movies rated by each user.

    if not isinstance(R, csr_matrix):
        raise TypeError("R must be a csr_matrix")

    # Difference between 2 consecutives entries in the indptr will be the number of entries for a specific row in the R matrix hence number of movies rated by that user
    counts = np.diff(R.indptr)
    return counts


def describe_user_activity(R: csr_matrix) -> None:
    """
    Print basic statistics about how many movies users rated
    """
    counts = user_movie_counts(R)
    print("User activity statistics (number of movies rated per user):")
    print(f"  min   : {counts.min()}")
    print(f"  max   : {counts.max()}")
    print(f"  mean  : {counts.mean():.2f}")
    print(f"  median: {np.median(counts):.2f}")
