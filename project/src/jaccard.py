# src/jaccard.py

"""
Exact Jaccard similarity computation for user pairs using a CSR User x Movie matrix,
and verification of candidate pairs produced by LSH.
"""

from typing import Iterable, Set, Tuple
import numpy as np
from scipy.sparse import csr_matrix


def jaccard_similarity_csr(R: csr_matrix, u1: int, u2: int) -> float:
    
    # Compute Jaccard similarity between two users u1 and u2 using a CSR matrix.

    if not isinstance(R, csr_matrix):
        raise TypeError("R must be a csr_matrix")

    if u1 == u2:
        # Jaccard of a set with itself is 1.0,
        # but in our context we only compare distinct users.
        return 1.0

    # Ensure u1 < u2 just for consistency (not strictly necessary here)
    if u1 > u2:
        u1, u2 = u2, u1

    indptr = R.indptr
    indices = R.indices

    # Movies rated by each user (sorted arrays of 0-based movie indices)
    start1, end1 = indptr[u1], indptr[u1 + 1]
    start2, end2 = indptr[u2], indptr[u2 + 1]

    cols1 = indices[start1:end1]
    cols2 = indices[start2:end2]

    # Two-pointer intersection on sorted arrays
    i = j = 0
    intersection = 0
    len1 = cols1.size
    len2 = cols2.size

    while i < len1 and j < len2:
        c1 = cols1[i]
        c2 = cols2[j]
        if c1 == c2:
            intersection += 1
            i += 1
            j += 1
        elif c1 < c2:
            i += 1
        else:
            j += 1

    if intersection == 0:
        return 0.0

    union_size = len1 + len2 - intersection
    if union_size == 0:
        # Should never happen (both empty sets), but guard anyway.
        return 0.0

    return intersection / union_size


def verify_candidates_and_write(
    R: csr_matrix,
    candidates: Iterable[Tuple[int, int]],
    threshold: float,
    output_path: str,
    close_each_write: bool = True,
    verbose: bool = True,
) -> int:
    
    # Verify candidate pairs by exact Jaccard similarity and write those above a threshold to an output file

    num_written = 0

    # To optionally do one open/close outside (when close_each_write=False)
    f = None
    if not close_each_write:
        f = open(output_path, "a", buffering=1)  # line-buffered

    for idx, (u1, u2) in enumerate(candidates):
        jacc = jaccard_similarity_csr(R, u1, u2)
        if jacc >= threshold:
            # Convert back to 1-based user IDs for output
            uid1 = u1 + 1
            uid2 = u2 + 1
            if uid1 > uid2:
                uid1, uid2 = uid2, uid1

            line = f"{uid1},{uid2}\n"

            if close_each_write:
                with open(output_path, "a") as f_out:
                    f_out.write(line)
            else:
                f.write(line)

            num_written += 1

        if verbose and (idx + 1) % 10_000 == 0:
            print(f"  Checked {idx + 1} candidate pairs, written {num_written} so far")

    if not close_each_write and f is not None:
        f.close()

    if verbose:
        print(f"Total candidate pairs written (J >= {threshold}): {num_written}")

    return num_written