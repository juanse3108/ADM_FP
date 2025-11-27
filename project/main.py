# main.py

import argparse
from scipy.sparse import csr_matrix

from src.data_loader import load_user_movie_matrix, describe_matrix
from src.sparse_utils import describe_user_activity
from src.minhash import compute_minhash_signatures
from src.lsh import lsh_candidate_pairs
from src.jaccard import verify_candidates_and_write


def parse_args():
    parser = argparse.ArgumentParser(
        description="Netflix LSH assignment: end-to-end test (subset of users)."
    )
    parser.add_argument(
        "--input_npy",
        type=str,
        required=True,
        help="Path to user_movie_rating.npy file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for minhash.",
    )
    parser.add_argument(
        "--num_hashes",
        type=int,
        default=100,
        help="Number of hash functions (signature length).",
    )
    parser.add_argument(
        "--rows_per_band",
        type=int,
        default=5,
        help="Number of rows per band for LSH.",
    )
    parser.add_argument(
        "--test_users",
        type=int,
        default=2000,
        help="Number of users to use for a quick local test.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Jaccard similarity threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result_test.txt",
        help="Output file path for similar user pairs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading User x Movie matrix from:", args.input_npy)
    R, n_users, n_movies = load_user_movie_matrix(args.input_npy, dtype="int8")
    describe_matrix(R, n_users, n_movies)

    print()
    describe_user_activity(R)

    # Use only a subset of users for local testing
    n_test_users = min(args.test_users, n_users)
    print(f"\nUsing first {n_test_users} users for test run.")

    R_small: csr_matrix = R[:n_test_users, :]

    print("\nComputing minhash signatures...")
    signatures = compute_minhash_signatures(
        R_small,
        num_hashes=args.num_hashes,
        seed=args.seed,
        verbose=True,
    )
    print("Signatures shape:", signatures.shape)

    print("\nRunning LSH banding to get candidate pairs...")
    candidates = lsh_candidate_pairs(
        signatures,
        rows_per_band=args.rows_per_band,
        max_bucket_size=500,
        verbose=True,
    )

    print(f"\nNumber of candidate pairs from LSH: {len(candidates)}")
    print(f"Writing verified pairs (J >= {args.threshold}) to: {args.output}")

    # For the subset, we use R_small and the subset indices (0..n_test_users-1)
    num_written = verify_candidates_and_write(
        R_small,
        candidates,
        threshold=args.threshold,
        output_path=args.output,
        close_each_write=True,
        verbose=True,
    )

    print(f"\nDone. Number of similar pairs written: {num_written}")
    print(f"Results stored in file: {args.output}")


if __name__ == "__main__":
    main()
