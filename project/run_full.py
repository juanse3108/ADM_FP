# run_full.py

import argparse
from scipy.sparse import csr_matrix

from src.data_loader import load_user_movie_matrix, describe_matrix
from src.sparse_utils import describe_user_activity
from src.minhash import compute_minhash_signatures
from src.lsh import lsh_candidate_pairs
from src.jaccard import verify_candidates_and_write


def parse_args():
    parser = argparse.ArgumentParser(
        description="LSH assignment: full run on all users."
    )
    parser.add_argument(
        "--input_npy",
        type=str,
        default="user_movie_rating.npy",
        help="Path to user_movie_rating.npy file (default: user_movie_rating.npy)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for minhash (required by assignment).",
    )
    parser.add_argument(
        "--num_hashes",
        type=int,
        default=100,
        help="Number of hash functions (signature length). Default: 100.",
    )
    parser.add_argument(
        "--rows_per_band",
        type=int,
        default=5,
        help="Number of rows per band for LSH. Default: 5.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Jaccard similarity threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result.txt",
        help="Output file path for similar user pairs. Default: result.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("==========================================")
    print("  LSH Assignment - FULL RUN")
    print("==========================================")
    print(f"Input file      : {args.input_npy}")
    print(f"Random seed     : {args.seed}")
    print(f"num_hashes (h)  : {args.num_hashes}")
    print(f"rows_per_band r : {args.rows_per_band}")
    print(f"threshold       : {args.threshold}")
    print(f"output file     : {args.output}")
    print("==========================================\n")

    # 1) Load full User x Movie matrix
    print("Loading full User x Movie matrix...")
    R, n_users, n_movies = load_user_movie_matrix(args.input_npy, dtype="int8")
    describe_matrix(R, n_users, n_movies)

    print()
    describe_user_activity(R)

    # 2) Compute minhash signatures for ALL users
    print("\nComputing minhash signatures for ALL users...")
    R_full: csr_matrix = R  # just rename for clarity
    signatures = compute_minhash_signatures(
        R_full,
        num_hashes=args.num_hashes,
        seed=args.seed,
        verbose=True,
    )
    print("Signatures shape:", signatures.shape)

    # 3) LSH banding → candidate pairs
    print("\nRunning LSH banding to get candidate pairs...")
    candidates = lsh_candidate_pairs(
        signatures,
        rows_per_band=args.rows_per_band,
        max_bucket_size=500,
        verbose=True,
    )

    print(f"\nNumber of candidate pairs from LSH: {len(candidates)}")
    print(f"Writing verified pairs (J >= {args.threshold}) to: {args.output}")

    # 4) Verify candidates with exact Jaccard and write to file
    num_written = verify_candidates_and_write(
        R_full,
        candidates,
        threshold=args.threshold,
        output_path=args.output,
        close_each_write=True,   # safer if job gets killed
        verbose=True,
    )

    print("\n==========================================")
    print("  FULL RUN COMPLETED")
    print(f"  Similar pairs written: {num_written}")
    print(f"  Output file: {args.output}")
    print("==========================================")


if __name__ == "__main__":
    main()