"""Dataset generation for PCFG sentence corpora.

Run via ``imsyn gen dataset`` or ``python -m imagining_syntax.data.generate``.
"""

import os
from sklearn.model_selection import train_test_split
from imagining_syntax.data.sentences import generate_sentence
from imagining_syntax.data.distributions import validate_parameter, get_parameter_name
from imagining_syntax.utils.seed import set_global_seed

def generate_unique_sentences(count, param_value, distribution_type="zipfian", vocab_size=None, unseen_count=None):
    """
    Generate unique sentences using the specified distribution type.

    Args:
        count: Number of sentences to generate
        param_value: Distribution parameter (Z for zipfian or ignored for oneshot)
        distribution_type: "zipfian" or "oneshot"
        vocab_size: Size of vocabulary
        unseen_count: Number of unseen pairs
    """
    # Validate parameter for the distribution type
    validate_parameter(distribution_type, param_value)

    # Use a dict (insertion-ordered) rather than a set: Python's str hash is
    # randomized per-process, so set iteration order is nondeterministic across
    # runs even with the same seed. Dict preserves insertion order in 3.7+.
    sentences = {}
    attempts = 0
    max_attempts = count * 10  # Avoid infinite loops

    while len(sentences) < count and attempts < max_attempts:
        sentence = generate_sentence(param_value, distribution_type, prep_obj_mismatch=False,
                                   both_pps_present=False, noun_OOD=False,
                                   vocab_size=vocab_size, unseen_count=unseen_count)
        sentences[sentence] = None
        attempts += 1

    if len(sentences) < count:  # pragma: no cover - informational warning when 10× attempt budget is insufficient; not a behavioral path
        print(f"Warning: Only generated {len(sentences)} unique sentences after {attempts} attempts.")

    return list(sentences.keys())

def split_and_save(sentences, train_path, val_path, test_path):
    # Split into 80% train, 10% val, 10% test
    train, temp = train_test_split(sentences, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    def write_to_file(path, data):
        with open(path, 'w') as f:
            for line in data:
                f.write(line + '\n')

    write_to_file(train_path, train)
    write_to_file(val_path, val)
    write_to_file(test_path, test)

def _add_args(parser):
    """Attach the dataset-generator arguments to `parser`."""
    parser.add_argument('param_value', type=float,
                       help='Distribution parameter value (Z for zipfian ≥0; ignored when --oneshot is set)')
    parser.add_argument('--oneshot', action='store_true',
                       help='Use oneshot (deterministic 1:1) distribution instead of zipfian')
    parser.add_argument('--output_dir', type=str, default='entropy_data',
                       help='Output directory for datasets (default: entropy_data)')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Output filename for training data')
    parser.add_argument('--val_file', type=str, required=True,
                       help='Output filename for validation data')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Output filename for test data')
    parser.add_argument('--sentence_count', type=int, default=12000,
                       help='Number of sentences to generate (default: 12000)')
    parser.add_argument('--vocab_size', type=int, default=40,
                       help='Vocabulary size (default: 40)')
    parser.add_argument('--unseen_count', type=int, default=10,
                       help='Number of unseen pairs (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible sentence sampling (default: unseeded)')


def add_parser(subparsers):
    """Register the `imsyn gen dataset` subparser."""
    p = subparsers.add_parser(
        "dataset",
        help="Generate a PCFG training/val/test dataset.",
        description='Generate entropy dataset with specified distribution parameter',
    )
    _add_args(p)
    p.set_defaults(func=main)
    return p


def main(args):
    """Generate the dataset with parsed args."""
    if args.seed is not None:
        set_global_seed(args.seed)

    distribution_type = "oneshot" if args.oneshot else "zipfian"
    all_sentences = generate_unique_sentences(
        args.sentence_count,
        args.param_value,
        distribution_type,
        args.vocab_size,
        args.unseen_count
    )

    # Ensure output folder exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save splits to files
    split_and_save(
        all_sentences,
        os.path.join(args.output_dir, args.train_file),
        os.path.join(args.output_dir, args.val_file),
        os.path.join(args.output_dir, args.test_file)
    )

    param_name = get_parameter_name(distribution_type)
    print(f"Generated dataset with {distribution_type} distribution: {param_name}={args.param_value}")
    print(f"Saved {len(all_sentences)} sentences to {args.output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate entropy dataset with specified distribution parameter')
    _add_args(parser)
    args = parser.parse_args()
    main(args)
