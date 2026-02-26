
# generate_entropy_dataset.py

import os
from sklearn.model_selection import train_test_split
from entropy_sentence import generate_sentence
from distribution_interface import validate_parameter, get_parameter_name

def generate_unique_sentences(count, param_value, distribution_type="geometric", vocab_size=None, unseen_count=None):
    """
    Generate unique sentences using the specified distribution type.

    Args:
        count: Number of sentences to generate
        param_value: Distribution parameter (C for geometric, Z for Zipfian)
        distribution_type: "geometric" or "zipfian"
        vocab_size: Size of vocabulary
        unseen_count: Number of unseen pairs
    """
    # Validate parameter for the distribution type
    validate_parameter(distribution_type, param_value)

    sentences = set()
    attempts = 0
    max_attempts = count * 10  # Avoid infinite loops

    while len(sentences) < count and attempts < max_attempts:
        sentence = generate_sentence(param_value, distribution_type, prep_obj_mismatch=False,
                                   both_pps_present=False, noun_OOD=False,
                                   vocab_size=vocab_size, unseen_count=unseen_count)
        sentences.add(sentence)
        attempts += 1

    if len(sentences) < count:
        print(f"Warning: Only generated {len(sentences)} unique sentences after {attempts} attempts.")

    return list(sentences)

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate entropy dataset with specified distribution parameter')
    parser.add_argument('param_value', type=float,
                       help='Distribution parameter value (C for geometric 0-1, Z for Zipfian ≥0, ignored for oneshot)')
    parser.add_argument('--distribution_type', type=str, default='geometric',
                       choices=['geometric', 'zipfian', 'oneshot'],
                       help='Distribution type: geometric (default), zipfian, or oneshot')
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

    args = parser.parse_args()

    # Generate unique sentences
    all_sentences = generate_unique_sentences(
        args.sentence_count,
        args.param_value,
        args.distribution_type,
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

    param_name = get_parameter_name(args.distribution_type)
    print(f"Generated dataset with {args.distribution_type} distribution: {param_name}={args.param_value}")
    print(f"Saved {len(all_sentences)} sentences to {args.output_dir}/")
