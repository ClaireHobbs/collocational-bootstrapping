import os
import random
import argparse
from entropy_sentence import generate_sentence, verbs_singular, verbs_plural
from distribution_interface import validate_parameter, get_parameter_name

def flip_verb(sentence, vocab_size=None):
    if vocab_size is None:
        vocab_size = len(verbs_singular)
        
    words = sentence.strip().split()
    verb = words[-1]

    # Use only the first vocab_size verbs
    verbs_sing_subset = verbs_singular[:vocab_size]
    verbs_plur_subset = verbs_plural[:vocab_size]

    if verb in verbs_sing_subset:
        flipped = verbs_plur_subset[verbs_sing_subset.index(verb)]
    elif verb in verbs_plur_subset:
        flipped = verbs_sing_subset[verbs_plur_subset.index(verb)]
    else:
        return None  

    return " ".join(words[:-1] + [flipped])

def generate_minimal_pairs(param_value, distribution_type="geometric", num_pairs=1000,
                          prep_obj_mismatch=True, both_pps_present=True, noun_OOD=False,
                          vocab_size=None, unseen_count=None):
    """
    Generate minimal pairs using the specified distribution type.

    Args:
        param_value: Distribution parameter (C for geometric, Z for Zipfian)
        distribution_type: "geometric" or "zipfian"
        num_pairs: Number of minimal pairs to generate
        prep_obj_mismatch: Make prepositional objects mismatch subject in number
        both_pps_present: Include both prepositional phrases
        noun_OOD: Use unseen noun-verb pairings
        vocab_size: Size of vocabulary
        unseen_count: Number of unseen pairs
    """
    # Validate parameter for the distribution type
    validate_parameter(distribution_type, param_value)

    seen = set()
    pairs = []

    while len(pairs) < num_pairs:
        g = generate_sentence(param_value, distribution_type,
                            prep_obj_mismatch=prep_obj_mismatch,
                            both_pps_present=both_pps_present,
                            noun_OOD=noun_OOD,
                            vocab_size=vocab_size,
                            unseen_count=unseen_count)
        u = flip_verb(g, vocab_size)

        if u and g not in seen:
            seen.add(g)
            pairs.append((g, u))

    random.shuffle(pairs)
    return pairs

def save_pairs(pairs, filename):
    with open(filename, "w") as f:
        for g, u in pairs:
            f.write(f"{g}\t{u}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate minimal pairs for model evaluation')
    parser.add_argument('param_value', type=float,
                       help='Distribution parameter value (C for geometric 0-1, Z for Zipfian ≥0, ignored for oneshot)')
    parser.add_argument('--distribution_type', type=str, default='geometric',
                       choices=['geometric', 'zipfian', 'oneshot'],
                       help='Distribution type: geometric (default), zipfian, or oneshot')
    parser.add_argument('--output_dir', type=str, default='entropy_eval',
                       help='Output directory (default: entropy_eval)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output filename for minimal pairs')
    parser.add_argument('--num_pairs', type=int, default=1000,
                       help='Number of minimal pairs to generate (default: 1000)')
    parser.add_argument('--noun_OOD', action='store_true',
                       help='Use unseen noun-verb pairings (out-of-distribution)')
    parser.add_argument('--prep_obj_mismatch', action='store_true',
                       help='Make prepositional objects mismatch subject in number')
    parser.add_argument('--both_pps_present', action='store_true', default=True,
                       help='Include both prepositional phrases (default: True)')
    parser.add_argument('--vocab_size', type=int, default=40,
                       help='Vocabulary size (default: 40)')
    parser.add_argument('--unseen_count', type=int, default=10,
                       help='Number of unseen pairs (default: 10)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate minimal pairs
    pairs = generate_minimal_pairs(
        param_value=args.param_value,
        distribution_type=args.distribution_type,
        num_pairs=args.num_pairs,
        prep_obj_mismatch=args.prep_obj_mismatch,
        both_pps_present=args.both_pps_present,
        noun_OOD=args.noun_OOD,
        vocab_size=args.vocab_size,
        unseen_count=args.unseen_count
    )

    # Save pairs
    output_path = os.path.join(args.output_dir, args.output_file)
    save_pairs(pairs, output_path)

    param_name = get_parameter_name(args.distribution_type)
    print(f"Generated {len(pairs)} minimal pairs with {args.distribution_type} distribution: {param_name}={args.param_value}")
    print(f"Settings: noun_OOD={args.noun_OOD}, prep_obj_mismatch={args.prep_obj_mismatch}")
    print(f"Saved to: {output_path}")
    

