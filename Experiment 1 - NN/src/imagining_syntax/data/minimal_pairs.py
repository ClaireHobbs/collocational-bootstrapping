import os
import random
from imagining_syntax.data.sentences import generate_sentence, verbs_singular, verbs_plural
from imagining_syntax.data.distributions import validate_parameter, get_parameter_name
from imagining_syntax.utils.seed import set_global_seed

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

def generate_minimal_pairs(param_value, distribution_type="zipfian", num_pairs=1000,
                          prep_obj_mismatch=True, both_pps_present=True, noun_OOD=False,
                          vocab_size=None, unseen_count=None):
    """
    Generate minimal pairs using the specified distribution type.

    Args:
        param_value: Distribution parameter (Z for zipfian or ignored for oneshot)
        distribution_type: "zipfian" or "oneshot"
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

def _add_args(parser):
    """Attach the minimal-pairs generator arguments to `parser`."""
    parser.add_argument('param_value', type=float,
                       help='Distribution parameter value (Z for zipfian ≥0; ignored when --oneshot is set)')
    parser.add_argument('--oneshot', action='store_true',
                       help='Use oneshot (deterministic 1:1) distribution instead of zipfian')
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
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible pair sampling (default: unseeded)')


def add_parser(subparsers):
    """Register the `imsyn gen pairs` subparser."""
    p = subparsers.add_parser(
        "pairs",
        help="Generate minimal pairs for model evaluation.",
        description='Generate minimal pairs for model evaluation',
    )
    _add_args(p)
    p.set_defaults(func=main)
    return p


def main(args):
    """Generate the minimal pairs with parsed args."""
    if args.seed is not None:
        set_global_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    distribution_type = "oneshot" if args.oneshot else "zipfian"
    pairs = generate_minimal_pairs(
        param_value=args.param_value,
        distribution_type=distribution_type,
        num_pairs=args.num_pairs,
        prep_obj_mismatch=args.prep_obj_mismatch,
        both_pps_present=args.both_pps_present,
        noun_OOD=args.noun_OOD,
        vocab_size=args.vocab_size,
        unseen_count=args.unseen_count
    )

    output_path = os.path.join(args.output_dir, args.output_file)
    save_pairs(pairs, output_path)

    param_name = get_parameter_name(distribution_type)
    print(f"Generated {len(pairs)} minimal pairs with {distribution_type} distribution: {param_name}={args.param_value}")
    print(f"Settings: noun_OOD={args.noun_OOD}, prep_obj_mismatch={args.prep_obj_mismatch}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate minimal pairs for model evaluation')
    _add_args(parser)
    args = parser.parse_args()
    main(args)
