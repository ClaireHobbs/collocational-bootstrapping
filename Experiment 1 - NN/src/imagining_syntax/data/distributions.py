#!/usr/bin/env python3
"""Distribution interface for noun-verb pairing experiments.

Two distributions: zipfian (the publication's main sweep) and oneshot (the
α→∞ limit case where each verb is paired deterministically with one noun)."""

import numpy as np


def create_zipfian_distribution(param_value, vocab_size=None, unseen_count=None):
    """Truncated Zipfian over `vocab_size - unseen_count` seen items, zeros for the
    `unseen_count` unseen tail. param_value is the exponent α (== s in the standard
    Zipfian formula P(k) = 1/k^s); α=0 is uniform over the seen portion."""
    if vocab_size is None:
        vocab_size = 40
    if unseen_count is None:
        unseen_count = 10

    seen_vocab_size = vocab_size - unseen_count
    if param_value == 0:
        prob_per_item = 1.0 / seen_vocab_size
        return [prob_per_item] * seen_vocab_size + [0] * unseen_count

    ranks = np.arange(1, seen_vocab_size + 1)
    unnormalized = 1.0 / (ranks ** param_value)
    seen_probs = unnormalized / np.sum(unnormalized)
    return seen_probs.tolist() + [0] * unseen_count


def create_oneshot_distribution(param_value, vocab_size=None, unseen_count=None):
    """Deterministic 1:1 pairing — all probability mass at offset 0 in the seen
    portion. param_value is ignored (the function takes it for parity with the
    zipfian builder so callers can dispatch uniformly)."""
    del param_value  # explicit: oneshot is parameterless
    if vocab_size is None:
        vocab_size = 40
    if unseen_count is None:
        unseen_count = 10

    seen_vocab_size = vocab_size - unseen_count
    return [1.0] + [0.0] * (seen_vocab_size - 1) + [0.0] * unseen_count


def create_distribution(distribution_type, param_value, vocab_size=None, unseen_count=None):
    """Dispatch to the requested distribution. Two types supported: 'zipfian',
    'oneshot'. Raises ValueError on anything else."""
    if distribution_type.lower() == "zipfian":
        return create_zipfian_distribution(param_value, vocab_size, unseen_count)
    if distribution_type.lower() == "oneshot":
        return create_oneshot_distribution(param_value, vocab_size, unseen_count)
    raise ValueError(
        f"Unsupported distribution type: {distribution_type}. "
        "Supported types: 'zipfian', 'oneshot'"
    )


def validate_parameter(distribution_type, param_value):
    """Raise ValueError if param_value is out of range for the given distribution."""
    if distribution_type.lower() == "zipfian":
        if param_value < 0:
            raise ValueError(
                f"Zipfian distribution parameter must be >= 0, got {param_value}"
            )
        return True
    if distribution_type.lower() == "oneshot":
        return True  # parameterless
    raise ValueError(f"Unsupported distribution type: {distribution_type}")


def get_parameter_name(distribution_type):
    """One-letter token used in directory paths (Z for zipfian, O for oneshot)."""
    if distribution_type.lower() == "zipfian":
        return "Z"
    if distribution_type.lower() == "oneshot":
        return "O"
    raise ValueError(f"Unsupported distribution type: {distribution_type}")


def format_param_for_filename(distribution_type, param_value):
    """Filesystem-safe parameter token. Zipfian: '15' for 1.5, '0' for 0.0,
    '150' for 1.50. Oneshot: always '0'."""
    if distribution_type.lower() == "oneshot":
        return "0"
    if distribution_type.lower() == "zipfian":
        if param_value == int(param_value):
            return str(int(param_value))
        param_str = f"{param_value:.2f}".replace(".", "")
        return param_str.lstrip("0") or "0"
    raise ValueError(f"Unsupported distribution type: {distribution_type}")
