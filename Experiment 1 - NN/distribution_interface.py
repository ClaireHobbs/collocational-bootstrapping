#!/usr/bin/env python3
"""
Unified distribution interface supporting both geometric and Zipfian distributions.
This module provides a consistent API for different distribution types used in
noun-verb pairing experiments.
"""

import numpy as np

def create_geometric_distribution(param_value, vocab_size=None, unseen_count=None):
    """
    Create a truncated geometric distribution (original implementation).

    Args:
        param_value: C parameter (0 <= C <= 1)
        vocab_size: Size of vocabulary (default: 40)
        unseen_count: Number of unseen pairs (default: 10)

    Returns:
        List of probabilities for each position
    """
    if vocab_size is None:
        vocab_size = 40
    if unseen_count is None:
        unseen_count = 10

    C = param_value

    if C == 1:
        starting_value = 1/vocab_size
    else:
        starting_value = (1 - C)/(1 - C**vocab_size)

    distribution = []

    for _ in range(vocab_size):
        distribution.append(starting_value)
        starting_value = starting_value * C

    # Withhold unseen_count nouns from each verb as always unseen
    distribution = distribution[:-unseen_count]
    distribution = distribution + [0] * unseen_count

    return distribution

def create_zipfian_distribution(param_value, vocab_size=None, unseen_count=None):
    """
    Create a Zipfian distribution with concentration parameter Z.

    Args:
        param_value: Z parameter (Z >= 0, equivalent to s in standard Zipf equation P(k) = 1/k^s)
                    - Z = 0: Uniform distribution
                    - Z = 1: Classic Zipf distribution
                    - Z > 1: More concentrated on first items
        vocab_size: Size of vocabulary (default: 40)
        unseen_count: Number of unseen pairs (default: 10)

    Returns:
        List of probabilities for each position

    Note:
        The Z parameter corresponds to the exponent s in the standard Zipfian formula:
        P(k) = (1/k^s) / H_N,s where s = Z
    """
    if vocab_size is None:
        vocab_size = 40
    if unseen_count is None:
        unseen_count = 10

    Z = param_value

    if Z == 0:
        # Uniform distribution (equivalent to C = 1 in geometric)
        prob_per_item = 1.0 / (vocab_size - unseen_count)
        distribution = [prob_per_item] * (vocab_size - unseen_count) + [0] * unseen_count
        return distribution

    # Calculate Zipfian probabilities for seen pairs only
    seen_vocab_size = vocab_size - unseen_count
    ranks = np.arange(1, seen_vocab_size + 1)  # [1, 2, 3, ..., seen_vocab_size]
    unnormalized_probs = 1.0 / (ranks ** Z)

    # Normalize to sum to 1
    normalization_constant = np.sum(unnormalized_probs)
    seen_probs = unnormalized_probs / normalization_constant

    # Combine seen probabilities with zeros for unseen pairs
    distribution = seen_probs.tolist() + [0] * unseen_count

    return distribution

def create_oneshot_distribution(param_value, vocab_size=None, unseen_count=None):
    """
    Create a one-shot (deterministic) distribution for one-to-one noun-verb pairing.

    In this distribution, each noun is deterministically paired with exactly one verb
    (noun_index = verb_index). This creates perfect one-to-one correspondence while
    still allowing sentence variation through prepositional phrase construction.

    Args:
        param_value: Ignored (included for API consistency). Always creates [1, 0, 0, ...]
        vocab_size: Size of vocabulary (default: 40)
        unseen_count: Number of unseen pairs (default: 10)

    Returns:
        List of probabilities: [1.0, 0.0, 0.0, ..., 0.0]

    Note:
        This is equivalent to geometric distribution with C=0, but with clearer semantics
        for one-shot learning experiments. Each seen noun-verb pair appears in a
        deterministic one-to-one mapping.
    """
    if vocab_size is None:
        vocab_size = 40
    if unseen_count is None:
        unseen_count = 10

    # Perfect deterministic pairing: probability 1 at offset 0, 0 everywhere else
    seen_vocab_size = vocab_size - unseen_count
    distribution = [1.0] + [0.0] * (seen_vocab_size - 1) + [0.0] * unseen_count

    return distribution

def create_distribution(distribution_type, param_value, vocab_size=None, unseen_count=None):
    """
    Unified interface for creating distributions.

    Args:
        distribution_type: "geometric", "zipfian", or "oneshot"
        param_value: C (for geometric), Z (for Zipfian), or ignored (for oneshot)
        vocab_size: Size of vocabulary (default: 40)
        unseen_count: Number of unseen pairs (default: 10)

    Returns:
        List of probabilities for each position

    Raises:
        ValueError: If distribution_type is not supported
    """
    if distribution_type.lower() == "geometric":
        return create_geometric_distribution(param_value, vocab_size, unseen_count)
    elif distribution_type.lower() == "zipfian":
        return create_zipfian_distribution(param_value, vocab_size, unseen_count)
    elif distribution_type.lower() == "oneshot":
        return create_oneshot_distribution(param_value, vocab_size, unseen_count)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}. "
                        "Supported types: 'geometric', 'zipfian', 'oneshot'")

def validate_parameter(distribution_type, param_value):
    """
    Validate parameter value for the given distribution type.

    Args:
        distribution_type: "geometric", "zipfian", or "oneshot"
        param_value: C or Z parameter (ignored for oneshot)

    Returns:
        bool: True if valid

    Raises:
        ValueError: If parameter is invalid for the distribution type
    """
    if distribution_type.lower() == "geometric":
        if not (0 <= param_value <= 1):
            raise ValueError(f"Geometric distribution parameter C must be in range [0, 1], got {param_value}")
    elif distribution_type.lower() == "zipfian":
        if param_value < 0:
            raise ValueError(f"Zipfian distribution parameter Z must be >= 0, got {param_value}")
    elif distribution_type.lower() == "oneshot":
        # No validation needed - param_value is ignored for oneshot
        pass
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

    return True

def get_parameter_name(distribution_type):
    """Get the standard parameter name for a distribution type."""
    if distribution_type.lower() == "geometric":
        return "C"
    elif distribution_type.lower() == "zipfian":
        return "Z"
    elif distribution_type.lower() == "oneshot":
        return "O"
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

def format_param_for_filename(distribution_type, param_value):
    """Format parameter value for use in filenames."""
    param_name = get_parameter_name(distribution_type)

    if distribution_type.lower() == "geometric":
        # Keep original C formatting: 0.5 -> 05, 0.05 -> 005
        param_str = f"{param_value:.2f}".replace(".", "")
        return param_str.lstrip("0") or "0"
    elif distribution_type.lower() == "zipfian":
        # Similar formatting for Z values
        if param_value == int(param_value):
            return str(int(param_value))
        else:
            param_str = f"{param_value:.2f}".replace(".", "")
            return param_str.lstrip("0") or "0"
    elif distribution_type.lower() == "oneshot":
        # Oneshot doesn't use param_value, always return "0"
        return "0"
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

def parse_experiment_directory(dir_name):
    """
    Parse experiment directory name to extract distribution type and parameter.

    Args:
        dir_name: Directory name (e.g., "C_50_20250914_145306", "Z_15_20250914_145306", or "O_0_20250914_145306")

    Returns:
        tuple: (distribution_type, param_value) or (None, None) if not parseable
    """
    parts = dir_name.split('_')
    if len(parts) < 3:
        return None, None

    # Handle two different formats:
    # Format 1: "C_50_timestamp" or "Z_15_timestamp" or "O_0_timestamp" (param in second part)
    # Format 2: "C50_timestamp" or "Z15_timestamp" or "O0_timestamp" (param in first part)

    param_part = parts[0]
    param_str = None

    if param_part.startswith('C') and len(param_part) > 1:
        # Format 2: "C50_timestamp"
        distribution_type = "geometric"
        param_str = param_part[1:]  # Remove 'C' prefix
    elif param_part.startswith('Z') and len(param_part) > 1:
        # Format 2: "Z15_timestamp"
        distribution_type = "zipfian"
        param_str = param_part[1:]  # Remove 'Z' prefix
    elif param_part.startswith('O') and len(param_part) > 1:
        # Format 2: "O0_timestamp"
        distribution_type = "oneshot"
        param_str = param_part[1:]  # Remove 'O' prefix
    elif param_part == 'C' and len(parts) > 1:
        # Format 1: "C_50_timestamp"
        distribution_type = "geometric"
        param_str = parts[1]
    elif param_part == 'Z' and len(parts) > 1:
        # Format 1: "Z_15_timestamp"
        distribution_type = "zipfian"
        param_str = parts[1]
    elif param_part == 'O' and len(parts) > 1:
        # Format 1: "O_0_timestamp"
        distribution_type = "oneshot"
        param_str = parts[1]
    else:
        return None, None

    # Parse parameter value based on distribution type
    try:
        if distribution_type == "geometric":
            if param_str == "0":
                param_value = 0.0
            else:
                # Convert back: "05" -> 0.5, "005" -> 0.05, "50" -> 0.5
                if len(param_str) == 1:
                    param_value = float(f"0.0{param_str}")
                elif len(param_str) == 2 and param_str[0] != '0':
                    param_value = float(f"0.{param_str}")
                else:
                    param_value = float(f"0.{param_str}")
            return "geometric", param_value

        elif distribution_type == "zipfian":
            if param_str == "0":
                param_value = 0.0
            elif '.' in param_str:
                param_value = float(param_str)
            else:
                # Convert back: "10" -> 1.0, "15" -> 1.5, "150" -> 1.5
                if len(param_str) == 1:
                    param_value = float(param_str)
                elif len(param_str) == 2:
                    param_value = float(f"{param_str[0]}.{param_str[1]}")
                else:  # len >= 3, handle like "150" -> 1.50
                    param_value = float(f"{param_str[0]}.{param_str[1:]}")
            return "zipfian", param_value

        elif distribution_type == "oneshot":
            # Oneshot always uses param_value 0 (parameter is ignored)
            return "oneshot", 0.0

    except (ValueError, IndexError):
        return None, None

    return None, None

if __name__ == "__main__":
    # Test the interface
    print("Testing Distribution Interface")
    print("=" * 50)

    # Test geometric distributions
    print("\nGeometric Distributions:")
    for C in [0.0, 0.5, 1.0]:
        dist = create_distribution("geometric", C, vocab_size=10, unseen_count=2)
        print(f"C = {C}: {[f'{p:.3f}' for p in dist[:5]]}... (sum = {sum(dist):.3f})")

    # Test Zipfian distributions
    print("\nZipfian Distributions:")
    for Z in [0.0, 1.0, 2.0]:
        dist = create_distribution("zipfian", Z, vocab_size=10, unseen_count=2)
        print(f"Z = {Z}: {[f'{p:.3f}' for p in dist[:5]]}... (sum = {sum(dist):.3f})")

    # Test filename formatting
    print("\nFilename Formatting:")
    print(f"Geometric C=0.5: {format_param_for_filename('geometric', 0.5)}")
    print(f"Zipfian Z=1.5: {format_param_for_filename('zipfian', 1.5)}")

    # Test directory parsing
    print("\nDirectory Parsing:")
    test_dirs = ["C_50_20250914_145306", "Z_15_20250914_145306"]
    for dir_name in test_dirs:
        dist_type, param = parse_experiment_directory(dir_name)
        print(f"{dir_name} -> {dist_type}, {param}")