from __future__ import annotations

import hashlib
import math
import random
from statistics import mean, median, stdev
from typing import Callable, Sequence


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def sample_std(values: Sequence[float]) -> float | None:
    return stdev(values) if len(values) >= 2 else None


def bootstrap_ci(
    values: Sequence[float],
    *,
    label: str,
    samples: int = 5000,
    statistic: Callable[[Sequence[float]], float] = mean,
) -> list[float] | None:
    if not values:
        return None
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    generator = random.Random(seed)
    estimates = sorted(
        statistic([values[generator.randrange(len(values))] for _ in values])
        for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [lower, upper]


def describe(values: Sequence[float], *, label: str) -> dict[str, object]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "standard_deviation": None,
            "median": None,
            "min": None,
            "max": None,
            "bootstrap_95_ci": None,
        }
    return {
        "n": len(values),
        "mean": mean(values),
        "standard_deviation": sample_std(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "bootstrap_95_ci": bootstrap_ci(values, label=label),
    }


def paired_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    label: str,
) -> dict[str, object]:
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    difference_std = sample_std(differences)
    standardized = (
        None
        if not differences or difference_std in (None, 0)
        else mean(differences) / difference_std
    )
    return {
        "n_pairs": len(differences),
        "mean_paired_difference": None if not differences else mean(differences),
        "median_paired_difference": None if not differences else median(differences),
        "bootstrap_95_ci": bootstrap_ci(differences, label=label),
        "standardized_paired_difference": standardized,
        "exploratory_exact_sign_flip_p": exact_sign_flip_p(differences),
        "differences_by_replicate": differences,
        "inference_label": "exploratory pilot analysis; not confirmatory",
    }


def exact_sign_flip_p(differences: Sequence[float]) -> float | None:
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0 if differences else None
    if len(nonzero) > 20:
        return None
    observed = abs(mean(nonzero))
    extreme = 0
    total = 2 ** len(nonzero)
    for mask in range(total):
        permuted = [
            value if mask & (1 << index) else -value
            for index, value in enumerate(nonzero)
        ]
        if abs(mean(permuted)) + 1e-15 >= observed:
            extreme += 1
    return extreme / total


def entropy(counts: Sequence[int], *, normalized: bool = False) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    value = -sum(
        (count / total) * math.log2(count / total) for count in counts if count > 0
    )
    if normalized and len(counts) > 1:
        return value / math.log2(len(counts))
    return value


def l1_distance(left: dict[str, int], right: dict[str, int]) -> int:
    return sum(abs(left.get(key, 0) - right.get(key, 0)) for key in set(left) | set(right))
