"""Patient-level aggregation and small-sample paired inference."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

try:  # SciPy is required by the project, but keep reporting graceful.
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:  # pragma: no cover - exercised only in a reduced environment.
    scipy_wilcoxon = None


@dataclass(frozen=True)
class TestResult:
    statistic: float
    pvalue: float
    n: int
    method: str
    alternative: str = "two-sided"


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    low: float
    high: float
    confidence_level: float
    n_resamples: int
    n: int


def aggregate_patient_level(
    data: pd.DataFrame | Iterable[dict[str, Any]],
    *,
    patient_col: str = "patient",
    group_cols: Sequence[str] = (),
    value_cols: Sequence[str] | None = None,
    agg: str | Callable[[pd.Series], float] = "mean",
) -> pd.DataFrame:
    """Collapse windows/seeds/runs to one row per patient and condition.

    ``group_cols`` should contain experimental factors such as ``method``,
    ``horizon`` or ``variant``. Replicate identifiers must intentionally be
    omitted so they are averaged within patient rather than pseudo-replicated.
    """

    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if patient_col not in frame and patient_col == "patient" and "subject" in frame:
        patient_col = "subject"
    keys = [patient_col, *group_cols]
    missing = [column for column in keys if column not in frame]
    if missing:
        raise ValueError(f"missing grouping columns: {missing}")
    if value_cols is None:
        value_cols = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column not in keys
        ]
    value_cols = list(value_cols)
    if not value_cols:
        raise ValueError("no numeric value columns were selected")
    missing_values = [column for column in value_cols if column not in frame]
    if missing_values:
        raise ValueError(f"missing value columns: {missing_values}")
    if frame.duplicated(keys).any() or len(frame) > frame[keys].drop_duplicates().shape[0]:
        # Expected path: collapse technical repeats within each biological unit.
        pass
    result = (
        frame.groupby(keys, dropna=False, sort=True)[value_cols]
        .agg(agg)
        .reset_index()
    )
    return result


patient_level_aggregation = aggregate_patient_level


def paired_patient_differences(
    data: pd.DataFrame,
    *,
    condition_col: str,
    baseline: Any,
    comparison: Any,
    value_col: str,
    patient_col: str = "patient",
    group_cols: Sequence[str] = (),
    comparison_minus_baseline: bool = True,
) -> pd.DataFrame:
    """Create complete patient-matched differences after patient aggregation."""

    if patient_col not in data and patient_col == "patient" and "subject" in data:
        patient_col = "subject"
    collapsed = aggregate_patient_level(
        data,
        patient_col=patient_col,
        group_cols=[*group_cols, condition_col],
        value_cols=[value_col],
    )
    index = [patient_col, *group_cols]
    wide = collapsed.pivot(index=index, columns=condition_col, values=value_col)
    if baseline not in wide.columns or comparison not in wide.columns:
        raise ValueError("baseline/comparison condition is absent")
    wide = wide.dropna(subset=[baseline, comparison]).reset_index()
    if comparison_minus_baseline:
        wide["difference"] = wide[comparison] - wide[baseline]
    else:
        wide["difference"] = wide[baseline] - wide[comparison]
    return wide


def _paired_differences(first: ArrayLike, second: ArrayLike | None) -> NDArray[np.float64]:
    left = np.asarray(first, dtype=float).ravel()
    if second is None:
        differences = left
    else:
        right = np.asarray(second, dtype=float).ravel()
        if left.shape != right.shape:
            raise ValueError("paired samples must have matching shapes")
        differences = left - right
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        raise ValueError("no finite paired differences")
    return differences


def exact_paired_sign_flip(
    first: ArrayLike,
    second: ArrayLike | None = None,
    *,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    max_exact_n: int = 20,
) -> TestResult:
    """Exact paired randomization test of a zero mean difference.

    ``first - second`` defines the effect when two samples are supplied. Zero
    differences are retained in the reported mean but removed from enumeration
    because flipping them creates redundant permutations.
    """

    differences = _paired_differences(first, second)
    observed = float(np.mean(differences))
    nonzero = differences[~np.isclose(differences, 0.0, atol=0.0, rtol=0.0)]
    n_effective = int(nonzero.size)
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("invalid alternative")
    if n_effective == 0:
        return TestResult(observed, 1.0, int(differences.size), "exact sign-flip", alternative)
    if n_effective > max_exact_n:
        raise ValueError(
            f"exact enumeration needs 2^{n_effective} permutations; "
            f"max_exact_n={max_exact_n}"
        )

    magnitudes = np.abs(nonzero)
    # The denominator does not alter tail ordering; use the full-n denominator
    # so enumerated statistics are directly comparable with the reported mean.
    denominator = differences.size
    permutation_statistics = np.fromiter(
        (
            float(np.dot(signs, magnitudes) / denominator)
            for signs in product((-1.0, 1.0), repeat=n_effective)
        ),
        dtype=float,
        count=2**n_effective,
    )
    tolerance = 1e-12 * max(1.0, abs(observed))
    if alternative == "two-sided":
        extreme = np.abs(permutation_statistics) >= abs(observed) - tolerance
    elif alternative == "greater":
        extreme = permutation_statistics >= observed - tolerance
    else:
        extreme = permutation_statistics <= observed + tolerance
    pvalue = float(np.mean(extreme))
    return TestResult(observed, pvalue, int(differences.size), "exact sign-flip", alternative)


paired_sign_flip_test = exact_paired_sign_flip


def paired_wilcoxon(
    first: ArrayLike,
    second: ArrayLike | None = None,
    *,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> TestResult | None:
    """Wilcoxon signed-rank sensitivity analysis, if SciPy is available."""

    if scipy_wilcoxon is None:
        return None
    differences = _paired_differences(first, second)
    if np.all(differences == 0):
        return TestResult(0.0, 1.0, int(differences.size), "Wilcoxon signed-rank", alternative)
    contains_zero = bool(np.any(differences == 0))
    method = "auto" if contains_zero else ("exact" if differences.size <= 50 else "auto")
    result = scipy_wilcoxon(
        differences,
        alternative=alternative,
        zero_method="wilcox",
        correction=False,
        method=method,
    )
    return TestResult(
        float(result.statistic),
        float(result.pvalue),
        int(differences.size),
        "Wilcoxon signed-rank",
        alternative,
    )


wilcoxon_if_available = paired_wilcoxon


def bootstrap_confidence_interval(
    values: ArrayLike,
    *,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapCI:
    """Patient-resampling percentile bootstrap confidence interval."""

    sample = np.asarray(values, dtype=float).ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("values contain no finite observations")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, sample.size, size=(n_resamples, sample.size))
    if statistic is np.mean:
        bootstrap = sample[indices].mean(axis=1)
    else:
        bootstrap = np.array([statistic(sample[index]) for index in indices], dtype=float)
    alpha = 1.0 - confidence_level
    low, high = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        estimate=float(statistic(sample)),
        low=float(low),
        high=float(high),
        confidence_level=float(confidence_level),
        n_resamples=int(n_resamples),
        n=int(sample.size),
    )


bootstrap_ci = bootstrap_confidence_interval


def paired_bootstrap_ci(
    first: ArrayLike,
    second: ArrayLike,
    **kwargs: Any,
) -> BootstrapCI:
    """Bootstrap the mean paired effect ``first - second`` by patient."""

    return bootstrap_confidence_interval(_paired_differences(first, second), **kwargs)


def holm_adjust(
    p_values: ArrayLike,
    *,
    alpha: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Holm step-down family-wise error correction with NaN preservation."""

    pvalues = np.asarray(p_values, dtype=float)
    if pvalues.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    finite = np.isfinite(pvalues)
    if np.any((pvalues[finite] < 0) | (pvalues[finite] > 1)):
        raise ValueError("p-values must be in [0, 1]")
    adjusted = np.full(pvalues.shape, np.nan, dtype=float)
    reject = np.zeros(pvalues.shape, dtype=bool)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size == 0:
        return adjusted, reject
    order_local = np.argsort(pvalues[finite_indices], kind="mergesort")
    order = finite_indices[order_local]
    ordered_p = pvalues[order]
    m = ordered_p.size
    ordered_adjusted = np.maximum.accumulate(
        np.array([(m - index) * value for index, value in enumerate(ordered_p)])
    )
    ordered_adjusted = np.minimum(1.0, ordered_adjusted)
    adjusted[order] = ordered_adjusted
    reject[finite] = adjusted[finite] <= alpha
    return adjusted, reject


def holm_correction(p_values: ArrayLike, *, alpha: float = 0.05) -> dict[str, NDArray[Any]]:
    """Dictionary-returning convenience wrapper for :func:`holm_adjust`."""

    adjusted, reject = holm_adjust(p_values, alpha=alpha)
    return {"p_adjusted": adjusted, "reject": reject}


def paired_patient_summary(
    first: ArrayLike,
    second: ArrayLike,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict[str, Any]:
    """Bundle effect, CI, exact sign-flip, and Wilcoxon sensitivity results."""

    differences = _paired_differences(first, second)
    ci = bootstrap_confidence_interval(
        differences,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    sign_flip = exact_paired_sign_flip(differences, alternative=alternative)
    wilcoxon = paired_wilcoxon(differences, alternative=alternative)
    return {
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "bootstrap": asdict(ci),
        "sign_flip": asdict(sign_flip),
        "wilcoxon": None if wilcoxon is None else asdict(wilcoxon),
    }
