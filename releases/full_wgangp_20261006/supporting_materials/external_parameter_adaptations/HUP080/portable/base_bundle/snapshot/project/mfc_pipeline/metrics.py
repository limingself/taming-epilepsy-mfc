"""Prediction and distribution-aware evaluation metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import welch
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from .network import plv_matrix


def _samples_by_feature(values: ArrayLike, time_axis: int = 0) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("data must be one- or two-dimensional")
    array = np.moveaxis(array, time_axis, 0)
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("data require at least two samples and one feature")
    if not np.all(np.isfinite(array)):
        raise ValueError("data contain NaN or infinite values")
    return array


def fixed_random_projections(
    n_features: int,
    *,
    n_projections: int = 128,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Create a deterministic bank of unit vectors for sliced Wasserstein."""

    if n_features < 1 or n_projections < 1:
        raise ValueError("n_features and n_projections must be positive")
    generator = np.random.default_rng(seed)
    projections = generator.normal(size=(n_projections, n_features))
    norms = np.linalg.norm(projections, axis=1, keepdims=True)
    # A Gaussian draw is almost surely nonzero; this also covers mocked RNGs.
    zero = norms[:, 0] == 0
    if np.any(zero):
        projections[zero, 0] = 1.0
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
    return projections / norms


def sliced_wasserstein_distance(
    first: ArrayLike,
    second: ArrayLike,
    *,
    projections: ArrayLike | None = None,
    n_projections: int = 128,
    seed: int = 0,
    time_axis: int = 0,
    return_projection_distances: bool = False,
) -> float | tuple[float, NDArray[np.float64]]:
    """Estimate multivariate Wasserstein-1 using fixed one-dimensional slices.

    Pass the same explicit ``projections`` bank to every method comparison, or
    use the same ``seed`` and feature count, to eliminate projection Monte Carlo
    noise from paired comparisons.
    """

    x = _samples_by_feature(first, time_axis=time_axis)
    y = _samples_by_feature(second, time_axis=time_axis)
    if x.shape[1] != y.shape[1]:
        raise ValueError("distributions must have the same feature count")
    if projections is None:
        directions = fixed_random_projections(
            x.shape[1], n_projections=n_projections, seed=seed
        )
    else:
        directions = np.asarray(projections, dtype=float)
        if directions.ndim != 2 or directions.shape[1] != x.shape[1]:
            raise ValueError("projections must have shape (n_projections, n_features)")
        if not np.all(np.isfinite(directions)):
            raise ValueError("projections contain NaN or infinite values")
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("projection vectors must be nonzero")
        directions = directions / norms

    projected_x = x @ directions.T
    projected_y = y @ directions.T
    distances = np.array(
        [
            wasserstein_distance(projected_x[:, index], projected_y[:, index])
            for index in range(directions.shape[0])
        ],
        dtype=float,
    )
    estimate = float(np.mean(distances))
    return (estimate, distances) if return_projection_distances else estimate


def multivariate_energy_distance(
    first: ArrayLike,
    second: ArrayLike,
    *,
    time_axis: int = 0,
    squared: bool = False,
) -> float:
    """Compute the biased, non-negative multivariate energy distance.

    The empirical expression includes diagonal within-sample pairs, which gives
    an exactly zero value when the two empirical samples are identical. By
    convention this function returns the square root of the energy statistic;
    set ``squared=True`` to return the statistic itself.
    """

    x = _samples_by_feature(first, time_axis=time_axis)
    y = _samples_by_feature(second, time_axis=time_axis)
    if x.shape[1] != y.shape[1]:
        raise ValueError("distributions must have the same feature count")
    statistic = (
        2.0 * float(np.mean(cdist(x, y)))
        - float(np.mean(cdist(x, x)))
        - float(np.mean(cdist(y, y)))
    )
    statistic = max(0.0, statistic)
    return statistic if squared else float(np.sqrt(statistic))


energy_distance = multivariate_energy_distance


def covariance_distance(
    first: ArrayLike,
    second: ArrayLike,
    *,
    time_axis: int = 0,
    normalization: Literal["none", "reference", "symmetric"] = "symmetric",
) -> float:
    """Frobenius distance between empirical channel covariance matrices."""

    x = _samples_by_feature(first, time_axis=time_axis)
    y = _samples_by_feature(second, time_axis=time_axis)
    if x.shape[1] != y.shape[1]:
        raise ValueError("distributions must have the same feature count")
    cov_x = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    cov_y = np.atleast_2d(np.cov(y, rowvar=False, ddof=1))
    numerator = float(np.linalg.norm(cov_x - cov_y, ord="fro"))
    if normalization == "none":
        return numerator
    if normalization == "reference":
        denominator = float(np.linalg.norm(cov_y, ord="fro"))
    elif normalization == "symmetric":
        denominator = 0.5 * float(
            np.linalg.norm(cov_x, ord="fro") + np.linalg.norm(cov_y, ord="fro")
        )
    else:
        raise ValueError("normalization must be none, reference, or symmetric")
    return numerator / max(denominator, np.finfo(float).eps)


def psd_distance(
    first: ArrayLike,
    second: ArrayLike,
    sfreq: float,
    *,
    fmin: float = 1.0,
    fmax: float | None = 50.0,
    nperseg: int | None = None,
    log_power: bool = True,
    time_axis: int = 0,
) -> float:
    """Root-mean-square distance between channel-wise Welch spectra."""

    x = _samples_by_feature(first, time_axis=time_axis)
    y = _samples_by_feature(second, time_axis=time_axis)
    if x.shape[1] != y.shape[1]:
        raise ValueError("signals must have the same channel count")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if fmin < 0 or (fmax is not None and (fmax <= fmin or fmax > sfreq / 2 + 1e-12)):
        raise ValueError("invalid frequency range")
    if nperseg is None:
        nperseg = min(256, x.shape[0], y.shape[0])
    if nperseg < 2 or nperseg > min(x.shape[0], y.shape[0]):
        raise ValueError("nperseg is incompatible with the signal lengths")

    frequencies, psd_x = welch(x, fs=sfreq, nperseg=nperseg, axis=0)
    _, psd_y = welch(y, fs=sfreq, nperseg=nperseg, axis=0)
    keep = frequencies >= fmin
    if fmax is not None:
        keep &= frequencies <= fmax
    if not np.any(keep):
        raise ValueError("frequency range contains no Welch bins")
    psd_x, psd_y = psd_x[keep], psd_y[keep]
    if log_power:
        scale = max(float(np.max(psd_x)), float(np.max(psd_y)), 1.0)
        floor = scale * 1e-12
        psd_x = 10.0 * np.log10(psd_x + floor)
        psd_y = 10.0 * np.log10(psd_y + floor)
    return float(np.sqrt(np.mean(np.square(psd_x - psd_y))))


def plv_matrix_distance(
    first_plv: ArrayLike,
    second_plv: ArrayLike,
    *,
    relative: bool = False,
) -> float:
    """Upper-triangle RMS distance between two PLV matrices."""

    first = np.asarray(first_plv, dtype=float)
    second = np.asarray(second_plv, dtype=float)
    if (
        first.ndim != 2
        or first.shape[0] != first.shape[1]
        or first.shape != second.shape
    ):
        raise ValueError("PLV matrices must be square and have matching shapes")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("PLV matrices contain NaN or infinite values")
    if first.shape[0] == 1:
        return 0.0
    upper = np.triu_indices(first.shape[0], k=1)
    numerator = float(np.sqrt(np.mean(np.square(first[upper] - second[upper]))))
    if not relative:
        return numerator
    denominator = float(np.sqrt(np.mean(np.square(second[upper]))))
    return numerator / max(denominator, np.finfo(float).eps)


def plv_distance(
    first: ArrayLike,
    second: ArrayLike,
    *,
    time_axis: int = 0,
    relative: bool = False,
) -> float:
    """Compute PLV matrices from signals and return their matrix distance."""

    return plv_matrix_distance(
        plv_matrix(first, time_axis=time_axis),
        plv_matrix(second, time_axis=time_axis),
        relative=relative,
    )


def distribution_metrics(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    sfreq: float,
    projections: ArrayLike | None = None,
    n_projections: int = 128,
    projection_seed: int = 0,
    fmin: float = 1.0,
    fmax: float | None = 50.0,
    psd_nperseg: int | None = None,
    time_axis: int = 0,
) -> dict[str, float]:
    """Compute the pre-specified complementary distribution distances."""

    return {
        "sliced_wasserstein": float(
            sliced_wasserstein_distance(
                candidate,
                reference,
                projections=projections,
                n_projections=n_projections,
                seed=projection_seed,
                time_axis=time_axis,
            )
        ),
        "energy": multivariate_energy_distance(
            candidate, reference, time_axis=time_axis
        ),
        "covariance": covariance_distance(
            candidate, reference, time_axis=time_axis
        ),
        "psd": psd_distance(
            candidate,
            reference,
            sfreq,
            fmin=fmin,
            fmax=fmax,
            nperseg=psd_nperseg,
            time_axis=time_axis,
        ),
        "plv": plv_distance(candidate, reference, time_axis=time_axis),
    }


def normalized_rmse(
    truth: ArrayLike,
    prediction: ArrayLike,
    *,
    normalization: Literal["std", "range", "rms"] = "std",
) -> float:
    """Root mean squared error normalized by a scale of the ground truth."""

    actual = np.asarray(truth, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("truth and prediction must have the same non-empty shape")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("truth/prediction contain NaN or infinite values")
    rmse = float(np.sqrt(np.mean(np.square(predicted - actual))))
    if normalization == "std":
        scale = float(np.std(actual))
    elif normalization == "range":
        scale = float(np.ptp(actual))
    elif normalization == "rms":
        scale = float(np.sqrt(np.mean(np.square(actual))))
    else:
        raise ValueError("normalization must be std, range, or rms")
    if scale <= np.finfo(float).eps:
        return 0.0 if rmse <= np.finfo(float).eps else float("inf")
    return rmse / scale


nrmse = normalized_rmse


def pearson_correlation(truth: ArrayLike, prediction: ArrayLike) -> float:
    """Pearson correlation after flattening matching arrays."""

    actual = np.asarray(truth, dtype=float).ravel()
    predicted = np.asarray(prediction, dtype=float).ravel()
    if actual.shape != predicted.shape or actual.size < 2:
        raise ValueError("truth and prediction must contain at least two paired values")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("truth/prediction contain NaN or infinite values")
    if np.std(actual) <= np.finfo(float).eps or np.std(predicted) <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def prediction_metrics(
    truth: ArrayLike,
    prediction: ArrayLike,
) -> dict[str, float]:
    """Return global and channel-balanced NRMSE/correlation metrics."""

    actual = np.asarray(truth, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("truth and prediction must have matching shapes")
    if actual.ndim == 1:
        actual = actual[:, None]
        predicted = predicted[:, None]
    if actual.ndim != 2:
        raise ValueError("prediction arrays must be time-by-channel")

    per_channel_nrmse = np.array(
        [normalized_rmse(actual[:, index], predicted[:, index]) for index in range(actual.shape[1])]
    )
    per_channel_correlation = np.array(
        [pearson_correlation(actual[:, index], predicted[:, index]) for index in range(actual.shape[1])]
    )
    finite_nrmse = per_channel_nrmse[np.isfinite(per_channel_nrmse)]
    finite_correlation = per_channel_correlation[np.isfinite(per_channel_correlation)]
    return {
        "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
        "nrmse": normalized_rmse(actual, predicted),
        "correlation": pearson_correlation(actual, predicted),
        "mean_channel_nrmse": (
            float(np.mean(finite_nrmse)) if finite_nrmse.size else float("nan")
        ),
        "mean_channel_correlation": (
            float(np.mean(finite_correlation))
            if finite_correlation.size
            else float("nan")
        ),
    }


prediction_nrmse_correlation = prediction_metrics
