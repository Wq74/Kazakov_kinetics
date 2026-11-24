import numpy as np
from scipy.optimize import curve_fit

from models import MODELS, clip100


# Add a 100% cap to a model function.
def with_cap(func):
    def wrapped(t, *params):
        return clip100(func(t, *params))

    return wrapped


# Leading coefficients that can scale the starting curve below the cap.
CAP_AMPLITUDES = {
    "Zero order": 1,
    "Higuchi": 1,
    "Korsmeyer-Peppas": 1,
    "Peppas-Sahlin": 2,
}
FIT_MODELS = MODELS.copy()
for name in CAP_AMPLITUDES:
    model = MODELS[name]
    FIT_MODELS[name + " (capped)"] = dict(model, func=with_cap(model["func"]))


# Convert concentration to percent using the initial concentration as baseline.
def concentration_to_percent(c, mode="adsorption", c_ref=None):
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("Concentration must be a non-empty 1D array")
    if not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError("Concentration must be finite and non-negative")

    c0 = c[0]
    if mode == "adsorption":
        change = c0 - c
        reference = c0
    elif mode == "release":
        change = c - c0
        reference = c_ref
    else:
        raise ValueError("Mode must be 'adsorption' or 'release'")

    if reference is None or not np.isfinite(reference) or reference <= 0:
        raise ValueError("Reference concentration must be positive and finite")
    if np.any(change < 0):
        raise ValueError("Concentration crosses the baseline in the wrong direction")
    return 100 * (change / reference)


# R2 is undefined for a constant response.
def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


# Count the fitted curve coefficients in the information criteria.
def aicc_bic(y, y_fit, k):
    n = len(y)
    rss = max(np.sum((y - y_fit) ** 2), 1e-12)

    score = n * np.log(rss / n)
    aicc = np.inf
    if n - k - 1 > 0:
        aicc = score + 2 * k + 2 * k * (k + 1) / (n - k - 1)
    bic = score + k * np.log(n)
    return aicc, bic


# Fit one model and estimate parameter errors from the covariance matrix.
def fit_one_model(name, t, y):
    model = FIT_MODELS[name]
    func = model["func"]
    k = len(model["params"])
    if len(t) - k - 1 <= 0:
        return None

    p0 = model["p0"]
    if name.endswith(" (capped)"):
        base = name.removesuffix(" (capped)")
        peak = np.max(MODELS[base]["func"](t, *p0))
        if peak >= 100:
            p0 = p0.copy()
            for i in range(CAP_AMPLITUDES[base]):
                p0[i] *= 90 / peak

    try:
        params, cov = curve_fit(
            func, t, y,
            p0=p0,
            bounds=(model["low"], model["high"]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as error:
        print("  Model", name, "failed:", error)
        return None

    y_fit = func(t, *params)
    aicc, bic = aicc_bic(y, y_fit, k)
    if not np.isfinite(aicc):
        return None

    errors = np.sqrt(np.diag(cov))
    return {
        "model": name,
        "params": dict(zip(model["params"], params)),
        "errors": dict(zip(model["params"], errors)),
        "R2": r_squared(y, y_fit),
        "AICc": aicc,
        "BIC": bic,
    }


# Prefer fewer coefficients among models within 2 AICc points of the minimum.
def fit_all_models(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
        raise ValueError("Time and response must be 1D arrays of the same length")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Time and response must be finite")
    if np.any(t < 0):
        raise ValueError("Time must be non-negative")

    results = []
    for name in FIT_MODELS:
        result = fit_one_model(name, t, y)
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("No model could be fitted and scored with AICc")

    results.sort(key=lambda r: r["AICc"])
    best = 0
    min_aicc = results[0]["AICc"]
    for i, result in enumerate(results):
        if result["AICc"] - min_aicc >= 2:
            break
        if len(result["params"]) < len(results[best]["params"]):
            best = i

    results.insert(0, results.pop(best))
    return results


# Print fitted coefficients with their standard errors.
def print_results(results):
    print(f"{'Model':<30} {'R2':>8} {'AICc':>9} {'BIC':>9}   parameters")
    print("-" * 80)
    for result in results:
        parts = []
        for name, value in result["params"].items():
            error = result["errors"][name]
            parts.append(f"{name}={value:.4g} ± {error:.2g}")
        p_text = ", ".join(parts)
        model = result["model"]
        print(f"{model:<30} {result['R2']:8.4f} {result['AICc']:9.2f} "
              f"{result['BIC']:9.2f}   {p_text}")
    print("-" * 80)
    print("Best model:", results[0]["model"])


if __name__ == "__main__":
    t = [0, 10, 20, 40, 60, 90, 120, 180]
    y = MODELS["Higuchi"]["func"](t, 5)
    results = fit_all_models(t, y)
    assert results[0]["model"] == "Higuchi"
    assert np.isclose(results[0]["params"]["kH"], 5)
    print_results(results)
