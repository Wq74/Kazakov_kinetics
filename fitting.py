import numpy as np
from scipy.optimize import curve_fit

from models import MODELS


def concentration_to_percent(c):
    # Decreasing concentration -> % removed from solution: F = (C0 - C) / C0 * 100
    c = np.array(c, dtype=float)
    c0 = c[0]
    return (c0 - c) / c0 * 100


def r_squared(y, y_fit):
    # Coefficient of determination R^2
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def aic_bic(y, y_fit, k):
    # Akaike and Bayesian information criteria (k = number of parameters)
    n = len(y)
    rss = np.sum((y - y_fit) ** 2)
    if rss <= 0:
        rss = 1e-12
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic


def fit_one_model(name, t, y):
    # Fit one model with curve_fit and compute quality metrics
    model = MODELS[name]
    func = model["func"]
    try:
        params, cov = curve_fit(
            func, t, y,
            p0=model["p0"],
            bounds=(model["low"], model["high"]),
            maxfev=20000,
        )
    except Exception as error:
        print("  Model", name, "failed:", error)
        return None

    y_fit = func(t, *params)
    aic, bic = aic_bic(y, y_fit, len(params))
    return {
        "model": name,
        "params": dict(zip(model["params"], params)),
        "R2": r_squared(y, y_fit),
        "AIC": aic,
        "BIC": bic,
    }


def fit_all_models(t, y):
    # Fit all models and sort them by AIC (best model first)
    t = np.array(t, dtype=float)
    y = np.array(y, dtype=float)

    results = []
    for name in MODELS:
        res = fit_one_model(name, t, y)
        if res is not None:
            results.append(res)

    results.sort(key=lambda r: r["AIC"])

    # Models with delta AIC < 2 are statistically equivalent:
    # among them choose the one with the fewest parameters
    best = results[0]
    for r in results:
        if r["AIC"] - results[0]["AIC"] < 2 and len(r["params"]) < len(best["params"]):
            best = r
    results.remove(best)
    results.insert(0, best)
    return results


def print_results(results):
    # Print a table of results
    print()
    print("{:<18} {:>8} {:>9} {:>9}   parameters".format("Model", "R2", "AIC", "BIC"))
    print("-" * 75)
    for r in results:
        p_text = ", ".join("{}={:.4g}".format(k, v) for k, v in r["params"].items())
        print("{:<18} {:>8.4f} {:>9.2f} {:>9.2f}   {}".format(
            r["model"], r["R2"], r["AIC"], r["BIC"], p_text))
    print("-" * 75)
    print("Best model:", results[0]["model"])
    print()
