import numpy as np

from fitting import (
    FIT_MODELS, aicc_bic, concentration_to_percent, fit_all_models, r_squared,
)
from models import MODELS


# Check a fit with a plateau and a separate capped candidate.
def check_cap():
    t = np.arange(8, dtype=float) * 2000
    name = "Zero order (capped)"
    k0 = 0.015
    y = FIT_MODELS[name]["func"](t, k0)
    best = fit_all_models(t, y)[0]
    assert best["model"] == name
    assert np.isclose(best["params"]["k0"], k0)
    assert np.isfinite(best["errors"]["k0"])
    assert MODELS["Zero order"]["func"](t, k0)[-1] > 100
    assert y[-1] == 100


# Check the concentration conversion and the undefined scores.
def check_limits():
    assert np.allclose(concentration_to_percent([10, 7, 4]), [0, 30, 60])
    assert np.allclose(
        concentration_to_percent([0, 3, 6], "release", c_ref=10),
        [0, 30, 60],
    )
    for c in ([0, 1], [10, 11], []):
        try:
            concentration_to_percent(c)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid concentration was accepted")

    y = np.array([2.0, 2.0])
    assert np.isnan(r_squared(y, y))
    assert np.isinf(aicc_bic(y, y, 1)[0])
    try:
        fit_all_models([0, 1], y)
    except RuntimeError:
        pass
    else:
        raise AssertionError("A model was selected without enough points")


# Compare selected model families on noisy synthetic curves.
def check_models():
    rng = np.random.default_rng(1)
    t = np.array([0, 15, 30, 60, 90, 120, 180, 240, 300, 360, 480, 600])
    params = {
        "Zero order": (0.15,),
        "First order": (85, 0.008),
        "Higuchi": (3.5,),
        "Korsmeyer-Peppas": (2.5, 0.55),
        "Peppas-Sahlin": (2.0, 0.3, 0.4),
        "Hixson-Crowell": (0.004,),
        "Weibull": (90, 30, 0.7),
    }

    correct = 0
    repeats = 20
    for name, p in params.items():
        curve = MODELS[name]["func"](t, *p)
        matches = 0
        for i in range(repeats):
            y = curve + rng.normal(0, 1, len(t))
            selected = fit_all_models(t, y)[0]["model"]
            if selected == name or selected == name + " (capped)":
                matches += 1
        correct += matches
        print(f"{name:<18} {matches:2d}/{repeats}")

    print(f"Total: {correct}/{repeats * len(params)}")


if __name__ == "__main__":
    check_cap()
    check_limits()
    print("Checks passed")
    check_models()
