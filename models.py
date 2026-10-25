import numpy as np


def zero_order(t, k0):
    # Zero order: F = k0 * t
    return k0 * t


def first_order(t, fmax, k1):
    # First order (exponential with plateau): F = Fmax * (1 - exp(-k1*t))
    return fmax * (1 - np.exp(-k1 * t))


def higuchi(t, kh):
    # Higuchi: F = kH * sqrt(t)
    return kh * np.sqrt(t)


def korsmeyer_peppas(t, k, n):
    # Korsmeyer-Peppas: F = k * t^n
    return k * t ** n


def hixson_crowell(t, khc):
    # Hixson-Crowell: 100^(1/3) - (100 - F)^(1/3) = kHC * t
    # Solved for F:  F = 100 - (100^(1/3) - kHC*t)^3
    inside = 100 ** (1 / 3) - khc * t
    inside = np.maximum(inside, 0)
    return 100 - inside ** 3


def weibull(t, fmax, a, b):
    # Weibull: F = Fmax * (1 - exp(-(t^b) / a))
    return fmax * (1 - np.exp(-(t ** b) / a))


# Table of all models. For each model we store:
# function, parameter names, initial guess,
# lower and upper parameter bounds.
MODELS = {
    "Zero order": {
        "func": zero_order,
        "params": ["k0"],
        "p0": [1.0],
        "low": [0],
        "high": [np.inf],
    },
    "First order": {
        "func": first_order,
        "params": ["Fmax", "k1"],
        "p0": [80.0, 0.01],
        "low": [0, 0],
        "high": [100, np.inf],
    },
    "Higuchi": {
        "func": higuchi,
        "params": ["kH"],
        "p0": [1.0],
        "low": [0],
        "high": [np.inf],
    },
    "Korsmeyer-Peppas": {
        "func": korsmeyer_peppas,
        "params": ["k", "n"],
        "p0": [5.0, 0.5],
        "low": [0, 0],
        "high": [np.inf, 2],
    },
    "Hixson-Crowell": {
        "func": hixson_crowell,
        "params": ["kHC"],
        "p0": [0.01],
        "low": [0],
        "high": [np.inf],
    },
    "Weibull": {
        "func": weibull,
        "params": ["Fmax", "a", "b"],
        "p0": [80.0, 50.0, 1.0],
        "low": [0, 1e-6, 0.05],
        "high": [100, np.inf, 5],
    },
}
