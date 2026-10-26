import numpy as np

def _t(t):
    return np.asarray(t, dtype=float)

def clip100(F):
    return np.minimum(F, 100.0)


def zero_order(t, k0):
    # Zero order: F = k0 * t
    return k0 * _t(t)


def first_order(t, fmax, k1):
    # First order (exponential with plateau): F = Fmax * (1 - exp(-k1*t))
    return fmax * (1 - np.exp(-k1 * _t(t)))


def higuchi(t, kh):
    # Higuchi: F = kH * sqrt(t)
    return kh * np.sqrt(_t(t))


def korsmeyer_peppas(t, k, n):
    # Korsmeyer-Peppas: F = k * t^n
    return k * _t(t) ** n

def peppas_sahlin(t, kd, kr, m):
    # F = kd * t^m + kr * t^(2m)
    t = _t(t)
    return kd * t ** m + kr * t ** (2 * m)

def hixson_crowell(t, khc):
    # Hixson-Crowell: 100^(1/3) - (100 - F)^(1/3) = kHC * t
    # Solved for F:  F = 100 - (100^(1/3) - kHC*t)^3
    inside = np.maximum(100 ** (1 / 3) - khc * _t(t), 0.0)
    return 100.0 - inside ** 3


def weibull(t, fmax, a, b):
    # Weibull: F = Fmax * (1 - exp(-(t^b) / a))
    a = np.maximum(a, 1e-10)
    return fmax * (1 - np.exp(-(_t(t) ** b) / a))

def with_lag(func):
    def wrapped(t, *params):
        *p, tlag = params
        return func(np.maximum(_t(t) - tlag, 0.0), *p)
    wrapped.__name__ = func.__name__ + "_lag"
    return wrapped

# Table of all models. For each model we store:
# function, parameter names, initial guess,
# lower and upper parameter bounds.
MODELS = {
    "Zero order": {
        "func": zero_order,
        "params": ["k0"],
        "p0": [0.1],
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
        "low": [0, 0.05],
        "high": [np.inf, 2],
    },
    "Peppas-Sahlin": {
        "func": peppas_sahlin,
        "params": ["kd", "kr", "m"],
        "p0": [1.0, 0.1, 0.5],
        "low": [0, 0, 0.05],
        "high": [np.inf, np.inf, 1],
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

if __name__ == "__main__":
    # FAST CHECK
    t = [0, 1, 5, 10, 30, 60]
    p = {"Zero order": (0.5,), "First order": (90, 0.05), "Higuchi": (5,),
         "Korsmeyer-Peppas": (5, 0.5), "Peppas-Sahlin": (4, 0.1, 0.5),
         "Hixson-Crowell": (0.02,), "Weibull": (90, 20, 1.0)}
    for name, m in MODELS.items():
        F = clip100(m["func"](t, *p[name]))
        assert abs(F[0]) < 1e-9, name
        print(f"{name:18s}", np.round(F, 2))
