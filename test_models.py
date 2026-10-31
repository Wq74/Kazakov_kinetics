import numpy as np

from models import MODELS
from fitting import fit_all_models


# Generate noisy data from each model and check that the program selects it
np.random.seed(1)
t = np.array([0, 15, 30, 60, 90, 120, 180, 240, 300, 360, 480, 600], dtype=float)

# True parameters used to generate the data (F stays below 100 %)
TRUE = {
    "Zero order": [0.15],
    "First order": [85, 0.008],
    "Higuchi": [3.5],
    "Korsmeyer-Peppas": [2.5, 0.55],
    "Peppas-Sahlin": [3.0, 0.05, 0.4],
    "Hixson-Crowell": [0.004],
    "Weibull": [90, 30, 0.7],
}

correct = 0
for name in TRUE:
    func = MODELS[name]["func"]
    y = func(t, *TRUE[name]) + np.random.normal(0, 1.0, len(t))
    results = fit_all_models(t, y)
    best = results[0]["model"]
    if best == name:
        correct += 1
        ok = "OK"
    else:
        ok = "  "
    print("{} true: {:<18} selected: {:<18} R2 = {:.4f}".format(ok, name, best, results[0]["R2"]))

print("Correct: {} of {}".format(correct, len(TRUE)))
