"""Generate a noisier fed-batch dataset for fedbatch_validation_example.ipynb.

This is a higher measurement-noise version of the dataset used by the other
fed-batch examples. The extra noise makes the overfitting effect in the
validation notebook clearly visible: larger networks fit the noise and their
held-out error grows. As with the other examples, the data-generation step
(which uses the true Monod kinetics) lives here rather than in the notebook,
and the output CSV is gitignored. Run this script once to create it:

    python docs/examples_gallery/generate_fedbatch_validation_data.py
"""
import os

import numpy as np
import pandas as pd
import jax

jax.config.update("jax_enable_x64", True)

from sindae import generate_data
from sindae.example_problems import FedBatchBioreactorProblem

# Known initial charge [X, P, S, V] of each logged batch.
BATCH_ICS = np.array([
    [0.05,  0.0, 10.0, 1.00],
    [0.025, 0.0,  5.0, 0.80],
    [0.5,   0.0,  7.5, 0.95],
])
# Roughly 3x the noise of the standard fed-batch dataset, to expose overfitting.
MEASUREMENT_NOISE = np.array([0.15, 0.15, 1.5, 0.3])   # std per state [X, P, S, V]
MEASURED_COLS = ["X", "P", "S", "V"]
SEED = 0

problem = FedBatchBioreactorProblem(ics=BATCH_ICS, nfe=40, ncp=3)
generate_data(problem, noise_std=MEASUREMENT_NOISE, obs_every=4, seed=SEED)

records = []
for batch_id, (times, values) in enumerate(zip(problem.obs_times, problem.obs_values)):
    for k, t in enumerate(times):
        row = {"batch": batch_id, "time": float(t)}
        row.update({c: float(values[k, j]) for j, c in enumerate(MEASURED_COLS)})
        records.append(row)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fedbatch_validation_measurements.csv")
pd.DataFrame(records).to_csv(out_path, index=False)
print(f"wrote {len(records)} rows to {out_path}")
