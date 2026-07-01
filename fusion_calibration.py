# fusion_calibration.py

import numpy as np
from semantic_patch_mapper import fuse_scores, ZONE_NAMES

# A_all, S_all: shape (N, 9), labels: shape (N,) -- 1=fake, 0=real
# Collect these by running Step 2's pipeline across N frames and stacking A, S, label

def sweep_beta(A_all, S_all, labels):
    for beta in np.arange(0.1, 1.0, 0.1):
        R_all = np.array([fuse_scores(A_all[i], S_all[i], beta) for i in range(len(labels))])
        fake_mean = R_all[labels == 1].mean(axis=0)
        real_mean = R_all[labels == 0].mean(axis=0)
        separation = (fake_mean - real_mean).mean()
        print(f"beta={beta:.1f}  mean_separation={separation:.4f}")
        print(f"  fake R(z): {np.round(fake_mean, 3)}")
        print(f"  real R(z): {np.round(real_mean, 3)}")