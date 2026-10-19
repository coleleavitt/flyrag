# FlyRAG follow-up (PREREG amendment A2)

Seeds 1–20; HEAD `cc6717c`; flag t = 0.10, o = 4 (chosen on dev seeds, A3); 95% bootstrap CIs over seeds (10,000 resamples).

| arm | nDCG@10 |
|---|---|
| learning | 0.092 |
| frozen | 0.058 |
| shuffled | 0.065 |
| independent | 0.057 |

Independent arm's p(helped): mean 0.021 (per seed, the learning arm's rate).

| bar | value | verdict |
|---|---|---|
| 2b. independent − frozen < +0.01 or CI incl. 0 | -0.001 [-0.005, +0.003] | **met** |
| 2c. learning − independent > 0, CI excl. 0 | +0.035 [+0.029, +0.042] (d_z 2.43) | **met** |
| 3 (re-test). flag AUC ≥ 0.90, fresh probes | 0.637 [0.614, 0.658] | **not met** |
| (secondary) continuous novelty AUC, fresh probes | 0.643 [0.618, 0.669] | — |
| (reference) shuffled − frozen | +0.007 [+0.003, +0.012] | — |
