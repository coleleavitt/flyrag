# FlyRAG evaluation (research/flyrag/PREREG.md)

Seeds 1..=20; HEAD `d50cf4c`; 95% bootstrap CIs over seeds (10,000 resamples).

| arm | nDCG@10 | MRR@10 |
|---|---|---|
| learning | 0.092 | 0.069 |
| frozen | 0.058 | 0.041 |
| shuffled | 0.065 | 0.046 |

| bar | value | verdict |
|---|---|---|
| 1. learning − frozen nDCG@10 ≥ +0.03, CI excl. 0 | +0.034 [+0.028, +0.040] (d_z 2.47) | **met** |
| 2. shuffled − frozen < +0.01 or CI incl. 0 | +0.007 [+0.003, +0.012] | **met** |
| 3. novelty flag AUC ≥ 0.90 | 0.507 (continuous novelty score AUC 0.997, secondary) | **not met** |
| 4. retained gain on topics 1–20 ≥ 80% | 83% [71%, 93%] (m1 0.103, m2 0.097, frozen 0.064) | **met** |
| 5. no feedback, no change | unit test `no_feedback_no_change` | see `cargo test -p flyrag` |

