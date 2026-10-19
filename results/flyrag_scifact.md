# FlyRAG on BEIR SciFact (research/flyrag-scifact/PREREG.md)

Seeds 1..=20; HEAD `cc6717c`; per-query nDCG@10 averaged over seeds; 95% bootstrap CIs over the 300 test queries (10,000 resamples).

| system | nDCG@10 (all 300) | unseen 104 | seen 196 |
|---|---|---|---|
| FlyRAG frozen | 0.065 | 0.065 | 0.064 |
| FlyRAG learning | 0.086 | 0.068 | 0.096 |
| FlyRAG shuffled | 0.062 | 0.068 | 0.058 |
| FlyRAG independent | 0.052 | 0.067 | 0.043 |
| BM25 (context, no bar) | 0.660 | 0.678 | 0.651 |

| bar | value | verdict |
|---|---|---|
| R1. learning − frozen > 0, CI excl. 0 | +0.0215 [+0.0159, +0.0278] | **met** |
| R2. learning − independent > 0, CI excl. 0 | +0.0346 [+0.0239, +0.0465] | **met** |
| R3. unseen: learning − frozen ≥ −0.005 or CI incl. 0 | +0.0031 [+0.0001, +0.0059] | **met** |
| (reference) seen: learning − frozen | +0.0314 [+0.0233, +0.0404] | — |
| (reference) shuffled − frozen | -0.0031 [-0.0077, +0.0013] | — |
