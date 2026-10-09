| Embedding | pos_weight | seeds | PR-AUC | ROC-AUC | F1 | MCC | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| esm2_35m | auto | 3 | 0.187 ± 0.001 | 0.707 ± 0.002 | 0.248 ± 0.003 | 0.160 ± 0.007 | 0.172 ± 0.006 | 0.456 ± 0.068 |
| esm2_35m | none | 3 | 0.192 ± 0.002 | 0.706 ± 0.002 | 0.248 ± 0.002 | 0.162 ± 0.006 | 0.164 ± 0.004 | 0.514 ± 0.055 |
| esm2_8m | auto | 3 | 0.169 ± 0.003 | 0.677 ± 0.004 | 0.233 ± 0.002 | 0.143 ± 0.005 | 0.151 ± 0.002 | 0.518 ± 0.044 |
| esm2_8m | none | 3 | 0.164 ± 0.002 | 0.671 ± 0.002 | 0.227 ± 0.001 | 0.132 ± 0.003 | 0.149 ± 0.002 | 0.476 ± 0.028 |

Random scorer (prevalence) on the test set: PR-AUC ≈ 0.090, ROC-AUC ≈ 0.5, MCC = 0 (an always-'non-epitope' model has MCC = 0 and F1 = 0).
Threshold-based metrics use the F1-optimal threshold selected on the validation set.
