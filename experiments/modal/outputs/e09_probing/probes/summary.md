# E09 probes: summary

Split (records): {'train': 125, 'val': 41, 'test': 42}. Labels: E07 'obeyed' (strict judge minus verbatim echoes).

## Injection detection

| model | positives train/val/test | best layer (val) | test AUROC [95% CI] | test AUPRC (prevalence) | attack-type baseline |
|---|---|---|---|---|---|
| base | 750/246/252 | 6 | 0.995 [0.9877, 0.9997] | 0.9992 (0.8571) | – |
| lora | 750/246/252 | 2 | 0.9908 [0.9757, 0.9998] | 0.9983 (0.8571) | – |
| e05 | 750/246/252 | 3 | 0.9978 [0.9926, 1.0] | 0.9996 (0.8571) | – |
| routed | 750/246/252 | 2 | 0.9842 [0.9627, 0.9991] | 0.9968 (0.8571) | – |

## Susceptibility (base obeyed)

| model | positives train/val/test | best layer (val) | test AUROC [95% CI] | test AUPRC (prevalence) | attack-type baseline |
|---|---|---|---|---|---|
| base | 363/122/123 | 14 | 0.8569 [0.7918, 0.9123] | 0.8758 (0.4881) | 0.6681 |
| lora | 363/122/123 | 9 | 0.753 [0.6863, 0.8202] | 0.7374 (0.4881) | 0.6681 |
| e05 | 363/122/123 | 7 | 0.706 [0.6199, 0.7822] | 0.6758 (0.4881) | 0.6681 |
| routed | 363/122/123 | 6 | 0.7202 [0.639, 0.8008] | 0.7094 (0.4881) | 0.6681 |

## Obedience (own labels)

| model | positives train/val/test | best layer (val) | test AUROC [95% CI] | test AUPRC (prevalence) | attack-type baseline |
|---|---|---|---|---|---|
| base | 363/122/123 | 14 | 0.8569 [0.7918, 0.9123] | 0.8758 (0.4881) | 0.6681 |
| lora | 26/6/7 | – | insufficient positives | – | – |
| e05 | 3/0/0 | – | insufficient positives | – | – |
| routed | 3/0/6 | – | insufficient positives | – | – |

## Transfer (base probe on each model, at the base probe's best val layer)

| model | task | layer | base probe AUROC | own probe AUROC | cos(base, own) |
|---|---|---|---|---|---|
| lora | detect | 6 | 0.9866 | 0.9954 | 0.6849 |
| lora | obey_base | 14 | 0.6106 | 0.66 | 0.2987 |
| e05 | detect | 6 | 0.981 | 0.9973 | 0.5848 |
| e05 | obey_base | 14 | 0.6036 | 0.6789 | 0.2788 |
| routed | detect | 6 | 0.9934 | 0.9946 | 0.7237 |
| routed | obey_base | 14 | 0.5904 | 0.6365 | 0.177 |

Random-direction |cos| (95%): < 0.0442.

