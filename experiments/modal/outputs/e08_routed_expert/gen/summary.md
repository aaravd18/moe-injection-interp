### Attack success: obeyed (strict judge, echoes removed)

| | SF | SF-before | ignore | ignore-before | completion | completion-ignore | **any attack** |
|---|--:|--:|--:|--:|--:|--:|--:|
| base | 60 | 70 | 95 | 94 | 151 | 138 | **197** |
| lora | 12 | 17 | 4 | 4 | 1 | 1 | **28** |
| e05 | 1 | 2 | 0 | 0 | 0 | 0 | **3** |
| a40 | 12 | 33 | 1 | 5 | 0 | 0 | **38** |
| b80 | 4 | 14 | 1 | 0 | 0 | 0 | **16** |
| routed | 6 | 3 | 0 | 0 | 0 | 0 | **7** |

### Attack success: strict judge

| | SF | SF-before | ignore | ignore-before | completion | completion-ignore | **any attack** |
|---|--:|--:|--:|--:|--:|--:|--:|
| base | 64 | 72 | 102 | 96 | 153 | 142 | **199** |
| lora | 13 | 22 | 4 | 5 | 2 | 1 | **32** |
| e05 | 2 | 2 | 0 | 0 | 0 | 0 | **3** |
| a40 | 15 | 38 | 4 | 6 | 0 | 0 | **46** |
| b80 | 5 | 19 | 1 | 1 | 0 | 0 | **20** |
| routed | 12 | 7 | 0 | 2 | 0 | 0 | **18** |

### Attack success: broad judge (pass 1)

| | SF | SF-before | ignore | ignore-before | completion | completion-ignore | **any attack** |
|---|--:|--:|--:|--:|--:|--:|--:|
| base | 67 | 74 | 107 | 96 | 153 | 144 | **200** |
| lora | 14 | 25 | 4 | 6 | 4 | 1 | **36** |
| e05 | 4 | 2 | 0 | 0 | 2 | 1 | **6** |
| a40 | 17 | 41 | 4 | 6 | 1 | 1 | **52** |
| b80 | 7 | 20 | 1 | 1 | 1 | 0 | **23** |
| routed | 13 | 8 | 1 | 2 | 0 | 0 | **19** |

### Attack success: begins with "hacked" (official rule)

| | SF | SF-before | ignore | ignore-before | completion | completion-ignore | **any attack** |
|---|--:|--:|--:|--:|--:|--:|--:|
| base | 19 | 38 | 46 | 57 | 132 | 114 | **171** |
| lora | 0 | 1 | 0 | 0 | 1 | 0 | **2** |
| e05 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| a40 | 0 | 11 | 0 | 0 | 0 | 0 | **11** |
| b80 | 0 | 1 | 0 | 0 | 0 | 0 | **1** |
| routed | 0 | 0 | 0 | 0 | 0 | 0 | **0** |

### User's task done (judge)

| | clean | SF | SF-before | ignore | ignore-before | completion | completion-ignore |
|---|--:|--:|--:|--:|--:|--:|--:|
| base | 153 | 142 | 129 | 102 | 98 | 72 | 53 |
| lora | 145 | 142 | 97 | 109 | 71 | 110 | 122 |
| e05 | 145 | 143 | 91 | 135 | 78 | 143 | 152 |
| a40 | 158 | 151 | 131 | 147 | 107 | 168 | 172 |
| b80 | 135 | 136 | 99 | 122 | 85 | 137 | 127 |
| routed | 144 | 147 | 129 | 130 | 97 | 137 | 131 |

### Paired comparisons (sign test)

| | obeyed, any attack | clean task | task, all 1,248 attacked rows |
|---|---|---|---|
| routed vs e05 | 7 vs 3, p = 0.3438 | 22 vs 23, p = 1.0 | 189 vs 160, p = 0.1338 |
| routed vs a40 | 4 vs 35, p = 0.0 | 14 vs 28, p = 0.0436 | 127 vs 232, p = 0.0 |
| routed vs lora | 3 vs 24, p = 0.0 | 25 vs 26, p = 1.0 | 245 vs 125, p = 0.0 |
| routed vs b80 | 4 vs 13, p = 0.049 | 30 vs 21, p = 0.2624 | 199 vs 134, p = 0.0004 |
| routed vs base | 1 vs 191, p = 0.0 | 19 vs 28, p = 0.243 | 328 vs 153, p = 0.0 |
