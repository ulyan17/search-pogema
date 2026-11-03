# Full-grid analysis (results/slice_prior.csv)

2880 episodes, 12 methods, seeds per cell: 10. t_* censored at max_steps (T = 8*size).

## Leaderboard (pooled over all configurations)

Mean [95% bootstrap CI] over episodes; every method is evaluated on the same seeds (maps, starts, targets, failures).

| method | episodes | success | t_last / T | t_first / T | coverage AUC | revisit | messages |
|---|---|---|---|---|---|---|---|
| swarm | 240 | 1.000 [1.000, 1.000] | 0.164 [0.148, 0.181] | 0.072 [0.066, 0.078] | 0.479 [0.457, 0.500] | 0.387 [0.368, 0.407] | 0 |
| frontier | 240 | 0.854 [0.804, 0.896] | 0.507 [0.467, 0.548] | 0.249 [0.217, 0.282] | 0.521 [0.503, 0.540] | 0.695 [0.680, 0.710] | 0 |
| frontier_comm r_c=2 | 240 | 0.942 [0.908, 0.971] | 0.314 [0.281, 0.351] | 0.150 [0.128, 0.174] | 0.580 [0.555, 0.603] | 0.474 [0.456, 0.491] | 990 |
| frontier_comm r_c=5 | 240 | 0.996 [0.988, 1.000] | 0.227 [0.202, 0.253] | 0.110 [0.095, 0.126] | 0.603 [0.579, 0.628] | 0.374 [0.356, 0.392] | 3244 |
| frontier_comm global | 240 | 1.000 [1.000, 1.000] | 0.178 [0.162, 0.195] | 0.097 [0.086, 0.108] | 0.613 [0.586, 0.638] | 0.330 [0.308, 0.352] | 15746 |
| levy | 240 | 0.596 [0.537, 0.658] | 0.653 [0.607, 0.697] | 0.348 [0.307, 0.391] | 0.447 [0.426, 0.466] | 0.713 [0.700, 0.727] | 0 |
| random_walk | 240 | 0.308 [0.250, 0.367] | 0.840 [0.800, 0.876] | 0.580 [0.528, 0.632] | 0.293 [0.277, 0.308] | 0.846 [0.837, 0.854] | 0 |
| ippo | 240 | 1.000 [1.000, 1.000] | 0.174 [0.158, 0.191] | 0.066 [0.061, 0.073] | 0.346 [0.330, 0.363] | 0.511 [0.494, 0.528] | 0 |
| frontier_coord global | 240 | 1.000 [1.000, 1.000] | 0.168 [0.154, 0.183] | 0.092 [0.083, 0.101] | 0.605 [0.578, 0.631] | 0.338 [0.315, 0.362] | 15915 |
| frontier_coord r_c=2 | 240 | 0.967 [0.942, 0.988] | 0.321 [0.291, 0.353] | 0.156 [0.136, 0.180] | 0.585 [0.563, 0.608] | 0.502 [0.485, 0.519] | 934 |
| frontier_coord r_c=5 | 240 | 0.996 [0.988, 1.000] | 0.235 [0.211, 0.261] | 0.117 [0.102, 0.132] | 0.603 [0.578, 0.627] | 0.402 [0.383, 0.423] | 3896 |
| ippo global | 240 | 0.996 [0.988, 1.000] | 0.143 [0.128, 0.159] | 0.065 [0.059, 0.071] | 0.434 [0.413, 0.456] | 0.400 [0.382, 0.420] | 13954 |

## Robustness to agent loss

ratio = metric(p) / metric(p = 0) on the same seeds, pooled over configurations, independent-bootstrap CI. success: higher is better; t_last: lower is better.

| method | success p=0 | success p=0.25 | success p=0.5 | t_last ratio p=0.25 | t_last ratio p=0.5 |
|---|---|---|---|---|---|
| swarm | 1.000 | nan | 1.000 | nan [nan, nan] | 1.36 [1.13, 1.62] |
| frontier | 0.892 | nan | 0.817 | nan [nan, nan] | 1.12 [0.96, 1.31] |
| frontier_comm r_c=2 | 0.975 | nan | 0.908 | nan [nan, nan] | 1.42 [1.14, 1.76] |
| frontier_comm r_c=5 | 1.000 | nan | 0.992 | nan [nan, nan] | 1.56 [1.29, 1.93] |
| frontier_comm global | 1.000 | nan | 1.000 | nan [nan, nan] | 1.54 [1.31, 1.79] |
| levy | 0.700 | nan | 0.492 | nan [nan, nan] | 1.19 [1.04, 1.36] |
| random_walk | 0.358 | nan | 0.258 | nan [nan, nan] | 1.09 [1.00, 1.19] |
| ippo | 1.000 | nan | 1.000 | nan [nan, nan] | 1.21 [1.00, 1.47] |
| frontier_coord global | 1.000 | nan | 1.000 | nan [nan, nan] | 1.40 [1.20, 1.65] |
| frontier_coord r_c=2 | 0.992 | nan | 0.942 | nan [nan, nan] | 1.43 [1.18, 1.72] |
| frontier_coord r_c=5 | 1.000 | nan | 0.992 | nan [nan, nan] | 1.56 [1.29, 1.88] |
| ippo global | 1.000 | nan | 0.992 | nan [nan, nan] | 1.38 [1.12, 1.69] |

## Communication trade-off (p = 0, pooled)

| method | messages / episode | cells transmitted / episode | t_last / T |
|---|---|---|---|
| swarm | 0 | 0 | 0.139 [0.122, 0.157] |
| frontier | 0 | 0 | 0.479 [0.428, 0.531] |
| frontier_comm r_c=2 | 1234 | 31341 | 0.259 [0.221, 0.304] |
| frontier_comm r_c=5 | 3930 | 71091 | 0.177 [0.153, 0.203] |
| frontier_comm global | 21320 | 87934 | 0.140 [0.127, 0.155] |
| ippo | 0 | 0 | 0.158 [0.137, 0.181] |
| frontier_coord global | 21792 | 88828 | 0.140 [0.128, 0.154] |
| frontier_coord r_c=2 | 1106 | 30147 | 0.264 [0.227, 0.302] |
| frontier_coord r_c=5 | 4692 | 75769 | 0.184 [0.163, 0.206] |
| ippo global | 18727 | 67071 | 0.120 [0.106, 0.136] |

## Best communication-free method per (map size, agents), p = 0

Cell: method with the lowest mean t_last/T; the winner is declared only if the paired 95% CI of its difference to the runner-up excludes 0 (else 'tie'). Also: paired mean difference swarm - ippo [95% CI] (negative = swarm faster).

| size \ agents | 8 | 16 | 32 |
|---|---|---|---|
| 32 | tie swarm~ippo (0.11); swarm-ippo -0.028 [-0.070, +0.001] | tie swarm~ippo (0.10); swarm-ippo -0.020 [-0.068, +0.010] | swarm (0.08); swarm-ippo -0.030 [-0.071, -0.000] |
| 64 | tie ippo~swarm (0.24); swarm-ippo +0.013 [-0.075, +0.085] | tie swarm~ippo (0.17); swarm-ippo -0.013 [-0.054, +0.025] | swarm (0.13); swarm-ippo -0.033 [-0.061, -0.010] |

