# Full-grid analysis (results/slice_maps.csv)

1440 episodes, 12 methods, seeds per cell: 10. t_* censored at max_steps (T = 8*size).

## Leaderboard (pooled over all configurations)

Mean [95% bootstrap CI] over episodes; every method is evaluated on the same seeds (maps, starts, targets, failures).

| method | episodes | success | t_last / T | t_first / T | coverage AUC | revisit | messages |
|---|---|---|---|---|---|---|---|
| swarm | 120 | 0.992 [0.975, 1.000] | 0.275 [0.241, 0.313] | 0.072 [0.060, 0.086] | 0.695 [0.666, 0.724] | 0.407 [0.383, 0.431] | 0 |
| frontier | 120 | 0.642 [0.558, 0.725] | 0.625 [0.566, 0.681] | 0.220 [0.174, 0.273] | 0.561 [0.537, 0.585] | 0.755 [0.736, 0.772] | 0 |
| frontier_comm r_c=2 | 120 | 0.983 [0.958, 1.000] | 0.320 [0.280, 0.363] | 0.085 [0.066, 0.108] | 0.677 [0.649, 0.702] | 0.507 [0.481, 0.533] | 1040 |
| frontier_comm r_c=5 | 120 | 1.000 [1.000, 1.000] | 0.223 [0.197, 0.252] | 0.071 [0.057, 0.089] | 0.701 [0.672, 0.728] | 0.381 [0.353, 0.408] | 3073 |
| frontier_comm global | 120 | 1.000 [1.000, 1.000] | 0.187 [0.169, 0.205] | 0.067 [0.057, 0.080] | 0.709 [0.681, 0.738] | 0.343 [0.312, 0.374] | 17594 |
| levy | 120 | 0.758 [0.675, 0.833] | 0.622 [0.568, 0.678] | 0.178 [0.146, 0.214] | 0.557 [0.532, 0.582] | 0.697 [0.683, 0.712] | 0 |
| random_walk | 120 | 0.142 [0.083, 0.208] | 0.950 [0.921, 0.976] | 0.453 [0.390, 0.520] | 0.291 [0.267, 0.315] | 0.885 [0.876, 0.893] | 0 |
| ippo | 120 | 0.975 [0.942, 1.000] | 0.302 [0.265, 0.342] | 0.070 [0.057, 0.084] | 0.666 [0.638, 0.692] | 0.440 [0.414, 0.465] | 0 |
| frontier_coord global | 120 | 1.000 [1.000, 1.000] | 0.183 [0.168, 0.198] | 0.073 [0.062, 0.085] | 0.711 [0.681, 0.739] | 0.335 [0.303, 0.369] | 17900 |
| frontier_coord r_c=2 | 120 | 0.975 [0.942, 1.000] | 0.366 [0.321, 0.414] | 0.105 [0.081, 0.132] | 0.668 [0.641, 0.694] | 0.565 [0.538, 0.590] | 1019 |
| frontier_coord r_c=5 | 120 | 1.000 [1.000, 1.000] | 0.236 [0.211, 0.264] | 0.087 [0.073, 0.102] | 0.698 [0.671, 0.724] | 0.424 [0.397, 0.450] | 3752 |
| ippo global | 120 | 0.975 [0.942, 1.000] | 0.220 [0.190, 0.252] | 0.055 [0.048, 0.063] | 0.699 [0.672, 0.725] | 0.310 [0.284, 0.336] | 16878 |

## Robustness to agent loss

ratio = metric(p) / metric(p = 0) on the same seeds, pooled over configurations, independent-bootstrap CI. success: higher is better; t_last: lower is better.

| method | success p=0 | success p=0.25 | success p=0.5 | t_last ratio p=0.25 | t_last ratio p=0.5 |
|---|---|---|---|---|---|
| swarm | 1.000 | nan | 0.983 | nan [nan, nan] | 1.37 [1.07, 1.77] |
| frontier | 0.667 | nan | 0.617 | nan [nan, nan] | 1.09 [0.90, 1.32] |
| frontier_comm r_c=2 | 1.000 | nan | 0.967 | nan [nan, nan] | 1.50 [1.18, 1.92] |
| frontier_comm r_c=5 | 1.000 | nan | 1.000 | nan [nan, nan] | 1.69 [1.37, 2.08] |
| frontier_comm global | 1.000 | nan | 1.000 | nan [nan, nan] | 1.40 [1.17, 1.68] |
| levy | 0.850 | nan | 0.667 | nan [nan, nan] | 1.31 [1.11, 1.58] |
| random_walk | 0.167 | nan | 0.117 | nan [nan, nan] | 1.00 [0.94, 1.06] |
| ippo | 1.000 | nan | 0.950 | nan [nan, nan] | 1.47 [1.15, 1.86] |
| frontier_coord global | 1.000 | nan | 1.000 | nan [nan, nan] | 1.34 [1.15, 1.55] |
| frontier_coord r_c=2 | 1.000 | nan | 0.950 | nan [nan, nan] | 1.40 [1.08, 1.78] |
| frontier_coord r_c=5 | 1.000 | nan | 1.000 | nan [nan, nan] | 1.53 [1.25, 1.85] |
| ippo global | 0.983 | nan | 0.967 | nan [nan, nan] | 1.50 [1.14, 1.91] |

## Communication trade-off (p = 0, pooled)

| method | messages / episode | cells transmitted / episode | t_last / T |
|---|---|---|---|
| swarm | 0 | 0 | 0.232 [0.189, 0.275] |
| frontier | 0 | 0 | 0.598 [0.515, 0.684] |
| frontier_comm r_c=2 | 1312 | 41290 | 0.255 [0.212, 0.301] |
| frontier_comm r_c=5 | 3774 | 77849 | 0.166 [0.144, 0.188] |
| frontier_comm global | 24107 | 97306 | 0.156 [0.138, 0.175] |
| ippo | 0 | 0 | 0.245 [0.204, 0.289] |
| frontier_coord global | 24550 | 98622 | 0.157 [0.142, 0.173] |
| frontier_coord r_c=2 | 1262 | 42119 | 0.305 [0.248, 0.365] |
| frontier_coord r_c=5 | 4542 | 82905 | 0.186 [0.165, 0.210] |
| ippo global | 22434 | 96737 | 0.176 [0.145, 0.212] |

## Best communication-free method per (map size, agents), p = 0

Cell: method with the lowest mean t_last/T; the winner is declared only if the paired 95% CI of its difference to the runner-up excludes 0 (else 'tie'). Also: paired mean difference swarm - ippo [95% CI] (negative = swarm faster).

| size \ agents | 8 | 16 | 32 |
|---|---|---|---|
| 32 | tie swarm~ippo (0.26); swarm-ippo -0.023 [-0.098, +0.056] | tie swarm~ippo (0.15); swarm-ippo -0.013 [-0.050, +0.032] | tie swarm~ippo (0.11); swarm-ippo -0.016 [-0.036, +0.000] |
| 64 | tie ippo~swarm (0.40); swarm-ippo +0.086 [-0.064, +0.210] | tie swarm~ippo (0.23); swarm-ippo -0.100 [-0.280, +0.061] | tie swarm~ippo (0.16); swarm-ippo -0.013 [-0.069, +0.049] |

