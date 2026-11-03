# Full-grid analysis (results/slice_maps.csv)

1440 episodes, 12 methods, seeds per cell: 10. t_* censored at max_steps (T = 8*size).

## Leaderboard (pooled over all configurations)

Mean [95% bootstrap CI] over episodes; every method is evaluated on the same seeds (maps, starts, targets, failures).

| method | episodes | success | t_last / T | t_first / T | coverage AUC | revisit | messages |
|---|---|---|---|---|---|---|---|
| swarm | 120 | 0.792 [0.717, 0.867] | 0.470 [0.406, 0.532] | 0.088 [0.069, 0.111] | 0.625 [0.598, 0.652] | 0.591 [0.565, 0.617] | 0 |
| frontier | 120 | 0.758 [0.683, 0.825] | 0.635 [0.581, 0.687] | 0.179 [0.141, 0.221] | 0.550 [0.523, 0.575] | 0.743 [0.720, 0.764] | 0 |
| frontier_comm r_c=2 | 120 | 0.950 [0.908, 0.983] | 0.349 [0.307, 0.395] | 0.089 [0.070, 0.110] | 0.645 [0.615, 0.673] | 0.550 [0.528, 0.572] | 1359 |
| frontier_comm r_c=5 | 120 | 1.000 [1.000, 1.000] | 0.247 [0.219, 0.278] | 0.069 [0.055, 0.085] | 0.664 [0.634, 0.694] | 0.441 [0.413, 0.470] | 3921 |
| frontier_comm global | 120 | 1.000 [1.000, 1.000] | 0.218 [0.197, 0.242] | 0.060 [0.052, 0.069] | 0.682 [0.652, 0.709] | 0.413 [0.382, 0.446] | 19938 |
| levy | 120 | 0.483 [0.392, 0.575] | 0.761 [0.705, 0.813] | 0.234 [0.187, 0.286] | 0.449 [0.424, 0.475] | 0.773 [0.757, 0.789] | 0 |
| random_walk | 120 | 0.083 [0.042, 0.133] | 0.969 [0.943, 0.989] | 0.452 [0.385, 0.520] | 0.267 [0.245, 0.290] | 0.895 [0.887, 0.902] | 0 |
| ippo | 120 | 0.867 [0.808, 0.925] | 0.421 [0.367, 0.477] | 0.083 [0.066, 0.104] | 0.639 [0.615, 0.664] | 0.566 [0.541, 0.591] | 0 |
| frontier_coord global | 120 | 1.000 [1.000, 1.000] | 0.216 [0.198, 0.237] | 0.067 [0.059, 0.075] | 0.688 [0.659, 0.716] | 0.409 [0.374, 0.445] | 20437 |
| frontier_coord r_c=2 | 120 | 0.950 [0.908, 0.983] | 0.375 [0.333, 0.418] | 0.104 [0.083, 0.127] | 0.661 [0.632, 0.688] | 0.563 [0.541, 0.585] | 1271 |
| frontier_coord r_c=5 | 120 | 0.983 [0.958, 1.000] | 0.262 [0.232, 0.294] | 0.078 [0.067, 0.090] | 0.672 [0.642, 0.700] | 0.461 [0.432, 0.489] | 4547 |
| ippo global | 120 | 0.875 [0.817, 0.925] | 0.390 [0.337, 0.447] | 0.080 [0.064, 0.097] | 0.675 [0.650, 0.698] | 0.530 [0.505, 0.555] | 26936 |

## Robustness to agent loss

ratio = metric(p) / metric(p = 0) on the same seeds, pooled over configurations, independent-bootstrap CI. success: higher is better; t_last: lower is better.

| method | success p=0 | success p=0.25 | success p=0.5 | t_last ratio p=0.25 | t_last ratio p=0.5 |
|---|---|---|---|---|---|
| swarm | 0.883 | nan | 0.700 | nan [nan, nan] | 1.44 [1.11, 1.88] |
| frontier | 0.767 | nan | 0.750 | nan [nan, nan] | 1.11 [0.94, 1.32] |
| frontier_comm r_c=2 | 0.983 | nan | 0.917 | nan [nan, nan] | 1.58 [1.23, 1.99] |
| frontier_comm r_c=5 | 1.000 | nan | 1.000 | nan [nan, nan] | 1.37 [1.10, 1.74] |
| frontier_comm global | 1.000 | nan | 1.000 | nan [nan, nan] | 1.48 [1.23, 1.79] |
| levy | 0.567 | nan | 0.400 | nan [nan, nan] | 1.21 [1.04, 1.40] |
| random_walk | 0.133 | nan | 0.033 | nan [nan, nan] | 1.03 [0.98, 1.08] |
| ippo | 0.933 | nan | 0.800 | nan [nan, nan] | 1.44 [1.12, 1.88] |
| frontier_coord global | 1.000 | nan | 1.000 | nan [nan, nan] | 1.48 [1.26, 1.73] |
| frontier_coord r_c=2 | 1.000 | nan | 0.900 | nan [nan, nan] | 1.63 [1.32, 2.02] |
| frontier_coord r_c=5 | 1.000 | nan | 0.967 | nan [nan, nan] | 1.56 [1.29, 1.92] |
| ippo global | 0.983 | nan | 0.767 | nan [nan, nan] | 1.97 [1.53, 2.54] |

## Communication trade-off (p = 0, pooled)

| method | messages / episode | cells transmitted / episode | t_last / T |
|---|---|---|---|
| swarm | 0 | 0 | 0.385 [0.310, 0.466] |
| frontier | 0 | 0 | 0.602 [0.527, 0.677] |
| frontier_comm r_c=2 | 1679 | 31148 | 0.271 [0.227, 0.322] |
| frontier_comm r_c=5 | 4947 | 76403 | 0.209 [0.176, 0.246] |
| frontier_comm global | 27395 | 96496 | 0.176 [0.158, 0.194] |
| ippo | 0 | 0 | 0.344 [0.278, 0.419] |
| frontier_coord global | 28456 | 95163 | 0.174 [0.159, 0.189] |
| frontier_coord r_c=2 | 1547 | 35104 | 0.286 [0.246, 0.331] |
| frontier_coord r_c=5 | 5736 | 78498 | 0.204 [0.180, 0.230] |
| ippo global | 35404 | 93599 | 0.263 [0.214, 0.316] |

## Best communication-free method per (map size, agents), p = 0

Cell: method with the lowest mean t_last/T; the winner is declared only if the paired 95% CI of its difference to the runner-up excludes 0 (else 'tie'). Also: paired mean difference swarm - ippo [95% CI] (negative = swarm faster).

| size \ agents | 8 | 16 | 32 |
|---|---|---|---|
| 32 | tie ippo~swarm (0.27); swarm-ippo +0.057 [-0.110, +0.226] | tie ippo~swarm (0.19); swarm-ippo +0.055 [-0.009, +0.141] | tie swarm~ippo (0.16); swarm-ippo -0.023 [-0.046, +0.002] |
| 64 | tie swarm~frontier (0.68); swarm-ippo -0.123 [-0.374, +0.152] | tie ippo~swarm (0.36); swarm-ippo +0.135 [-0.066, +0.355] | tie ippo~swarm (0.26); swarm-ippo +0.142 [-0.030, +0.329] |

