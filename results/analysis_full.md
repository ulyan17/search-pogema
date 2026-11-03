# Full-grid analysis (results/full_grid.csv)

43200 episodes, 12 methods, seeds per cell: 10. t_* censored at max_steps (T = 8*size).

## Leaderboard (pooled over all configurations)

Mean [95% bootstrap CI] over episodes; every method is evaluated on the same seeds (maps, starts, targets, failures).

| method | episodes | success | t_last / T | t_first / T | coverage AUC | revisit | messages |
|---|---|---|---|---|---|---|---|
| swarm | 3600 | 0.900 [0.890, 0.909] | 0.259 [0.249, 0.269] | 0.142 [0.134, 0.149] | 0.598 [0.590, 0.605] | 0.463 [0.455, 0.471] | 0 |
| frontier | 3600 | 0.868 [0.857, 0.878] | 0.409 [0.399, 0.420] | 0.229 [0.220, 0.237] | 0.538 [0.531, 0.545] | 0.675 [0.668, 0.680] | 0 |
| frontier_comm r_c=2 | 3600 | 0.951 [0.944, 0.958] | 0.242 [0.234, 0.250] | 0.134 [0.128, 0.141] | 0.592 [0.585, 0.600] | 0.481 [0.474, 0.488] | 1619 |
| frontier_comm r_c=5 | 3600 | 0.975 [0.969, 0.980] | 0.194 [0.187, 0.201] | 0.111 [0.106, 0.117] | 0.607 [0.599, 0.614] | 0.430 [0.422, 0.438] | 6026 |
| frontier_comm global | 3600 | 0.985 [0.981, 0.989] | 0.178 [0.172, 0.184] | 0.104 [0.099, 0.109] | 0.614 [0.607, 0.621] | 0.412 [0.404, 0.421] | 25483 |
| levy | 3600 | 0.740 [0.725, 0.754] | 0.501 [0.489, 0.514] | 0.294 [0.283, 0.305] | 0.504 [0.496, 0.512] | 0.692 [0.687, 0.698] | 0 |
| random_walk | 3600 | 0.461 [0.444, 0.477] | 0.716 [0.704, 0.727] | 0.477 [0.464, 0.490] | 0.396 [0.388, 0.404] | 0.825 [0.820, 0.828] | 0 |
| ippo | 3600 | 0.914 [0.905, 0.922] | 0.259 [0.250, 0.269] | 0.136 [0.129, 0.143] | 0.606 [0.599, 0.613] | 0.470 [0.462, 0.478] | 0 |
| frontier_coord global | 3600 | 0.989 [0.986, 0.992] | 0.173 [0.168, 0.178] | 0.104 [0.100, 0.108] | 0.620 [0.612, 0.627] | 0.424 [0.415, 0.433] | 28753 |
| frontier_coord r_c=2 | 3600 | 0.958 [0.952, 0.964] | 0.243 [0.235, 0.251] | 0.136 [0.131, 0.143] | 0.590 [0.583, 0.598] | 0.502 [0.494, 0.509] | 1581 |
| frontier_coord r_c=5 | 3600 | 0.976 [0.972, 0.981] | 0.200 [0.194, 0.207] | 0.117 [0.112, 0.122] | 0.611 [0.603, 0.618] | 0.456 [0.448, 0.464] | 7459 |
| ippo global | 3600 | 0.934 [0.925, 0.942] | 0.223 [0.214, 0.232] | 0.119 [0.113, 0.126] | 0.609 [0.602, 0.617] | 0.428 [0.420, 0.437] | 29011 |

## Robustness to agent loss

ratio = metric(p) / metric(p = 0) on the same seeds, pooled over configurations, independent-bootstrap CI. success: higher is better; t_last: lower is better.

| method | success p=0 | success p=0.25 | success p=0.5 | t_last ratio p=0.25 | t_last ratio p=0.5 |
|---|---|---|---|---|---|
| swarm | 0.935 | 0.897 | 0.868 | 1.21 [1.10, 1.33] | 1.44 [1.31, 1.58] |
| frontier | 0.897 | 0.874 | 0.831 | 1.06 [1.00, 1.14] | 1.19 [1.11, 1.27] |
| frontier_comm r_c=2 | 0.978 | 0.962 | 0.913 | 1.18 [1.08, 1.28] | 1.54 [1.42, 1.67] |
| frontier_comm r_c=5 | 0.994 | 0.985 | 0.945 | 1.17 [1.08, 1.27] | 1.60 [1.47, 1.73] |
| frontier_comm global | 0.999 | 0.995 | 0.962 | 1.16 [1.08, 1.25] | 1.54 [1.42, 1.67] |
| levy | 0.794 | 0.749 | 0.676 | 1.09 [1.03, 1.16] | 1.26 [1.19, 1.34] |
| random_walk | 0.512 | 0.465 | 0.404 | 1.06 [1.01, 1.10] | 1.12 [1.07, 1.16] |
| ippo | 0.958 | 0.922 | 0.861 | 1.20 [1.09, 1.32] | 1.54 [1.41, 1.69] |
| frontier_coord global | 1.000 | 0.995 | 0.972 | 1.17 [1.09, 1.25] | 1.54 [1.44, 1.66] |
| frontier_coord r_c=2 | 0.982 | 0.968 | 0.925 | 1.18 [1.09, 1.28] | 1.49 [1.37, 1.61] |
| frontier_coord r_c=5 | 0.995 | 0.984 | 0.950 | 1.19 [1.10, 1.29] | 1.56 [1.44, 1.69] |
| ippo global | 0.969 | 0.933 | 0.898 | 1.22 [1.10, 1.34] | 1.53 [1.39, 1.68] |

## Communication trade-off (p = 0, pooled)

| method | messages / episode | cells transmitted / episode | t_last / T |
|---|---|---|---|
| swarm | 0 | 0 | 0.213 [0.198, 0.227] |
| frontier | 0 | 0 | 0.377 [0.359, 0.395] |
| frontier_comm r_c=2 | 1879 | 34472 | 0.195 [0.184, 0.206] |
| frontier_comm r_c=5 | 7228 | 120928 | 0.154 [0.146, 0.163] |
| frontier_comm global | 34661 | 111144 | 0.144 [0.136, 0.151] |
| ippo | 0 | 0 | 0.207 [0.193, 0.221] |
| frontier_coord global | 40203 | 111331 | 0.140 [0.134, 0.146] |
| frontier_coord r_c=2 | 1764 | 34654 | 0.199 [0.187, 0.212] |
| frontier_coord r_c=5 | 8962 | 125740 | 0.160 [0.152, 0.169] |
| ippo global | 37666 | 110671 | 0.178 [0.166, 0.191] |

## Best communication-free method per (map size, agents), p = 0

Cell: method with the lowest mean t_last/T; the winner is declared only if the paired 95% CI of its difference to the runner-up excludes 0 (else 'tie'). Also: paired mean difference swarm - ippo [95% CI] (negative = swarm faster).

| size \ agents | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 16 | swarm (0.12); swarm-ippo -0.061 [-0.103, -0.024] | swarm (0.09); swarm-ippo -0.033 [-0.051, -0.016] | swarm (0.08); swarm-ippo -0.010 [-0.020, -0.002] | tie swarm~ippo (0.07); swarm-ippo -0.004 [-0.010, +0.001] | tie swarm~ippo (0.05); swarm-ippo -0.001 [-0.004, +0.002] |
| 32 | tie ippo~swarm (0.37); swarm-ippo +0.001 [-0.063, +0.067] | tie ippo~swarm (0.24); swarm-ippo +0.003 [-0.053, +0.059] | tie swarm~ippo (0.16); swarm-ippo -0.001 [-0.028, +0.025] | tie swarm~ippo (0.11); swarm-ippo -0.013 [-0.032, +0.004] | tie swarm~ippo (0.09); swarm-ippo -0.006 [-0.016, +0.004] |
| 64 | tie ippo~swarm (0.59); swarm-ippo +0.041 [-0.048, +0.131] | ippo (0.41); swarm-ippo +0.080 [+0.013, +0.151] | tie ippo~swarm (0.28); swarm-ippo +0.052 [-0.005, +0.112] | tie ippo~swarm (0.17); swarm-ippo +0.025 [-0.014, +0.069] | tie ippo~swarm (0.14); swarm-ippo +0.003 [-0.019, +0.027] |

