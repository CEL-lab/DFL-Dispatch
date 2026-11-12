# Aggregated results for `ccast_results_v3/results/v3`

Environment of first run: python 3.12.2, torch 2.6.0+cu124, cvxpy 1.8.2, cvxpylayers 1.0.4, diffcp 1.1.8, scs 3.2.11, ecos 2.0.14, cuda NVIDIA A10


## base_a0  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.019230837149940783, 'max_abs_grad_diff': 0.10211073215896249, 'max_rel_grad_diff_under': 0.00010226801046660964, 'max_rel_grad_diff_over': 0.0020733044054645432, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 1846.9 ± 43.0 | 1842.6 ± 43.7 | 2.1 ± 0.4 | 0.4 ± 0.1 | 1844.7 ± 43.3 | 59.4 ± 0.2 | 0 ± 0 | 0.000 | 2001.8 ± 42.7 | 4.6 ± 0.3 | 152.2 ± 30.1 | 10339 ± 371 | 75 ± 15 | 0 |
| ASYM | 5 | 1843.6 ± 32.5 | 1839.2 ± 33.0 | 2.2 ± 0.3 | 0.4 ± 0.1 | 1841.4 ± 32.7 | 59.4 ± 0.2 | 0 ± 0 | 0.000 | 1995.8 ± 25.4 | 4.7 ± 0.3 | 11.2 ± 2.5 | 10339 ± 371 | 75 ± 16 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.1 [-0.3, 0.1]

## base_a0.5  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.019230837149940783, 'max_abs_grad_diff': 0.10211073215896249, 'max_rel_grad_diff_under': 0.00010226801046660964, 'max_rel_grad_diff_over': 0.0020733044054645432, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 415.3 ± 5.2 | 245.4 ± 12.8 | 84.9 ± 4.6 | 25.4 ± 1.4 | 330.4 ± 8.7 | 131.3 ± 4.5 | 0 ± 0 | 0.000 | 423.5 ± 3.0 | 83.0 ± 3.8 | 112.6 ± 31.8 | 10339 ± 371 | 85 ± 9 | 0 |
| ASYM | 5 | 415.3 ± 5.3 | 245.3 ± 13.0 | 85.0 ± 4.7 | 25.4 ± 1.4 | 330.3 ± 8.8 | 131.4 ± 4.5 | 0 ± 0 | 0.000 | 423.5 ± 3.0 | 83.0 ± 3.8 | 6.7 ± 2.2 | 10339 ± 371 | 85 ± 9 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.0 [-0.1, -0.0]

## base_a0.5_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.03307482649302074, 'max_abs_grad_diff': 0.23290326611856926, 'max_rel_grad_diff_under': 0.00023335991022102462, 'max_rel_grad_diff_over': 0.0026198432579142648, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 411.8 ± 4.8 | 237.7 ± 12.1 | 87.0 ± 4.6 | 26.0 ± 1.3 | 324.8 ± 8.0 | 151.3 ± 4.4 | 0 ± 0 | 0.000 | 419.4 ± 2.6 | 84.8 ± 3.6 | 112.2 ± 30.3 | 10772 ± 360 | 85 ± 9 | 0 |
| ASYM | 5 | 412.0 ± 4.9 | 238.0 ± 12.1 | 87.0 ± 4.5 | 26.0 ± 1.3 | 325.0 ± 8.1 | 151.2 ± 4.4 | 0 ± 0 | 0.000 | 419.6 ± 2.6 | 84.7 ± 3.6 | 6.6 ± 2.2 | 10772 ± 360 | 85 ± 9 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.1 [0.0, 0.1]

## base_a0.7  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.019230837149940783, 'max_abs_grad_diff': 0.10211073215896249, 'max_rel_grad_diff_under': 0.00010226801046660964, 'max_rel_grad_diff_over': 0.0020733044054645432, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 373.1 ± 2.4 | 125.7 ± 14.6 | 123.7 ± 7.2 | 36.2 ± 1.7 | 249.4 ± 7.6 | 169.3 ± 7.1 | 0 ± 0 | 0.000 | 373.1 ± 1.2 | 117.0 ± 5.3 | 95.4 ± 0.6 | 10339 ± 371 | 85 ± 9 | 0 |
| ASYM | 5 | 373.1 ± 2.4 | 125.6 ± 14.4 | 123.7 ± 7.2 | 36.2 ± 1.7 | 249.3 ± 7.5 | 169.4 ± 7.0 | 0 ± 0 | 0.000 | 373.0 ± 1.1 | 117.1 ± 5.3 | 5.2 ± 0.4 | 10339 ± 371 | 85 ± 9 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.0 [-0.0, -0.0]

## base_a0.7_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.03307482649302074, 'max_abs_grad_diff': 0.23290326611856926, 'max_rel_grad_diff_under': 0.00023335991022102462, 'max_rel_grad_diff_over': 0.0026198432579142648, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 372.1 ± 2.4 | 121.1 ± 14.6 | 125.5 ± 7.3 | 36.6 ± 1.8 | 246.6 ± 7.5 | 188.8 ± 7.2 | 0 ± 0 | 0.000 | 371.7 ± 1.0 | 118.7 ± 5.4 | 92.5 ± 9.8 | 10772 ± 360 | 85 ± 9 | 0 |
| ASYM | 5 | 372.1 ± 2.4 | 121.1 ± 14.6 | 125.5 ± 7.3 | 36.6 ± 1.8 | 246.6 ± 7.5 | 188.8 ± 7.2 | 0 ± 0 | 0.000 | 371.8 ± 1.0 | 118.7 ± 5.4 | 5.6 ± 0.5 | 10772 ± 360 | 85 ± 9 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [0.0, 0.0]

## base_a0.8  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.019230837149940783, 'max_abs_grad_diff': 0.10211073215896249, 'max_rel_grad_diff_under': 0.00010226801046660964, 'max_rel_grad_diff_over': 0.0020733044054645432, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 361.4 ± 4.6 | -33.9 ± 11.5 | 197.6 ± 7.7 | 53.3 ± 1.7 | 163.8 ± 4.2 | 242.2 ± 7.7 | 0 ± 0 | 0.000 | 351.7 ± 3.1 | 184.2 ± 5.2 | 3.0 ± 0.3 | - | 89 ± 8 | 0 |
| ICO | 5 | 362.0 ± 1.8 | 69.6 ± 13.5 | 146.2 ± 6.9 | 42.0 ± 1.6 | 215.8 ± 6.7 | 191.5 ± 6.8 | 0 ± 0 | 0.000 | 359.1 ± 1.6 | 136.9 ± 4.8 | 96.2 ± 0.9 | 10339 ± 371 | 89 ± 8 | 0 |
| ASYM | 5 | 362.0 ± 1.8 | 69.6 ± 13.5 | 146.2 ± 6.9 | 42.0 ± 1.6 | 215.8 ± 6.7 | 191.5 ± 6.8 | 0 ± 0 | 0.000 | 359.1 ± 1.6 | 136.9 ± 4.8 | 5.6 ± 0.4 | 10339 ± 371 | 89 ± 8 | 0 |
| OFFSET (offset 92.0 ± 7.3 MW) | 5 | 364.3 ± 2.5 | 58.1 ± 6.1 | 153.1 ± 3.7 | 43.2 ± 1.0 | 211.2 ± 2.8 | 198.3 ± 3.7 | 0 ± 0 | 0.000 | 360.9 ± 2.8 | 142.9 ± 1.5 | - | - | - | 0 |
| LS | 5 | 561.3 ± 0.0 | -22.3 ± 0.0 | 291.8 ± 0.0 | 45.7 ± 0.0 | 269.5 ± 0.0 | 336.1 ± 0.0 | 0 ± 0 | 0.000 | 552.0 ± 0.0 | 273.7 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -51.4 [-52.7, -50.1]; vs ASYM: -0.0 [-0.0, 0.0]; vs OFFSET: -6.9 [-7.7, -6.1]

## base_a0.8_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.03307482649302074, 'max_abs_grad_diff': 0.23290326611856926, 'max_rel_grad_diff_under': 0.00023335991022102462, 'max_rel_grad_diff_over': 0.0026198432579142648, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 361.4 ± 4.6 | -33.9 ± 11.5 | 197.6 ± 7.7 | 53.3 ± 1.7 | 163.8 ± 4.2 | 259.7 ± 7.6 | 0 ± 0 | 0.000 | 351.7 ± 3.1 | 184.2 ± 5.2 | 2.8 ± 0.5 | - | 89 ± 8 | 0 |
| ICO | 5 | 361.8 ± 1.5 | 67.6 ± 13.1 | 147.1 ± 7.0 | 42.3 ± 1.7 | 214.7 ± 6.2 | 209.9 ± 6.8 | 0 ± 0 | 0.000 | 358.8 ± 1.2 | 138.0 ± 4.9 | 91.3 ± 10.9 | 10772 ± 360 | 90 ± 8 | 0 |
| ASYM | 5 | 361.8 ± 1.5 | 67.6 ± 13.1 | 147.1 ± 7.0 | 42.2 ± 1.6 | 214.7 ± 6.2 | 209.9 ± 6.9 | 0 ± 0 | 0.000 | 358.8 ± 1.2 | 138.0 ± 4.9 | 4.9 ± 0.5 | 10772 ± 360 | 90 ± 8 | 0 |
| OFFSET (offset 88.2 ± 7.7 MW) | 5 | 363.8 ± 2.6 | 54.3 ± 6.1 | 154.7 ± 3.7 | 43.5 ± 1.0 | 209.1 ± 2.8 | 217.5 ± 3.7 | 0 ± 0 | 0.000 | 360.1 ± 2.9 | 144.4 ± 1.4 | - | - | - | 0 |
| LS | 5 | 561.3 ± 0.0 | -22.3 ± 0.0 | 291.8 ± 0.0 | 45.7 ± 0.0 | 269.5 ± 0.0 | 353.6 ± 0.0 | 0 ± 0 | 0.000 | 552.0 ± 0.0 | 273.7 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -50.6 [-51.8, -49.3]; vs ASYM: 0.0 [-0.0, 0.0]; vs OFFSET: -7.7 [-8.6, -6.9]

## base_a0.8_avrealized  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.13511859622224165, 'max_abs_grad_diff': 0.23315645230127302, 'max_rel_grad_diff_under': 0.00023559669972661617, 'max_rel_grad_diff_over': 0.0008438874639055722, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 361.4 ± 4.6 | -33.9 ± 11.5 | 197.6 ± 7.7 | 53.3 ± 1.7 | 163.8 ± 4.2 | 320.9 ± 7.6 | 0 ± 0 | 0.000 | 351.7 ± 3.1 | 184.2 ± 5.2 | 3.1 ± 0.4 | - | 89 ± 8 | 0 |
| ICO | 5 | 360.5 ± 2.0 | 53.7 ± 12.4 | 153.4 ± 6.7 | 43.8 ± 1.6 | 207.1 ± 5.8 | 277.6 ± 6.6 | 0 ± 0 | 0.000 | 356.4 ± 1.9 | 143.4 ± 4.6 | 107.4 ± 11.3 | 12466 ± 325 | 89 ± 8 | 0 |
| ASYM | 5 | 360.5 ± 2.0 | 53.7 ± 12.3 | 153.4 ± 6.7 | 43.8 ± 1.6 | 207.1 ± 5.8 | 277.6 ± 6.5 | 0 ± 0 | 0.000 | 356.4 ± 1.9 | 143.5 ± 4.6 | 5.4 ± 1.0 | 12466 ± 325 | 89 ± 8 | 0 |
| OFFSET (offset 76.0 ± 7.7 MW) | 5 | 362.4 ± 2.8 | 42.1 ± 6.2 | 160.1 ± 3.9 | 44.8 ± 1.0 | 202.3 ± 2.9 | 284.2 ± 3.8 | 0 ± 0 | 0.000 | 357.9 ± 3.0 | 149.3 ± 1.3 | - | - | - | 0 |
| LS | 5 | 561.3 ± 0.0 | -22.3 ± 0.0 | 291.8 ± 0.0 | 45.7 ± 0.0 | 269.5 ± 0.0 | 415.1 ± 0.0 | 0 ± 0 | 0.000 | 552.0 ± 0.0 | 273.7 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -44.3 [-45.4, -43.1]; vs ASYM: -0.0 [-0.0, -0.0]; vs OFFSET: -6.8 [-7.6, -6.0]

## base_a0.8_fl0.02  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.019230837149940783, 'max_abs_grad_diff': 0.10211073215896249, 'max_rel_grad_diff_under': 0.00010226801046660964, 'max_rel_grad_diff_over': 0.0020733044054645432, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.5 | 93.1 | 134.7 | 39.2 | 227.8 | 180.1 | 0 | 0.000 | 359.4 | 132.2 | 174.4 | 10044 | 74 | 0 |
| ASYM | 1 | 362.5 | 93.1 | 134.7 | 39.3 | 227.8 | 180.1 | 0 | 0.000 | 359.4 | 132.2 | 11.3 | 10044 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.0 [-0.0, 0.0]

## base_a0.8_fl0.02_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.03307482649302074, 'max_abs_grad_diff': 0.23290326611856926, 'max_rel_grad_diff_under': 0.00023335991022102462, 'max_rel_grad_diff_over': 0.0026198432579142648, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 361.7 | 88.8 | 136.5 | 39.8 | 225.3 | 199.5 | 0 | 0.000 | 358.4 | 133.9 | 69.9 | 10502 | 74 | 0 |
| ASYM | 1 | 361.7 | 88.8 | 136.5 | 39.8 | 225.3 | 199.5 | 0 | 0.000 | 358.4 | 133.9 | 3.5 | 10502 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [-0.0, 0.0]

## base_a0.8_fl0.1  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.019230837149940783, 'max_abs_grad_diff': 0.10211073215896249, 'max_rel_grad_diff_under': 0.00010226801046660964, 'max_rel_grad_diff_over': 0.0020733044054645432, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.5 | 93.1 | 134.7 | 39.2 | 227.8 | 180.1 | 0 | 0.000 | 359.4 | 132.2 | 175.2 | 10044 | 74 | 0 |
| ASYM | 1 | 362.5 | 93.1 | 134.7 | 39.3 | 227.8 | 180.0 | 0 | 0.000 | 359.4 | 132.2 | 10.4 | 10044 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [0.0, 0.0]

## base_a0.8_fl0.1_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.03307482649302074, 'max_abs_grad_diff': 0.23290326611856926, 'max_rel_grad_diff_under': 0.00023335991022102462, 'max_rel_grad_diff_over': 0.0026198432579142648, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 361.8 | 89.1 | 136.4 | 39.8 | 225.5 | 199.3 | 0 | 0.000 | 358.4 | 133.9 | 115.6 | 10502 | 74 | 0 |
| ASYM | 1 | 361.7 | 88.8 | 136.5 | 39.8 | 225.3 | 199.5 | 0 | 0.000 | 358.4 | 133.9 | 4.7 | 10502 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.1 [-0.1, -0.1]

## base_a0.8_hr1.1  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.007104535750386276, 'max_abs_grad_diff': 0.1701893310739706, 'max_rel_grad_diff_under': 0.0001704643759017969, 'max_rel_grad_diff_over': 0.0008258937666516899, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.1 | 91.2 | 135.5 | 39.6 | 226.7 | 189.4 | 0 | 0.000 | 358.9 | 133.0 | 173.3 | 10253 | 74 | 0 |
| ASYM | 1 | 362.1 | 91.1 | 135.5 | 39.6 | 226.6 | 189.4 | 0 | 0.000 | 358.9 | 133.0 | 10.3 | 10253 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.0 [-0.0, -0.0]

## base_a0.8_hr1.1_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.005182129963031912, 'max_abs_grad_diff': 0.1618063479028251, 'max_rel_grad_diff_under': 0.00016348164473075377, 'max_rel_grad_diff_over': 0.0009443367577513832, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 361.4 | 86.7 | 137.4 | 40.0 | 224.0 | 209.8 | 0 | 0.000 | 357.9 | 134.9 | 94.3 | 10771 | 74 | 0 |
| ASYM | 1 | 361.4 | 86.7 | 137.4 | 40.0 | 224.0 | 209.8 | 0 | 0.000 | 357.9 | 134.9 | 5.6 | 10771 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [-0.0, 0.0]

## base_a0.8_hr1.1_fl0.02  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.007104535750386276, 'max_abs_grad_diff': 0.1701893310739706, 'max_rel_grad_diff_under': 0.0001704643759017969, 'max_rel_grad_diff_over': 0.0007082866679291609, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.2 | 91.2 | 135.5 | 39.6 | 226.7 | 189.4 | 0 | 0.000 | 358.9 | 133.0 | 69.0 | 10253 | 74 | 0 |
| ASYM | 1 | 362.1 | 91.1 | 135.5 | 39.6 | 226.6 | 189.4 | 0 | 0.000 | 358.9 | 133.0 | 3.6 | 10253 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.0 [-0.0, -0.0]

## base_a0.8_hr1.1_fl0.02_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.005182129963031912, 'max_abs_grad_diff': 0.1618063479028251, 'max_rel_grad_diff_under': 0.00016348164473075377, 'max_rel_grad_diff_over': 0.0009443367577513832, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 361.4 | 86.7 | 137.4 | 40.0 | 224.0 | 209.8 | 0 | 0.000 | 357.9 | 134.9 | 95.1 | 10771 | 74 | 0 |
| ASYM | 1 | 361.4 | 86.7 | 137.4 | 40.0 | 224.0 | 209.8 | 0 | 0.000 | 357.9 | 134.9 | 5.6 | 10771 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [-0.0, 0.0]

## base_a0.8_hr1.1_fl0.1  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.007104535750386276, 'max_abs_grad_diff': 0.1701893310739706, 'max_rel_grad_diff_under': 0.0001704643759017969, 'max_rel_grad_diff_over': 0.0008258937666516899, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.1 | 91.0 | 135.5 | 39.6 | 226.6 | 189.4 | 0 | 0.000 | 358.9 | 133.0 | 173.6 | 10253 | 74 | 0 |
| ASYM | 1 | 362.1 | 91.1 | 135.5 | 39.6 | 226.6 | 189.4 | 0 | 0.000 | 358.9 | 133.0 | 10.9 | 10253 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [0.0, 0.0]

## base_a0.8_hr1.1_fl0.1_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.005182129963031912, 'max_abs_grad_diff': 0.1618063479028251, 'max_rel_grad_diff_under': 0.00016348164473075377, 'max_rel_grad_diff_over': 0.0009443367577513832, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 361.4 | 86.7 | 137.4 | 40.0 | 224.0 | 209.8 | 0 | 0.000 | 357.9 | 134.9 | 183.2 | 10771 | 74 | 0 |
| ASYM | 1 | 361.4 | 86.7 | 137.4 | 40.0 | 224.0 | 209.8 | 0 | 0.000 | 357.9 | 134.9 | 10.2 | 10771 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [-0.0, 0.0]

## base_a0.8_hr1.3  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.006567691578879931, 'max_abs_grad_diff': 0.230900766926311, 'max_rel_grad_diff_under': 0.00023330634608687183, 'max_rel_grad_diff_over': 0.0017897401565776766, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.8 | 94.7 | 134.0 | 39.1 | 228.8 | 172.5 | 0 | 0.000 | 359.9 | 131.5 | 69.0 | 9884 | 74 | 0 |
| ASYM | 1 | 362.8 | 94.7 | 134.0 | 39.1 | 228.7 | 172.5 | 0 | 0.000 | 359.9 | 131.5 | 4.2 | 9884 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [-0.0, 0.0]

## base_a0.8_hr1.3_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.172594192066299, 'max_abs_grad_diff': 0.22371062957870436, 'max_rel_grad_diff_under': 0.00022604548675745966, 'max_rel_grad_diff_over': 0.007320484460803497, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.1 | 90.8 | 135.6 | 39.6 | 226.5 | 190.5 | 0 | 0.000 | 358.9 | 133.1 | 94.6 | 10279 | 74 | 0 |
| ASYM | 1 | 362.0 | 90.6 | 135.7 | 39.6 | 226.3 | 190.6 | 0 | 0.000 | 358.8 | 133.2 | 4.4 | 10279 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.1 [-0.1, -0.1]

## base_a0.8_hr1.3_fl0.02  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.006567691578879931, 'max_abs_grad_diff': 0.230900766926311, 'max_rel_grad_diff_under': 0.00023330634608687183, 'max_rel_grad_diff_over': 0.0017023310044369895, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.8 | 94.7 | 134.0 | 39.1 | 228.7 | 172.5 | 0 | 0.000 | 359.9 | 131.6 | 176.1 | 9884 | 74 | 0 |
| ASYM | 1 | 362.8 | 94.7 | 134.0 | 39.1 | 228.7 | 172.5 | 0 | 0.000 | 359.9 | 131.5 | 10.4 | 9884 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.0 [-0.0, -0.0]

## base_a0.8_hr1.3_fl0.02_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.172594192066299, 'max_abs_grad_diff': 0.22371062957870436, 'max_rel_grad_diff_under': 0.00022604548675745966, 'max_rel_grad_diff_over': 0.007320484460803497, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.1 | 90.8 | 135.6 | 39.6 | 226.4 | 190.5 | 0 | 0.000 | 358.9 | 133.1 | 69.7 | 10279 | 74 | 0 |
| ASYM | 1 | 362.1 | 90.7 | 135.7 | 39.6 | 226.4 | 190.6 | 0 | 0.000 | 358.8 | 133.2 | 3.4 | 10279 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.1 [-0.1, -0.0]

## base_a0.8_hr1.3_fl0.1  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.006567691578879931, 'max_abs_grad_diff': 0.230900766926311, 'max_rel_grad_diff_under': 0.00023330634608687183, 'max_rel_grad_diff_over': 0.0017897401565776766, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.8 | 94.7 | 134.0 | 39.1 | 228.7 | 172.5 | 0 | 0.000 | 359.9 | 131.5 | 69.3 | 9884 | 74 | 0 |
| ASYM | 1 | 362.8 | 94.7 | 134.0 | 39.1 | 228.7 | 172.5 | 0 | 0.000 | 359.9 | 131.5 | 4.4 | 9884 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [-0.0, 0.0]

## base_a0.8_hr1.3_fl0.1_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.172594192066299, 'max_abs_grad_diff': 0.22371062957870436, 'max_rel_grad_diff_under': 0.00022604548675745966, 'max_rel_grad_diff_over': 0.007320484460803497, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 1 | 362.1 | 90.8 | 135.6 | 39.6 | 226.4 | 190.5 | 0 | 0.000 | 358.8 | 133.2 | 95.8 | 10279 | 74 | 0 |
| ASYM | 1 | 362.1 | 90.7 | 135.7 | 39.6 | 226.4 | 190.5 | 0 | 0.000 | 358.8 | 133.2 | 6.0 | 10279 | 74 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: -0.0 [-0.1, -0.0]

## base_a0.9  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.019230837149940783, 'max_abs_grad_diff': 0.10211073215896249, 'max_rel_grad_diff_under': 0.00010226801046660964, 'max_rel_grad_diff_over': 0.0020733044054645432, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 358.8 ± 3.0 | 19.1 ± 12.4 | 169.9 ± 7.5 | 47.5 ± 1.9 | 188.9 ± 5.0 | 214.8 ± 7.4 | 0 ± 0 | 0.000 | 352.4 ± 2.3 | 158.8 ± 5.2 | 78.9 ± 17.7 | 10339 ± 371 | 90 ± 8 | 0 |
| ASYM | 5 | 358.8 ± 3.0 | 19.1 ± 12.4 | 169.9 ± 7.5 | 47.5 ± 1.9 | 188.9 ± 5.0 | 214.8 ± 7.4 | 0 ± 0 | 0.000 | 352.5 ± 2.3 | 158.8 ± 5.2 | 5.4 ± 0.1 | 10339 ± 371 | 90 ± 8 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [0.0, 0.0]

## base_a0.95  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.019230837149940783, 'max_abs_grad_diff': 0.10211073215896249, 'max_rel_grad_diff_under': 0.00010226801046660964, 'max_rel_grad_diff_over': 0.0020733044054645432, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 359.4 ± 3.8 | -8.6 ± 11.9 | 184.0 ± 7.5 | 50.6 ± 1.9 | 175.4 ± 4.7 | 228.8 ± 7.4 | 0 ± 0 | 0.000 | 351.2 ± 2.9 | 171.4 ± 5.1 | 75.0 ± 9.9 | 10339 ± 371 | 89 ± 8 | 0 |
| ASYM | 5 | 359.4 ± 3.8 | -8.6 ± 11.9 | 184.0 ± 7.5 | 50.6 ± 1.9 | 175.4 ± 4.7 | 228.8 ± 7.4 | 0 ± 0 | 0.000 | 351.2 ± 2.9 | 171.4 ± 5.1 | 5.4 ± 0.3 | 10339 ± 371 | 89 ± 8 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [0.0, 0.0]

## base_a0.95_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.03307482649302074, 'max_abs_grad_diff': 0.23290326611856926, 'max_rel_grad_diff_under': 0.00023335991022102462, 'max_rel_grad_diff_over': 0.0026198432579142648, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 359.4 ± 3.8 | -9.6 ± 11.8 | 184.5 ± 7.5 | 50.7 ± 1.8 | 174.9 ± 4.6 | 246.8 ± 7.4 | 0 ± 0 | 0.000 | 351.2 ± 2.9 | 172.0 ± 5.0 | 97.4 ± 1.6 | 10772 ± 360 | 89 ± 8 | 0 |
| ASYM | 5 | 359.4 ± 3.8 | -9.6 ± 11.8 | 184.5 ± 7.5 | 50.7 ± 1.8 | 174.9 ± 4.6 | 246.8 ± 7.4 | 0 ± 0 | 0.000 | 351.2 ± 2.9 | 172.0 ± 5.0 | 5.4 ± 0.4 | 10772 ± 360 | 89 ± 8 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [0.0, 0.0]

## base_a0.9_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.03307482649302074, 'max_abs_grad_diff': 0.23290326611856926, 'max_rel_grad_diff_under': 0.00023335991022102462, 'max_rel_grad_diff_over': 0.0026198432579142648, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 358.8 ± 3.0 | 17.0 ± 12.3 | 170.9 ± 7.5 | 47.7 ± 1.9 | 187.9 ± 4.9 | 233.3 ± 7.4 | 0 ± 0 | 0.000 | 352.3 ± 2.3 | 159.8 ± 5.2 | 92.0 ± 9.5 | 10772 ± 360 | 90 ± 8 | 0 |
| ASYM | 5 | 358.8 ± 3.0 | 17.0 ± 12.3 | 170.9 ± 7.5 | 47.7 ± 1.9 | 187.9 ± 4.9 | 233.3 ± 7.4 | 0 ± 0 | 0.000 | 352.3 ± 2.3 | 159.8 ± 5.2 | 5.2 ± 0.8 | 10772 ± 360 | 90 ± 8 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.0 [0.0, 0.0]

## base_a0_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.03307482649302074, 'max_abs_grad_diff': 0.23290326611856926, 'max_rel_grad_diff_under': 0.00023335991022102462, 'max_rel_grad_diff_over': 0.0026198432579142648, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICO | 5 | 1739.6 ± 42.5 | 1734.5 ± 42.6 | 2.6 ± 0.2 | 0.5 ± 0.1 | 1737.1 ± 42.5 | 79.8 ± 0.3 | 0 ± 0 | 0.000 | 1854.7 ± 46.5 | 5.3 ± 0.2 | 93.7 ± 5.0 | 10772 ± 360 | 79 ± 11 | 0 |
| ASYM | 5 | 1765.6 ± 32.2 | 1761.1 ± 32.7 | 2.3 ± 0.4 | 0.4 ± 0.1 | 1763.3 ± 32.5 | 79.8 ± 0.3 | 0 ± 0 | 0.000 | 1872.3 ± 28.7 | 5.1 ± 0.2 | 5.1 ± 0.4 | 10772 ± 360 | 73 ± 10 | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs ASYM: 0.3 [0.1, 0.5]

## base_testyear2016_a0.8  (n_test = 8784, n_usable = 34681, test starts 2016-01-01 00:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.010873042201893668, 'max_abs_grad_diff': 0.2640492458731387, 'max_rel_grad_diff_under': 0.0002647032345862691, 'max_rel_grad_diff_over': 0.0011203573081765988, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 517.2 ± 9.3 | -2.5 ± 15.6 | 259.8 ± 10.8 | 50.7 ± 1.2 | 257.4 ± 7.0 | 298.9 ± 10.7 | 0 ± 0 | 0.000 | 530.5 ± 14.6 | 258.9 ± 8.8 | 1.3 ± 0.5 | - | 68 ± 16 | 0 |
| ICO | 5 | 524.4 ± 7.5 | 147.1 ± 21.9 | 188.7 ± 10.2 | 39.8 ± 2.2 | 335.8 ± 12.8 | 228.5 ± 10.1 | 0 ± 0 | 0.000 | 543.1 ± 14.1 | 194.2 ± 11.5 | 35.1 ± 12.2 | 7474 ± 848 | 77 ± 15 | 0 |
| ASYM | 5 | 524.4 ± 7.5 | 147.1 ± 21.9 | 188.7 ± 10.2 | 39.8 ± 2.2 | 335.8 ± 12.8 | 228.5 ± 10.1 | 0 ± 0 | 0.000 | 543.1 ± 14.1 | 194.2 ± 11.6 | 2.3 ± 0.9 | 7474 ± 848 | 77 ± 15 | 0 |
| OFFSET (offset 81.0 ± 14.0 MW) | 5 | 520.8 ± 7.3 | 78.5 ± 21.8 | 221.1 ± 13.0 | 44.9 ± 2.0 | 299.7 ± 9.8 | 260.6 ± 13.0 | 0 ± 0 | 0.000 | 536.2 ± 13.9 | 221.2 ± 6.4 | - | - | - | 0 |
| LS | 5 | 584.0 ± 0.0 | 2.5 ± 0.0 | 290.7 ± 0.0 | 44.2 ± 0.0 | 293.3 ± 0.0 | 329.4 ± 0.0 | 0 ± 0 | 0.000 | 614.5 ± 0.0 | 300.2 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -71.2 [-73.1, -69.4]; vs ASYM: 0.0 [-0.0, 0.0]; vs OFFSET: -32.5 [-34.2, -30.9]

## base_testyear2016_a0.8_avphysical  (n_test = 8784, n_usable = 34681, test starts 2016-01-01 00:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.9216395304420075, 'max_abs_grad_diff': 0.3317623870572106, 'max_rel_grad_diff_under': 0.0003327761631753709, 'max_rel_grad_diff_over': 0.0015643382606181193, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 517.2 ± 9.3 | -2.5 ± 15.6 | 259.8 ± 10.8 | 50.7 ± 1.2 | 257.4 ± 7.0 | 312.1 ± 10.7 | 0 ± 0 | 0.000 | 530.5 ± 14.6 | 258.9 ± 8.8 | 0.6 ± 0.2 | - | 68 ± 16 | 0 |
| ICO | 5 | 523.5 ± 7.5 | 141.2 ± 21.6 | 191.1 ± 10.2 | 40.2 ± 2.2 | 332.3 ± 12.5 | 244.4 ± 10.1 | 0 ± 0 | 0.000 | 542.1 ± 13.9 | 196.5 ± 11.6 | 22.5 ± 3.5 | 7771 ± 853 | 76 ± 14 | 0 |
| ASYM | 5 | 523.4 ± 7.5 | 141.2 ± 21.5 | 191.1 ± 10.2 | 40.2 ± 2.2 | 332.3 ± 12.4 | 244.4 ± 10.1 | 0 ± 0 | 0.000 | 542.1 ± 13.9 | 196.5 ± 11.6 | 1.2 ± 0.3 | 7771 ± 853 | 76 ± 14 | 0 |
| OFFSET (offset 76.4 ± 14.3 MW) | 5 | 520.4 ± 7.4 | 73.9 ± 22.3 | 223.2 ± 13.3 | 45.2 ± 2.0 | 297.2 ± 9.9 | 276.0 ± 13.2 | 0 ± 0 | 0.000 | 535.7 ± 13.9 | 223.3 ± 6.5 | - | - | - | 0 |
| LS | 5 | 584.0 ± 0.0 | 2.5 ± 0.0 | 290.7 ± 0.0 | 44.2 ± 0.0 | 293.3 ± 0.0 | 342.7 ± 0.0 | 0 ± 0 | 0.000 | 614.5 ± 0.0 | 300.2 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -68.7 [-70.5, -67.0]; vs ASYM: -0.0 [-0.0, 0.0]; vs OFFSET: -32.1 [-33.7, -30.6]

## base_testyear2017_a0.8  (n_test = 8545, n_usable = 34681, test starts 2017-01-01 00:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.012576872191857547, 'max_abs_grad_diff': 0.1628681555273488, 'max_rel_grad_diff_under': 0.0001646100109382745, 'max_rel_grad_diff_over': 0.00035389036414839963, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 430.3 ± 2.8 | 7.9 ± 5.8 | 211.2 ± 3.8 | 49.2 ± 0.9 | 219.1 ± 2.4 | 260.5 ± 3.8 | 0 ± 0 | 0.000 | 410.1 ± 4.9 | 203.0 ± 7.8 | 1.5 ± 0.1 | - | 78 ± 8 | 0 |
| ICO | 5 | 439.6 ± 3.8 | 119.1 ± 17.8 | 160.2 ± 7.7 | 39.2 ± 1.9 | 279.3 ± 10.3 | 210.3 ± 7.6 | 0 ± 0 | 0.000 | 414.4 ± 5.0 | 152.1 ± 4.5 | 47.6 ± 10.6 | 9817 ± 890 | 79 ± 7 | 0 |
| ASYM | 5 | 439.6 ± 3.8 | 119.1 ± 17.8 | 160.2 ± 7.7 | 39.2 ± 1.9 | 279.3 ± 10.3 | 210.3 ± 7.6 | 0 ± 0 | 0.000 | 414.4 ± 5.0 | 152.1 ± 4.5 | 2.7 ± 0.1 | 9817 ± 890 | 79 ± 7 | 0 |
| OFFSET (offset 67.8 ± 11.8 MW) | 5 | 435.5 ± 1.8 | 75.7 ± 10.3 | 179.9 ± 5.5 | 43.1 ± 0.9 | 255.6 ± 4.9 | 229.6 ± 5.4 | 0 ± 0 | 0.000 | 412.5 ± 4.4 | 170.3 ± 2.0 | - | - | - | 0 |
| LS | 5 | 579.1 ± 0.0 | 20.9 ± 0.0 | 279.1 ± 0.0 | 43.6 ± 0.0 | 300.0 ± 0.0 | 328.2 ± 0.0 | 0 ± 0 | 0.000 | 568.1 ± 0.0 | 284.1 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -51.0 [-52.4, -49.6]; vs ASYM: -0.0 [-0.0, 0.0]; vs OFFSET: -19.7 [-21.0, -18.4]

## base_testyear2017_a0.8_avphysical  (n_test = 8545, n_usable = 34681, test starts 2017-01-01 00:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.22892132162078838, 'max_abs_grad_diff': 0.07819554473132939, 'max_rel_grad_diff_under': 7.830973633354372e-05, 'max_rel_grad_diff_over': 0.00667554924307388, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 430.3 ± 2.8 | 7.9 ± 5.8 | 211.2 ± 3.8 | 49.2 ± 0.9 | 219.1 ± 2.4 | 280.9 ± 3.8 | 0 ± 0 | 0.000 | 410.1 ± 4.9 | 203.0 ± 7.8 | 1.5 ± 0.4 | - | 78 ± 8 | 0 |
| ICO | 5 | 438.8 ± 3.7 | 115.0 ± 17.3 | 161.9 ± 7.6 | 39.6 ± 1.9 | 276.9 ± 10.0 | 232.5 ± 7.4 | 0 ± 0 | 0.000 | 413.8 ± 4.9 | 153.9 ± 4.4 | 49.3 ± 9.3 | 10250 ± 898 | 79 ± 7 | 0 |
| ASYM | 5 | 438.8 ± 3.7 | 114.9 ± 17.3 | 161.9 ± 7.6 | 39.6 ± 1.9 | 276.9 ± 10.0 | 232.6 ± 7.4 | 0 ± 0 | 0.000 | 413.8 ± 4.9 | 154.0 ± 4.4 | 2.8 ± 0.8 | 10250 ± 898 | 79 ± 7 | 0 |
| OFFSET (offset 63.8 ± 12.2 MW) | 5 | 434.9 ± 1.9 | 71.7 ± 10.5 | 181.6 ± 5.6 | 43.5 ± 0.9 | 253.3 ± 5.0 | 251.8 ± 5.5 | 0 ± 0 | 0.000 | 412.1 ± 4.4 | 172.1 ± 2.1 | - | - | - | 0 |
| LS | 5 | 579.1 ± 0.0 | 20.9 ± 0.0 | 279.1 ± 0.0 | 43.6 ± 0.0 | 300.0 ± 0.0 | 348.6 ± 0.0 | 0 ± 0 | 0.000 | 568.1 ± 0.0 | 284.1 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -49.3 [-50.7, -47.9]; vs ASYM: -0.0 [-0.0, -0.0]; vs OFFSET: -19.7 [-21.1, -18.4]

## base_testyear2018_a0.8  (n_test = 8759, n_usable = 34681, test starts 2018-01-01 00:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.010043990166650474, 'max_abs_grad_diff': 0.26705632285450065, 'max_rel_grad_diff_under': 0.0002698568254306705, 'max_rel_grad_diff_over': 0.001259439941244132, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 383.1 ± 5.8 | -27.2 ± 9.3 | 205.2 ± 6.8 | 52.3 ± 1.1 | 177.9 ± 3.7 | 245.2 ± 6.8 | 0 ± 0 | 0.000 | 377.5 ± 5.8 | 180.8 ± 5.9 | 3.2 ± 1.5 | - | 83 ± 4 | 0 |
| ICO | 5 | 384.2 ± 4.3 | 64.8 ± 6.2 | 159.7 ± 5.0 | 42.8 ± 1.0 | 224.5 ± 1.9 | 200.2 ± 5.0 | 0 ± 0 | 0.000 | 383.3 ± 5.1 | 138.6 ± 4.7 | 84.7 ± 8.5 | 11952 ± 809 | 83 ± 4 | 0 |
| ASYM | 5 | 384.2 ± 4.3 | 64.8 ± 6.2 | 159.7 ± 5.0 | 42.8 ± 1.0 | 224.5 ± 1.9 | 200.2 ± 5.0 | 0 ± 0 | 0.000 | 383.3 ± 5.1 | 138.6 ± 4.7 | 4.7 ± 0.6 | 11952 ± 809 | 83 ± 4 | 0 |
| OFFSET (offset 51.2 ± 6.1 MW) | 5 | 383.4 ± 5.3 | 24.0 ± 4.4 | 179.7 ± 3.4 | 46.9 ± 0.5 | 203.7 ± 3.4 | 220.0 ± 3.4 | 0 ± 0 | 0.000 | 380.6 ± 5.9 | 156.7 ± 3.1 | - | - | - | 0 |
| LS | 5 | 572.7 ± 0.0 | -4.5 ± 0.0 | 288.6 ± 0.0 | 44.8 ± 0.0 | 284.1 ± 0.0 | 328.4 ± 0.0 | 0 ± 0 | 0.000 | 578.9 ± 0.0 | 264.6 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -45.5 [-46.3, -44.7]; vs ASYM: -0.0 [-0.0, -0.0]; vs OFFSET: -20.0 [-20.5, -19.5]

## base_testyear2018_a0.8_avphysical  (n_test = 8759, n_usable = 34681, test starts 2018-01-01 00:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.20142929800977072, 'max_abs_grad_diff': 0.10364771388617555, 'max_rel_grad_diff_under': 0.00010472614357356586, 'max_rel_grad_diff_over': 0.001259439941244132, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 383.1 ± 5.8 | -27.2 ± 9.3 | 205.2 ± 6.8 | 52.3 ± 1.1 | 177.9 ± 3.7 | 260.9 ± 6.8 | 0 ± 0 | 0.000 | 377.5 ± 5.8 | 180.8 ± 5.9 | 2.4 ± 0.2 | - | 83 ± 4 | 0 |
| ICO | 5 | 383.7 ± 4.4 | 60.3 ± 6.3 | 161.7 ± 5.1 | 43.3 ± 0.9 | 222.0 ± 1.9 | 218.0 ± 5.0 | 0 ± 0 | 0.000 | 382.6 ± 5.1 | 140.5 ± 4.7 | 81.5 ± 1.2 | 12618 ± 830 | 83 ± 4 | 0 |
| ASYM | 5 | 383.7 ± 4.4 | 60.3 ± 6.3 | 161.7 ± 5.1 | 43.3 ± 0.9 | 222.0 ± 1.9 | 218.0 ± 5.0 | 0 ± 0 | 0.000 | 382.6 ± 5.1 | 140.5 ± 4.7 | 4.3 ± 0.4 | 12618 ± 830 | 83 ± 4 | 0 |
| OFFSET (offset 46.6 ± 6.3 MW) | 5 | 383.1 ± 5.3 | 19.4 ± 3.9 | 181.9 ± 3.4 | 47.3 ± 0.5 | 201.2 ± 3.2 | 237.9 ± 3.4 | 0 ± 0 | 0.000 | 380.1 ± 6.0 | 158.8 ± 3.0 | - | - | - | 0 |
| LS | 5 | 572.7 ± 0.0 | -4.5 ± 0.0 | 288.6 ± 0.0 | 44.8 ± 0.0 | 284.1 ± 0.0 | 344.1 ± 0.0 | 0 ± 0 | 0.000 | 578.9 ± 0.0 | 264.6 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -43.5 [-44.3, -42.7]; vs ASYM: 0.0 [0.0, 0.0]; vs OFFSET: -20.2 [-20.7, -19.7]

## s2_a0.8  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.16573250394912975, 'max_abs_grad_diff': 0.07450022175009963, 'max_rel_grad_diff_under': 7.460421252861947e-05, 'max_rel_grad_diff_over': 0.001375454718473646, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 1908.0 ± 6.1 | -92.1 ± 41.3 | 1000.0 ± 21.1 | 50.6 ± 0.8 | 907.9 ± 20.6 | 1044.3 ± 20.8 | 0 ± 0 | 0.000 | 1850.1 ± 5.0 | 947.9 ± 16.8 | 1.6 ± 0.2 | - | 33 ± 5 | 0 |
| ICO | 5 | 1922.8 ± 7.3 | 227.3 ± 43.3 | 847.8 ± 20.7 | 45.1 ± 0.8 | 1075.1 ± 23.1 | 894.0 ± 20.4 | 0 ± 0 | 0.000 | 1868.1 ± 7.4 | 800.3 ± 15.3 | 53.2 ± 4.5 | 4396 ± 119 | 33 ± 5 | 0 |
| ASYM | 5 | 1922.8 ± 7.3 | 227.2 ± 43.2 | 847.8 ± 20.7 | 45.0 ± 0.8 | 1075.0 ± 23.1 | 894.0 ± 20.4 | 0 ± 0 | 0.000 | 1868.1 ± 7.4 | 800.3 ± 15.4 | 3.1 ± 0.4 | 4396 ± 119 | 33 ± 5 | 0 |
| OFFSET (offset 344.2 ± 32.9 MW) | 5 | 1924.4 ± 4.7 | 252.1 ± 19.2 | 836.1 ± 10.5 | 44.6 ± 0.5 | 1088.2 ± 9.2 | 882.5 ± 10.4 | 0 ± 0 | 0.000 | 1871.9 ± 4.7 | 786.8 ± 1.2 | - | - | - | 0 |
| LS | 5 | 2141.5 ± 0.0 | 2.0 ± 0.0 | 1069.7 ± 0.0 | 48.5 ± 0.0 | 1071.7 ± 0.0 | 1113.6 ± 0.0 | 0 ± 0 | 0.000 | 2075.8 ± 0.0 | 1012.5 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -152.2 [-156.3, -148.8]; vs ASYM: -0.0 [-0.0, -0.0]; vs OFFSET: 11.6 [8.0, 15.5]

## s2_a0.8_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.03210348136224184, 'max_abs_grad_diff': 0.18031902937923405, 'max_rel_grad_diff_under': 0.0001821971385116827, 'max_rel_grad_diff_over': 0.006539449591765519, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 1908.0 ± 6.1 | -92.1 ± 41.3 | 1000.0 ± 21.1 | 50.6 ± 0.8 | 907.9 ± 20.6 | 1061.6 ± 20.7 | 0 ± 0 | 0.000 | 1850.1 ± 5.0 | 947.9 ± 16.8 | 1.7 ± 0.6 | - | 33 ± 5 | 0 |
| ICO | 5 | 1921.3 ± 6.7 | 220.5 ± 44.2 | 850.4 ± 21.0 | 45.1 ± 0.8 | 1070.9 ± 23.6 | 914.5 ± 20.6 | 0 ± 0 | 0.000 | 1868.2 ± 7.8 | 801.1 ± 15.5 | 52.9 ± 17.2 | 4456 ± 119 | 33 ± 5 | 0 |
| ASYM | 5 | 1921.3 ± 6.7 | 220.5 ± 44.2 | 850.4 ± 21.0 | 45.1 ± 0.8 | 1070.9 ± 23.6 | 914.5 ± 20.6 | 0 ± 0 | 0.000 | 1868.2 ± 7.8 | 801.2 ± 15.5 | 3.1 ± 1.2 | 4456 ± 119 | 33 ± 5 | 0 |
| OFFSET (offset 339.2 ± 32.9 MW) | 5 | 1923.8 ± 4.7 | 247.1 ± 19.2 | 838.4 ± 10.6 | 44.7 ± 0.5 | 1085.5 ± 9.2 | 902.7 ± 10.4 | 0 ± 0 | 0.000 | 1871.3 ± 4.7 | 789.0 ± 1.2 | - | - | - | 0 |
| LS | 5 | 2141.5 ± 0.0 | 2.0 ± 0.0 | 1069.7 ± 0.0 | 48.5 ± 0.0 | 1071.7 ± 0.0 | 1131.4 ± 0.0 | 0 ± 0 | 0.000 | 2075.8 ± 0.0 | 1012.5 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -149.6 [-152.6, -147.1]; vs ASYM: 0.0 [0.0, 0.0]; vs OFFSET: 12.0 [9.2, 15.1]

## s3_a0.8  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 2.3435034389212888, 'max_abs_grad_diff': 0.09256291650092408, 'max_rel_grad_diff_under': 29.68332069031021, 'max_rel_grad_diff_over': 166.72069341439055, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}; S3: {'scale': 0.5794082629300448, 'bind_trainval': 0.1500064077918749, 'bind_test': 0.1847794753531277}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 361.4 ± 4.6 | -33.9 ± 11.5 | 472.7 ± 4.9 | 61.0 ± 1.4 | 137.3 ± 2.7 | 609.5 ± 4.8 | 646 ± 4 | 279.444 | 351.7 ± 3.1 | 402.8 ± 4.1 | 4.3 ± 1.3 | - | 89 ± 8 | 0 |
| ICO | 5 | 358.8 ± 3.1 | 16.0 ± 11.1 | 448.9 ± 4.0 | 55.8 ± 1.4 | 158.4 ± 2.7 | 586.3 ± 3.9 | 654 ± 6 | 281.326 | 352.1 ± 2.4 | 377.6 ± 3.9 | 133.6 ± 41.1 | 19180 ± 213 | 90 ± 8 | 0 |
| ASYM | 5 | 358.8 ± 3.2 | 17.3 ± 11.6 | 448.3 ± 4.1 | 55.7 ± 1.3 | 159.0 ± 2.7 | 585.7 ± 4.0 | 654 ± 5 | 281.255 | 352.3 ± 2.2 | 377.2 ± 4.1 | 7.7 ± 2.4 | 19180 ± 213 | 90 ± 8 | 0 |
| OFFSET (offset 48.2 ± 7.7 MW) | 5 | 360.5 ± 3.3 | 14.3 ± 6.1 | 452.6 ± 2.2 | 56.4 ± 0.8 | 156.3 ± 2.0 | 589.9 ± 2.2 | 655 ± 4 | 282.830 | 354.0 ± 3.1 | 382.3 ± 1.7 | - | - | - | 0 |
| LS | 5 | 561.3 ± 0.0 | -22.3 ± 0.0 | 546.9 ± 0.0 | 52.4 ± 0.0 | 237.7 ± 0.0 | 684.0 ± 0.0 | 622 ± 0 | 255.072 | 552.0 ± 0.0 | 479.6 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -23.7 [-24.4, -23.1]; vs ASYM: 0.6 [0.3, 0.9]; vs OFFSET: -3.7 [-4.3, -3.0]

## s3_a0.8_avphysical  (n_test = 3469, n_usable = 34681, test starts 2018-08-09 10:00:00+00:00)
QP-vs-closed-form check: {'max_abs_value_diff': 0.10899431411780824, 'max_abs_grad_diff': 0.0727332987879663, 'max_rel_grad_diff_under': 136.0384643476209, 'max_rel_grad_diff_over': 280.64250943771185, 'n_under': 27, 'n_over': 37, 'note': 'under: dOC/dD ~ lambda - 1000; over: dOC/dD = lambda (~1.5-11)'}; S3: {'scale': 0.60379417173542, 'bind_trainval': 0.1500064077918749, 'bind_test': 0.1772845200345921}

| model | seeds | MAE (MW) | bias (MW) | shortfall (MW) | rate (%) | excess (MW) | op. cost | slack h | identity resid. | val MAE | val shortfall | train min | kappa | best ep | QP fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PtO | 5 | 361.4 ± 4.6 | -33.9 ± 11.5 | 457.9 ± 5.4 | 61.5 ± 1.5 | 130.6 ± 2.9 | 605.8 ± 5.3 | 620 ± 6 | 263.885 | 351.7 ± 3.1 | 365.3 ± 4.1 | 3.4 ± 1.2 | - | 89 ± 8 | 0 |
| ICO | 5 | 359.5 ± 4.6 | 15.0 ± 11.7 | 433.9 ± 4.4 | 56.1 ± 1.4 | 152.0 ± 3.0 | 582.4 ± 4.3 | 627 ± 9 | 265.551 | 351.5 ± 2.3 | 340.1 ± 3.6 | 110.9 ± 36.5 | 18434 ± 206 | 85 ± 9 | 0 |
| ASYM | 5 | 358.5 ± 3.0 | 15.8 ± 10.5 | 433.5 ± 4.2 | 56.1 ± 1.4 | 151.7 ± 3.0 | 582.0 ± 4.1 | 628 ± 6 | 265.481 | 351.6 ± 2.5 | 339.7 ± 3.4 | 6.4 ± 2.5 | 18434 ± 206 | 89 ± 8 | 0 |
| OFFSET (offset 49.6 ± 7.8 MW) | 5 | 360.5 ± 3.2 | 15.7 ± 6.5 | 436.7 ± 2.5 | 56.7 ± 0.9 | 150.1 ± 2.2 | 585.2 ± 2.4 | 632 ± 6 | 267.165 | 354.2 ± 3.1 | 344.2 ± 1.5 | - | - | - | 0 |
| LS | 5 | 561.3 ± 0.0 | -22.3 ± 0.0 | 521.7 ± 0.0 | 51.9 ± 0.0 | 231.7 ± 0.0 | 670.0 ± 0.0 | 591 ± 0 | 229.871 | 552.0 ± 0.0 | 439.4 ± 0.0 | - | - | - | 0 |

Paired hourly shortfall difference ICO minus other, MW (mean; 95% moving-block bootstrap, block 168 h, seeds pooled): vs PtO: -23.9 [-24.6, -23.3]; vs ASYM: 0.4 [0.1, 0.8]; vs OFFSET: -2.8 [-3.5, -2.1]
