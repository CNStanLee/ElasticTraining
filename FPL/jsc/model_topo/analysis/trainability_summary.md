# Trainability Summary

| target | method | ebops | grad_norm | near_zero | first/last | one_step_drop | trainable | reason |
|---:|---|---:|---:|---:|---:|---:|:---:|---|
| 400 | gradual | 366 | 1.417e-01 | 0.983 | 3.896e-01 | 2.652e-02 | yes | ok: finite gradients + non-flat + depth-balanced |
| 400 | spectral_quant | 407 | 2.550e+00 | 0.990 | 9.335e-01 | 0.000e+00 | yes | ok: finite gradients + non-flat + depth-balanced |
| 400 | sensitivity | 405 | 3.640e-01 | 0.999 | 3.638e+11 | 0.000e+00 | no | too_many_zero_grads,depth_imbalance |
| 400 | uniform | 395 | 1.098e-02 | 1.000 | 0.000e+00 | 0.000e+00 | no | too_many_zero_grads,depth_imbalance |
| 1500 | gradual | 1591 | 1.358e-01 | 0.932 | 6.730e-01 | -3.356e-02 | yes | ok: finite gradients + non-flat + depth-balanced |
| 1500 | spectral_quant | 1523 | 1.217e+00 | 0.969 | 2.418e+00 | 0.000e+00 | yes | ok: finite gradients + non-flat + depth-balanced |
| 1500 | sensitivity | 1498 | 1.093e+00 | 0.996 | 1.093e+12 | 0.000e+00 | no | too_many_zero_grads,depth_imbalance |
| 1500 | uniform | 1519 | 1.217e+00 | 0.969 | 2.418e+00 | 0.000e+00 | yes | ok: finite gradients + non-flat + depth-balanced |
| 2500 | gradual | 2547 | 7.956e-02 | 0.871 | 1.825e+00 | 0.000e+00 | yes | ok: finite gradients + non-flat + depth-balanced |
| 2500 | spectral_quant | 2509 | 1.577e+00 | 0.916 | 1.054e+00 | 0.000e+00 | yes | ok: finite gradients + non-flat + depth-balanced |
| 2500 | sensitivity | 2507 | 1.533e+00 | 0.994 | 1.533e+12 | 0.000e+00 | no | depth_imbalance |
| 2500 | uniform | 2501 | 1.577e+00 | 0.916 | 1.054e+00 | 0.000e+00 | yes | ok: finite gradients + non-flat + depth-balanced |
