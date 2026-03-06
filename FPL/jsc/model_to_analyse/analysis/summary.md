# Gradual vs One-shot vs Optimized One-shot

## low(<=700)
- oneshot_opt - oneshot (val_acc_eval): +0.0000
- oneshot_opt - gradual (val_acc_eval): -0.4720
- oneshot_opt - oneshot (one_step_loss_drop): +9.4968e+08
- oneshot_opt - oneshot (dead_ratio_kernel): -0.0050
- oneshot_opt - oneshot (disconnected_output_ratio): -0.0657
- oneshot_opt - oneshot (spectral_stable_rank_mean): +1.3413
- random_init - oneshot_opt (train200_best_acc): +0.0182
- oneshot_opt - oneshot (train200_best_acc): +0.0008
- oneshot_opt - gradual (train200_best_acc): -0.4674

## mid(700,5000]
- oneshot_opt - oneshot (val_acc_eval): -0.0036
- oneshot_opt - gradual (val_acc_eval): -0.5416
- oneshot_opt - oneshot (one_step_loss_drop): +2.5113e+08
- oneshot_opt - oneshot (dead_ratio_kernel): -0.0089
- oneshot_opt - oneshot (disconnected_output_ratio): -0.1491
- oneshot_opt - oneshot (spectral_stable_rank_mean): +1.0069
- random_init - oneshot_opt (train200_best_acc): -0.0678
- oneshot_opt - oneshot (train200_best_acc): +0.0059
- oneshot_opt - gradual (train200_best_acc): -0.4629

## high(>5000)
- oneshot_opt - oneshot (val_acc_eval): -0.0516
- oneshot_opt - gradual (val_acc_eval): -0.2100
- oneshot_opt - oneshot (one_step_loss_drop): +0.0000e+00
- oneshot_opt - oneshot (dead_ratio_kernel): -0.0020
- oneshot_opt - oneshot (disconnected_output_ratio): -0.1182
- oneshot_opt - oneshot (spectral_stable_rank_mean): +0.1202
- random_init - oneshot_opt (train200_best_acc): -0.4006
- oneshot_opt - oneshot (train200_best_acc): -0.0754
- oneshot_opt - gradual (train200_best_acc): -0.1029

## all
- oneshot_opt - oneshot (val_acc_eval): -0.0150
- oneshot_opt - gradual (val_acc_eval): -0.4253
- oneshot_opt - oneshot (one_step_loss_drop): +4.6358e+08
- oneshot_opt - oneshot (dead_ratio_kernel): -0.0055
- oneshot_opt - oneshot (disconnected_output_ratio): -0.1075
- oneshot_opt - oneshot (spectral_stable_rank_mean): +0.9042
- random_init - oneshot_opt (train200_best_acc): -0.1221
- oneshot_opt - oneshot (train200_best_acc): -0.0178
- oneshot_opt - gradual (train200_best_acc): -0.3687
