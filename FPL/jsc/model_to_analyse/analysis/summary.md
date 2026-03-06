# Gradual vs One-shot vs Optimized One-shot

## low(<=700)
- oneshot_opt - oneshot (val_acc_eval): +0.0109
- oneshot_opt - gradual (val_acc_eval): -0.4714
- oneshot_opt - oneshot (one_step_loss_drop): +6.6699e+09
- oneshot_opt - oneshot (dead_ratio_kernel): -0.0043
- oneshot_opt - oneshot (disconnected_output_ratio): -0.1354
- oneshot_opt - oneshot (spectral_stable_rank_mean): +1.5412
- random_init - oneshot_opt (train200_best_acc): -0.0029
- oneshot_opt - oneshot (train200_best_acc): +0.0013
- oneshot_opt - gradual (train200_best_acc): -0.4482

## mid(700,5000]
- oneshot_opt - oneshot (val_acc_eval): -0.0223
- oneshot_opt - gradual (val_acc_eval): -0.5211
- oneshot_opt - oneshot (one_step_loss_drop): +4.1176e+09
- oneshot_opt - oneshot (dead_ratio_kernel): -0.0112
- oneshot_opt - oneshot (disconnected_output_ratio): -0.2739
- oneshot_opt - oneshot (spectral_stable_rank_mean): +1.4797
- random_init - oneshot_opt (train200_best_acc): -0.2098
- oneshot_opt - oneshot (train200_best_acc): +0.1156
- oneshot_opt - gradual (train200_best_acc): -0.3246

## high(>5000)
- oneshot_opt - oneshot (val_acc_eval): -0.3398
- oneshot_opt - gradual (val_acc_eval): -0.4258
- oneshot_opt - oneshot (one_step_loss_drop): +0.0000e+00
- oneshot_opt - oneshot (dead_ratio_kernel): -0.0288
- oneshot_opt - oneshot (disconnected_output_ratio): -0.2652
- oneshot_opt - oneshot (spectral_stable_rank_mean): +0.3950
- random_init - oneshot_opt (train200_best_acc): -0.4253
- oneshot_opt - oneshot (train200_best_acc): -0.0459
- oneshot_opt - gradual (train200_best_acc): -0.0703

## all
- oneshot_opt - oneshot (val_acc_eval): -0.0937
- oneshot_opt - gradual (val_acc_eval): -0.4758
- oneshot_opt - oneshot (one_step_loss_drop): +4.0405e+09
- oneshot_opt - oneshot (dead_ratio_kernel): -0.0131
- oneshot_opt - oneshot (disconnected_output_ratio): -0.2162
- oneshot_opt - oneshot (spectral_stable_rank_mean): +1.2151
- random_init - oneshot_opt (train200_best_acc): -0.1845
- oneshot_opt - oneshot (train200_best_acc): +0.0268
- oneshot_opt - gradual (train200_best_acc): -0.3063
