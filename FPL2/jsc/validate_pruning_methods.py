#!/usr/bin/env python3
"""Quick validation: Verify G3 and C_spectral use spectral-quant (not sensitivity)."""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from run_all_v2 import DEFAULT_CONFIG, get_target_overrides_v2
from run_experiment_G_restart_decay import build_config as build_G_config
from run_experiment_C_pruning_methods import build_config as build_C_config

def check_pruning_method(cfg, name):
    """Determine which pruning method will be used given config."""
    pruning_method = cfg.get('pruning_method', 'auto')
    use_sensitivity = cfg.get('use_sensitivity_pruner', False)
    
    if pruning_method in ('random', 'magnitude', 'random_init', 'synflow', 'snip', 'grasp'):
        method = f"{pruning_method} (explicit)"
    elif use_sensitivity or pruning_method == 'sensitivity':
        method = "SensitivityAwarePruner"
    else:
        method = "spectral_quant_prune_to_ebops"
    
    return method

print("=" * 70)
print("PRUNING METHOD VALIDATION")
print("=" * 70)

g3_cfg = build_G_config('G3')
g3_method = check_pruning_method(g3_cfg, 'G3')
print(f"\nG3 → {g3_method}")
print(f"  use_sensitivity_pruner: {g3_cfg.get('use_sensitivity_pruner', 'NOT SET')}")
print(f"  pruning_method: {g3_cfg.get('pruning_method', 'NOT SET (defaults to auto)')}")

c_spectral_cfg = build_C_config('C_spectral_ebops400')
c_spectral_method = check_pruning_method(c_spectral_cfg, 'C_spectral')
print(f"\nC_spectral @ 400 eBOPs → {c_spectral_method}")
print(f"  use_sensitivity_pruner: {c_spectral_cfg.get('use_sensitivity_pruner', 'NOT SET')}")
print(f"  pruning_method: {c_spectral_cfg.get('pruning_method', 'NOT SET (defaults to auto)')}")

# Also test C_sensitivity to ensure it still works
c_sensitivity_cfg = build_C_config('C_sensitivity_ebops400')
c_sensitivity_method = check_pruning_method(c_sensitivity_cfg, 'C_sensitivity')
print(f"\nC_sensitivity @ 400 eBOPs → {c_sensitivity_method}")
print(f"  use_sensitivity_pruner: {c_sensitivity_cfg.get('use_sensitivity_pruner', 'NOT SET')}")
print(f"  pruning_method: {c_sensitivity_cfg.get('pruning_method', 'NOT SET (defaults to auto)')}")

print("\n" + "=" * 70)

if g3_method == "spectral_quant_prune_to_ebops" and c_spectral_method == "spectral_quant_prune_to_ebops":
    print("✓ SUCCESS: Both G3 and C_spectral use spectral_quant_prune_to_ebops")
    print("  → Results should now be consistent (differences only from restart_decay etc.)")
else:
    print("✗ FAILED: Pruning methods differ")
    if g3_method != "spectral_quant_prune_to_ebops":
        print(f"  G3 uses {g3_method}, expected spectral_quant_prune_to_ebops")
    if c_spectral_method != "spectral_quant_prune_to_ebops":
        print(f"  C_spectral uses {c_spectral_method}, expected spectral_quant_prune_to_ebops")

if c_sensitivity_method == "SensitivityAwarePruner":
    print("✓ C_sensitivity correctly uses SensitivityAwarePruner")
else:
    print(f"✗ C_sensitivity uses {c_sensitivity_method}, expected SensitivityAwarePruner")

print("=" * 70)
