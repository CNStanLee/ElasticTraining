#!/usr/bin/env python3
"""Debug script: Compare G3 vs C_spectral configuration."""

import os
import json
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import configs from both experiments
from run_all_v2 import DEFAULT_CONFIG, get_target_overrides_v2
from run_experiment_G_restart_decay import build_config as build_G_config
from run_experiment_C_pruning_methods import build_config as build_C_config

print("=" * 80)
print("CONFIG COMPARISON: G3 vs C_spectral @ 400 eBOPs")
print("=" * 80)

try:
    g3_cfg = build_G_config('G3')
    print("\n✓ Built G3 config successfully")
except Exception as e:
    print(f"\n✗ Failed to build G3 config: {e}")
    import traceback
    traceback.print_exc()
    g3_cfg = None

try:
    c_spectral_cfg = build_C_config('C_spectral_ebops400')
    print("✓ Built C_spectral_ebops400 config successfully")
except Exception as e:
    print(f"✗ Failed to build C_spectral_ebops400 config: {e}")
    import traceback
    traceback.print_exc()
    c_spectral_cfg = None

if g3_cfg and c_spectral_cfg:
    print("\n" + "-" * 80)
    print("KEY PARAMETERS COMPARISON")
    print("-" * 80)
    
    # Critical keys
    critical_keys = [
        'target_ebops',
        'pruning_method',
        'use_sensitivity_pruner',
        'beta_curriculum_enabled',
        'beta_restart_decay',
        'beta_stall_patience',
        'beta_recover_epochs',
        'beta_max_restarts',
        'phase1_epochs',
        'phase2_epochs',
        'phase1_lr',
        'phase2_lr',
        'phase1_lr_cycle',
        'phase2_lr_cycle',
        'warmup_ebops_mul',
        'budget_decay_epochs',
        'init_bw',
        'init_bw_a',
    ]
    
    print(f"{'Parameter':<30} {'G3':<25} {'C_spectral':<25}")
    print("-" * 80)
    
    all_same = True
    for key in critical_keys:
        g3_val = g3_cfg.get(key, 'NOT SET')
        c_val = c_spectral_cfg.get(key, 'NOT SET')
        match = "✓" if g3_val == c_val else "✗ DIFF"
        if g3_val != c_val:
            all_same = False
        print(f"{key:<30} {str(g3_val):<25} {str(c_val):<25} {match}")
    
    print("-" * 80)
    
    if all_same:
        print("\n✓ CONFIGS ARE IDENTICAL - Differences must be expected or environmental")
    else:
        print("\n✗ CONFIGS DIFFER - Key mismatches found above")
    
    # Show all keys that differ
    print("\n" + "-" * 80)
    print("ALL DIFFERENCES (keys in either config)")
    print("-" * 80)
    
    all_keys = set(g3_cfg.keys()) | set(c_spectral_cfg.keys())
    diffs = []
    for key in sorted(all_keys):
        g3_val = g3_cfg.get(key, 'NOT SET')
        c_val = c_spectral_cfg.get(key, 'NOT SET')
        if g3_val != c_val:
            diffs.append((key, g3_val, c_val))
    
    if diffs:
        print(f"{'Parameter':<35} {'G3':<30} {'C_spectral':<30}")
        print("-" * 95)
        for key, g3_val, c_val in diffs:
            # Truncate long values
            g3_str = str(g3_val)[:30] if g3_val != 'NOT SET' else 'NOT SET'
            c_str = str(c_val)[:30] if c_val != 'NOT SET' else 'NOT SET'
            print(f"{key:<35} {g3_str:<30} {c_str:<30}")
    else:
        print("None - all config keys match!")
    
    # Record to file for review
    with open('config_comparison.json', 'w') as f:
        json.dump({
            'G3': g3_cfg,
            'C_spectral_ebops400': c_spectral_cfg,
            'all_same': all_same,
            'differences': [(k, g3_cfg.get(k), c_spectral_cfg.get(k)) 
                           for k in sorted(set(g3_cfg.keys()) | set(c_spectral_cfg.keys()))
                           if g3_cfg.get(k) != c_spectral_cfg.get(k)]
        }, f, indent=2, default=str)
    
    print(f"\nSaved detailed comparison to config_comparison.json")
else:
    print("\n✗ Could not build both configs - cannot compare")
