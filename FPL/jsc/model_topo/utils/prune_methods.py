from __future__ import annotations

DEFAULT_COMPARE_METHODS = ['spectral_quant', 'sensitivity', 'uniform']
DEFAULT_TOPOLOGY_METHODS = ['gradual', 'spectral_quant', 'sensitivity', 'uniform']

METHOD_STYLE = {
    'gradual': 'Gradual baseline',
    'spectral_quant': 'Opt (spectral_quant)',
    'sensitivity': 'Baseline (sensitivity)',
    'uniform': 'Baseline (uniform)',
}


def normalize_methods(raw: str, default: list[str]) -> list[str]:
    if not raw or not raw.strip():
        return list(default)
    return [m.strip() for m in raw.split(',') if m.strip()]
