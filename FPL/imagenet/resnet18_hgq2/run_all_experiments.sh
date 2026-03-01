#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"
RESULT_ROOT="${RUN_DIR}/result"

DATA_ROOT=""
EPOCHS=90
EXPLORE_EPOCHS=5
BATCH_SIZE=256
IMAGE_SIZE=224
TRAIN_SUBSET=0
VAL_SUBSET=0
PROBE_BATCH=128
EXPECTED_CLASSES=1000
STRICT_CLASSES=1

usage() {
  cat <<EOF
Usage:
  bash run_all_experiments.sh --data-root /path/to/imagenet [options]

Options:
  --data-root PATH         ImageNet root with train/ and val/ (required)
  --epochs N               Epochs for fp32/hgq2 (default: 90)
  --explore-epochs N       Epochs for explore script (default: 5)
  --batch-size N           Batch size (default: 256)
  --image-size N           Input size (default: 224)
  --train-subset N         Use first N train images, 0 means full (default: 0)
  --val-subset N           Use first N val images, 0 means full (default: 0)
  --probe-batch N          Probe batch for explore (default: 128)
  --expected-classes N     Expected class count (default: 1000)
  --no-strict-classes      Disable class-count strict check
  --help                   Show this help

Output:
  All outputs are written to: ./result
    ./result/fp32
    ./result/hgq2
    ./result/explore
    ./result/compare_summary.json
    ./result/compare_summary.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --explore-epochs)
      EXPLORE_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --image-size)
      IMAGE_SIZE="$2"
      shift 2
      ;;
    --train-subset)
      TRAIN_SUBSET="$2"
      shift 2
      ;;
    --val-subset)
      VAL_SUBSET="$2"
      shift 2
      ;;
    --probe-batch)
      PROBE_BATCH="$2"
      shift 2
      ;;
    --expected-classes)
      EXPECTED_CLASSES="$2"
      shift 2
      ;;
    --no-strict-classes)
      STRICT_CLASSES=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Error: --data-root is required"
  usage
  exit 1
fi

if [[ ! -d "${DATA_ROOT}/train" || ! -d "${DATA_ROOT}/val" ]]; then
  echo "Error: DATA_ROOT must contain train/ and val/"
  echo "Given: ${DATA_ROOT}"
  exit 1
fi

mkdir -p "${RESULT_ROOT}/fp32" "${RESULT_ROOT}/hgq2" "${RESULT_ROOT}/explore" "${RESULT_ROOT}/logs"

if command -v conda >/dev/null 2>&1; then
  :
elif [[ -f "/home/changhong/anaconda3/etc/profile.d/conda.sh" ]]; then
  source /home/changhong/anaconda3/etc/profile.d/conda.sh
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate py12tf
else
  echo "Warning: conda not found, using current python environment"
fi

STRICT_FLAG=""
if [[ "${STRICT_CLASSES}" -eq 1 ]]; then
  STRICT_FLAG="--strict-classes"
fi

echo "[1/3] Running FP32 training..."
python "${SCRIPT_DIR}/train_resnet18_fp32.py" \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --train-subset "${TRAIN_SUBSET}" \
  --val-subset "${VAL_SUBSET}" \
  --expected-classes "${EXPECTED_CLASSES}" \
  ${STRICT_FLAG} \
  --output-dir "${RESULT_ROOT}/fp32" \
  2>&1 | tee "${RESULT_ROOT}/logs/fp32.log"

echo "[2/3] Running HGQ2 training..."
python "${SCRIPT_DIR}/train_resnet18_hgq2.py" \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --train-subset "${TRAIN_SUBSET}" \
  --val-subset "${VAL_SUBSET}" \
  --expected-classes "${EXPECTED_CLASSES}" \
  ${STRICT_FLAG} \
  --output-dir "${RESULT_ROOT}/hgq2" \
  2>&1 | tee "${RESULT_ROOT}/logs/hgq2.log"

echo "[3/3] Running HGQ2 deep-issue exploration..."
python "${SCRIPT_DIR}/explore_hgq2_deep_issue.py" \
  --data-root "${DATA_ROOT}" \
  --epochs "${EXPLORE_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --train-subset "${TRAIN_SUBSET}" \
  --val-subset "${VAL_SUBSET}" \
  --probe-batch "${PROBE_BATCH}" \
  --expected-classes "${EXPECTED_CLASSES}" \
  ${STRICT_FLAG} \
  --output-dir "${RESULT_ROOT}/explore" \
  2>&1 | tee "${RESULT_ROOT}/logs/explore.log"

python - <<'PY'
import json
from pathlib import Path

root = Path('result')
fp = root / 'fp32' / 'summary_fp32.json'
hq = root / 'hgq2' / 'summary_hgq2.json'
ex = root / 'explore' / 'diagnostic_report.json'

fp_data = json.loads(fp.read_text(encoding='utf-8')) if fp.exists() else {}
hq_data = json.loads(hq.read_text(encoding='utf-8')) if hq.exists() else {}
ex_data = json.loads(ex.read_text(encoding='utf-8')) if ex.exists() else {}

val_acc_fp = fp_data.get('final_val_acc')
val_acc_hq = hq_data.get('final_val_acc')
val_acc_gap = None
if isinstance(val_acc_fp, (int, float)) and isinstance(val_acc_hq, (int, float)):
    val_acc_gap = float(val_acc_fp) - float(val_acc_hq)

summary = {
    'paths': {
        'fp32': str((root / 'fp32').resolve()),
        'hgq2': str((root / 'hgq2').resolve()),
        'explore': str((root / 'explore').resolve()),
    },
    'fp32': fp_data,
    'hgq2': hq_data,
    'explore': ex_data,
    'key_compare': {
        'val_acc_fp32_minus_hgq2': val_acc_gap,
        'hgq2_final_ebops': hq_data.get('final_ebops'),
        'hgq2_findings': ex_data.get('findings', []),
    },
}

(root / 'compare_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

lines = []
lines.append('=== Experiment Summary ===')
lines.append(f"fp32 val_acc: {val_acc_fp}")
lines.append(f"hgq2 val_acc: {val_acc_hq}")
lines.append(f"val_acc gap (fp32-hgq2): {val_acc_gap}")
lines.append(f"hgq2 final ebops: {hq_data.get('final_ebops')}")
findings = ex_data.get('findings', []) if isinstance(ex_data, dict) else []
if findings:
    lines.append('findings:')
    for item in findings:
        lines.append(f"- {item}")
(root / 'compare_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Wrote result/compare_summary.json and result/compare_summary.txt')
PY

echo "Done. All outputs are under: ${RESULT_ROOT}"
