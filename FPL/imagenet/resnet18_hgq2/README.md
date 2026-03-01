# ImageNet-1K ResNet-18 FP32 vs HGQ2

用于对比：
- 浮点 `ResNet-18`
- `HGQ2` 量化 `ResNet-18`

并探索 HGQ2 在深层网络训练中的问题（梯度消失、位宽坍缩、精度差距）。

## 已发现并修复的问题

### Bug 1: QuantizerConfigScope 参数 `f0` 对默认 KBI 量化器无效

**原因**: HGQ2 默认使用 `kbi` 类型量化器（参数为 `b0`=总比特, `i0`=整数比特），
但代码使用了 `f0`（小数比特）参数，这仅适用于 `kif` 类型量化器。
结果导致 `init_bw_k` 参数完全不生效，量化器始终使用默认 `b=4, i=2`。

**修复**: 改用 `b0 = init_bw_k + 2, i0 = 2`（KBI 格式），
使 fractional bits = `b0 - i0 = init_bw_k` 正确生效。

### Bug 2: GradientProbe 对 HGQ2 层检测不到梯度

**原因**: 仅通过 `id(layer.kernel)` 匹配梯度，当 HGQ2 的量化层 kernel 属性
返回不同对象时失败。

**修复**: 增加 fallback —— 遍历 `layer.trainable_weights` 查找含 `kernel` 的变量。

### Bug 3: `steps_per_execution` 与 dataset 不匹配

**原因**: HGQ2 训练 hardcode `steps_per_execution=8`，当 `train_steps < 8` 时
会报错。`tf.data.Dataset` 也缺少 `.repeat()` 导致多 epoch 训练时 dataset 耗尽。

**修复**: 使用 `min(8, train_steps)` 并为训练 dataset 添加 `.repeat()`。

## HGQ2 深层网络问题实验结果

### 问题 1: 量化步长 vs 权重初始化不匹配 → Dead Network

ResNet-18 每层参数量大（fan_in + fan_out 高），导致：
- **Glorot 初始化的权重值非常小**（7×7 stem conv: max ≈ 0.043）
- **低比特量化步长过大**（`f=2` 时 step=0.25 >> 0.043）
- **100% 的权重被量化为 0** → 所有激活值为 0 → 所有梯度为 0
- **网络完全无法训练**（dead network）

| init_bw_k | 总比特 | 量化步长 | 权重存活率 | 梯度 | 训练结果 |
|---|---|---|---|---|---|
| 2 | 4 | 0.250 | 0% | 全 0 | ❌ 完全失败（5% random） |
| 4 | 6 | 0.063 | 27% | 部分存活 | ⚠️ 可训练但不稳定 |
| 6 | 8 | 0.016 | 82% | 健康 | ✅ val_acc 99.4% |
| 8 | 10 | 0.004 | 95% | 健康 | ✅ 接近 FP32 |

> **结论**: 对于 ResNet-18 等深层网络，`init_bw_k` 至少需要 6（8-bit 量化），
> 否则 Glorot 初始化的权重全部被量化归零。

### 问题 2: EBOPs 正则化过强 → 训练发散

即使 `init_bw_k=6`（8-bit），若 `beta_max` 过高：
- EBOPs 惩罚在训练后期主导 loss
- 量化器快速减少比特数以降低 EBOPs
- 触发剧烈的 **位宽坍缩**（bits → 0）
- 最终 loss 变为 NaN，训练崩溃

实验对比（`init_bw_k=6`）：
- `beta_max=5e-4`: EBOPs 从 7.7B → 0，epoch 8 训练崩溃（NaN）
- `beta_max=1e-6`: EBOPs 从 7.8B → 6.4B，训练稳定至 val_acc 99.4%

### 问题 3: 前层梯度衰减

在 FP32 与 HGQ2（bw=6, stable）对比中：
- FP32 梯度比 (first/last) = 0.32
- HGQ2 梯度比 (first/last) = 0.75

量化实际上增大了前层梯度（通过 STE），但训练 loss 显著高于 FP32。

## 目录

- `model_resnet18.py`：模型定义（FP32/HGQ2）
- `train_resnet18_fp32.py`：训练浮点 ResNet-18
- `train_resnet18_hgq2.py`：训练 HGQ2 量化 ResNet-18
- `explore_hgq2_deep_issue.py`：自动诊断脚本（对比梯度与位宽，ImageNet 版）
- `create_synthetic_imagenet.py`：生成合成 ImageNet 格式数据集（用于快速测试）
- `run_all_experiments.sh`：一键运行 FP32/HGQ2/诊断并汇总结果

## 数据目录要求

`--data-root` 指向 ImageNet-1K 根目录，且包含：

- `train/`（按类别子目录组织）
- `val/`（按类别子目录组织）

也可用 `create_synthetic_imagenet.py` 生成合成数据集：

```bash
python create_synthetic_imagenet.py --num-classes 20 --train-per-class 200 --val-per-class 50 --image-size 64
```

## 运行方式

### 快速验证（合成数据集）

```bash
# 生成合成数据
python create_synthetic_imagenet.py --output /tmp/imagenet_synth

# 诊断 HGQ2 深层网络问题（低比特 → 死网络）
python explore_hgq2_deep_issue.py \
  --data-root /tmp/imagenet_synth \
  --epochs 10 --batch-size 32 --image-size 64 \
  --expected-classes 20 --strict-classes \
  --init-bw-k 2 --init-bw-a 2 \
  --output-dir result/explore_bw2

# 诊断 HGQ2 深层网络问题（高比特 → 正常训练）
python explore_hgq2_deep_issue.py \
  --data-root /tmp/imagenet_synth \
  --epochs 15 --batch-size 32 --image-size 64 \
  --expected-classes 20 --strict-classes \
  --init-bw-k 6 --init-bw-a 6 --beta-max 1e-6 \
  --output-dir result/explore_bw6_stable
```

### 一键运行（推荐使用真实 ImageNet）

```bash
bash run_all_experiments.sh --data-root /path/to/imagenet
```

## 输出

- `result/explore_bw*/diagnostic_report.json`：JSON 诊断报告（含每层梯度历史、位宽历史、EBOPs 历史）
- `result/explore_bw*/grad_compare.png`：FP32 vs HGQ2 梯度传播对比图
- `result/explore_bw*/bitwidth_evolution.png`：HGQ2 位宽演化图
- `result/explore_bw*/accuracy_loss_compare.png`：精度/损失/EBOPs 对比图

## 可重点观察的指标

- `val_acc_gap_fp32_minus_hgq2`：量化后精度损失
- `final_grad_ratio_first_last`：前层/后层梯度比（NaN 表示梯度完全消失）
- `final_pct_bits_le_1`：位宽坍缩程度（低比特占比过高会导致训练困难）
- `ebops_history`：EBOPs 演化（骤降表示计算能力丧失）
- `grad_history`：每层梯度历史（全 0 表示 dead network）
