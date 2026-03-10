"""
RigL 风格谱梯度连接复活 — SpectralGradientRevivalCallback
========================================================

A+B 结合的连接复活机制：
  B — 谱引导候选过滤：找出入度/出度不足的孤立节点，标记相关死连接为候选
  A — 梯度探针排序：临时注入 probe_val，用 GradientTape 获取 |∂L/∂kq.b|，
      按梯度优先级复活 top-k

可选 swap-kill：复活时同时淘汰等量弱连接，维持 eBOPs 预算不变
"""

from __future__ import annotations

import math

import keras
import numpy as np
import tensorflow as tf

from . import _get_kq_var, _flatten_layers


class SpectralGradientRevivalCallback(keras.callbacks.Callback):
    """RigL 风格的谱梯度连接复活。

    触发条件 (AND)：
      - epoch % revival_interval == 0
      - (eBOPs 不足 OR 死连接比例过高)
      - 冷却期已过

    每次触发：
      1. [B] 谱条件过滤：找欠度节点的死连接候选集
      2. [A] 梯度探针：临时注入 → GradientTape → |grad| 排序
      3. 复活 top-k + 可选 swap-kill 弱连接
    """

    def __init__(
        self,
        target_ebops: float,
        probe_x=None,
        probe_y=None,
        min_degree: int = 2,
        probe_val: float = 0.6,
        revival_b_val: float = 1.0,
        max_revival_per_layer: int = 8,
        revival_interval: int = 200,
        ebops_deficit_threshold: float = 0.20,
        dead_fraction_threshold: float = 0.90,
        grad_min_threshold: float = 0.0,
        cool_down: int = 100,
        swap_kill: bool = False,
    ):
        super().__init__()
        self.target_ebops = float(target_ebops)
        self.probe_x = probe_x
        self.probe_y = probe_y
        self.probe_val = float(probe_val)
        self.min_degree = int(min_degree)
        self.revival_b_val = float(revival_b_val)
        self.max_revival_per_layer = int(max_revival_per_layer)
        self.revival_interval = int(revival_interval)
        self.ebops_deficit_threshold = float(ebops_deficit_threshold)
        self.dead_fraction_threshold = float(dead_fraction_threshold)
        self.grad_min_threshold = float(grad_min_threshold)
        self.cool_down = int(cool_down)
        self.swap_kill = bool(swap_kill)
        self._ebops_ema = None
        self._last_revival_epoch = -cool_down
        self._total_revived = 0

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _prunable_layers(self):
        layers = []
        for layer in _flatten_layers(self.model):
            if not hasattr(layer, 'kernel') or not hasattr(layer, 'kq'):
                continue
            b_var = _get_kq_var(layer.kq, 'b')
            if b_var is None:
                b_var = _get_kq_var(layer.kq, 'f')
            if b_var is None:
                continue
            layers.append((layer, b_var))
        return layers

    def _compute_dead_pct(self) -> float:
        total, dead = 0, 0
        for (_layer, b_var) in self._prunable_layers():
            b_arr = b_var.numpy()
            total += b_arr.size
            dead += int((b_arr < 0.5).sum())
        return dead / max(total, 1)

    def _spectral_candidates(self, b_arr: np.ndarray) -> list[tuple[int, int]]:
        """B: 返回因入度/出度不足需复活的候选 (row, col) 索引。"""
        if b_arr.ndim == 1:
            return []
        if b_arr.ndim > 2:
            b_arr = b_arr.reshape(b_arr.shape[0], -1)

        dead = (b_arr == 0.0)
        active = ~dead
        out_degree = active.sum(axis=1)  # 每个 input node 出度
        in_degree = active.sum(axis=0)   # 每个 output node 入度

        candidates = set()
        for r in np.where(out_degree < self.min_degree)[0]:
            for c in np.where(dead[r])[0]:
                candidates.add((int(r), int(c)))
        for c in np.where(in_degree < self.min_degree)[0]:
            for r in np.where(dead[:, c])[0]:
                candidates.add((int(r), int(c)))
        return list(candidates)

    def _batch_gradient_probe(self, layer_info: list) -> dict:
        """A: 梯度探针 — |∂L/∂kq.b| 排序复活优先级。"""
        has_probe = (self.probe_x is not None and self.probe_y is not None)
        if not has_probe:
            return self._weight_magnitude_fallback(layer_info)

        # 保存原始 kq.b 并临时注入 probe_val
        originals = {}
        b_vars_watched = []
        b_var_to_idx = {}

        for i, (layer, b_var, b_arr_orig, candidates, orig_shape) in enumerate(layer_info):
            originals[i] = b_arr_orig.copy()
            if not candidates:
                continue
            b_tmp = b_arr_orig.copy()
            b_2d = b_tmp.reshape(b_arr_orig.shape[0], -1) if b_arr_orig.ndim > 2 else b_tmp
            for (r, c) in candidates:
                b_2d[r, c] = self.probe_val
            if b_arr_orig.ndim > 2:
                b_tmp = b_2d.reshape(orig_shape)
            b_var.assign(b_tmp.astype(np.float32))
            b_vars_watched.append(b_var)
            b_var_to_idx[id(b_var)] = i

        # GradientTape
        grad_map = {}
        try:
            all_tv = self.model.trainable_variables
            with tf.GradientTape() as tape:
                y_pred = self.model(self.probe_x, training=True)
                loss = tf.reduce_mean(
                    keras.losses.sparse_categorical_crossentropy(
                        self.probe_y, y_pred, from_logits=True
                    )
                )
            grads = tape.gradient(loss, all_tv)
            tv_to_grad = {id(v): g for v, g in zip(all_tv, grads) if g is not None}
            for bv in b_vars_watched:
                idx = b_var_to_idx[id(bv)]
                g = tv_to_grad.get(id(bv))
                grad_map[idx] = np.abs(g.numpy()) if g is not None else None
        except Exception as e:
            print(f'  [Revival] GradientTape failed ({e}), falling back to |kernel|')
            for i, (layer, b_var, b_arr_orig, candidates, orig_shape) in enumerate(layer_info):
                b_var.assign(originals[i].astype(np.float32))
            return self._weight_magnitude_fallback(layer_info)

        # 恢复原始 kq.b
        for i, (layer, b_var, b_arr_orig, candidates, orig_shape) in enumerate(layer_info):
            b_var.assign(originals[i].astype(np.float32))

        # 按 |grad| 排序
        results = {}
        for i, (layer, b_var, b_arr_orig, candidates, orig_shape) in enumerate(layer_info):
            if not candidates:
                results[i] = []
                continue
            g_arr = grad_map.get(i)
            scored = []
            for (r, c) in candidates:
                if g_arr is not None:
                    g_2d = g_arr.reshape(g_arr.shape[0], -1) if g_arr.ndim > 2 else g_arr
                    score = float(g_2d[r, c]) if r < g_2d.shape[0] and c < g_2d.shape[1] else 0.0
                else:
                    score = float(np.random.rand())
                scored.append((score, r, c))
            scored.sort(reverse=True)
            results[i] = scored
        return results

    def _weight_magnitude_fallback(self, layer_info: list) -> dict:
        """回退：按 |kernel| 排序。"""
        results = {}
        for i, (layer, b_var, b_arr_orig, candidates, orig_shape) in enumerate(layer_info):
            if not candidates:
                results[i] = []
                continue
            w_mat = None
            kernel = getattr(layer, 'kernel', None)
            if kernel is not None:
                try:
                    w = np.abs(np.array(kernel))
                    w_mat = w.reshape(w.shape[0], -1)
                except Exception:
                    pass
            scored = []
            for (r, c) in candidates:
                if w_mat is not None and r < w_mat.shape[0] and c < w_mat.shape[1]:
                    score = float(w_mat[r, c])
                else:
                    score = float(np.random.rand())
                scored.append((score, r, c))
            scored.sort(reverse=True)
            results[i] = scored
        return results

    # ── Callback 主钩子 ───────────────────────────────────────────────────

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        raw_ebops = float(logs.get('ebops', float('nan')))
        if not math.isfinite(raw_ebops) or raw_ebops <= 0:
            return

        # EMA 平滑
        if self._ebops_ema is None:
            self._ebops_ema = raw_ebops
        else:
            self._ebops_ema = 0.3 * raw_ebops + 0.7 * self._ebops_ema

        # 触发条件
        deficit_floor = self.target_ebops * (1.0 - self.ebops_deficit_threshold)
        ebops_starved = (self._ebops_ema < deficit_floor)
        bw_0bit_pct = self._compute_dead_pct()
        topology_dead = (bw_0bit_pct > self.dead_fraction_threshold)

        if epoch % self.revival_interval == 0:
            print(f'  [Revival] epoch={epoch} ebops_ema={self._ebops_ema:.1f} '
                  f'deficit_floor={deficit_floor:.1f} starved={ebops_starved} '
                  f'dead={bw_0bit_pct*100:.1f}% topo_dead={topology_dead}')

        if not (ebops_starved or topology_dead):
            return
        if epoch % self.revival_interval != 0:
            return
        if (epoch - self._last_revival_epoch) < self.cool_down:
            return

        # ── 主流程 ────────────────────────────────────────────────────────
        pl = self._prunable_layers()
        if not pl:
            return

        layer_info = []
        for (layer, b_var) in pl:
            b_arr = b_var.numpy()
            orig_shape = b_arr.shape
            b_2d = b_arr.reshape(b_arr.shape[0], -1) if b_arr.ndim > 2 else b_arr
            candidates = self._spectral_candidates(b_2d)
            layer_info.append((layer, b_var, b_arr, candidates, orig_shape))

        total_candidates = sum(len(info[3]) for info in layer_info)
        if total_candidates == 0:
            return

        ranked = self._batch_gradient_probe(layer_info)

        # 复活 + 可选 swap-kill
        n_revived, n_killed = 0, 0
        for i, (layer, b_var, b_arr, candidates, orig_shape) in enumerate(layer_info):
            top_k = ranked.get(i, [])
            top_k = [x for x in top_k if x[0] >= self.grad_min_threshold]
            top_k = top_k[:self.max_revival_per_layer]
            if not top_k:
                continue

            b_new = b_arr.copy()
            b2d = b_new.reshape(b_arr.shape[0], -1) if b_arr.ndim > 2 else b_new

            # Swap-kill
            kill_set = []
            if self.swap_kill:
                revived_flat = {(r, c) for (_, r, c) in top_k}
                active_conns = []
                for r in range(b2d.shape[0]):
                    for c in range(b2d.shape[1]):
                        if b2d[r, c] > 0.0 and (r, c) not in revived_flat:
                            active_conns.append((float(b2d[r, c]), r, c))
                active_conns.sort()
                kill_set = [(r, c) for (_, r, c) in active_conns[:len(top_k)]]
                for (r, c) in kill_set:
                    b2d[r, c] = 0.0

            # 复活
            for (g_abs, r, c) in top_k:
                b2d[r, c] = self.revival_b_val
            if b_arr.ndim > 2:
                b_new = b2d.reshape(orig_shape)
            b_var.assign(b_new.astype(np.float32))

            n_revived += len(top_k)
            n_killed += len(kill_set)
            lname = getattr(layer, 'name', str(id(layer)))
            print(f'  [Revival] {lname}: revived={len(top_k)} killed={len(kill_set)} '
                  f'(candidates={len(candidates)}, '
                  f'top |grad|={top_k[0][0]:.3e}..{top_k[-1][0]:.3e})')

        if n_revived > 0:
            self._last_revival_epoch = epoch
            self._total_revived += n_revived
            trig = []
            if ebops_starved:
                trig.append(f'eBOPs_starved({self._ebops_ema:.0f}<{deficit_floor:.0f})')
            if topology_dead:
                trig.append(f'topo_dead({bw_0bit_pct*100:.1f}%)')
            print(f'  [Revival] epoch={epoch} revived={n_revived} killed={n_killed} '
                  f'total_revived={self._total_revived} trigger=[{", ".join(trig)}]')
            logs['revival_count'] = n_revived
