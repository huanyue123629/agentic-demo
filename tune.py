"""超参小网格搜索（开发时用，不参与 demo 运行）。

用法：python tune.py
用途是排除"模型根本没训起来 / 过拟合到没边"这类实现问题，
不是刷指标——泛化估计以 cross_validate() 为准。

计时参考（见 bench.py）：特征 1628 维、430 条样本，
逻辑回归 100 轮约 4s，MLP(h=64) 每轮约 0.4s，所以这里只搜小规模配置。
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "02_intent_classifier")]

from classifier import MLP, SoftmaxRegression, _features, build_dataset  # noqa: E402
from common.data_intent import LABELS  # noqa: E402

tr_t, tr_y, te_t, te_y = build_dataset(include_augment=True)
vec, dim, Xtr, Xte = _features(tr_t, te_t)
print(f"dim={dim} n_train={len(Xtr)} n_test={len(Xte)}")

print("--- 逻辑回归 ---")
for lr in (0.05, 0.1):
    for l2 in (1e-4, 1e-3):
        t0 = time.time()
        m = SoftmaxRegression(dim, len(LABELS), lr=lr, l2=l2, epochs=100).fit(Xtr, tr_y)
        print(f"lr={lr} l2={l2:<7} train={m.accuracy(Xtr, tr_y):.3f} test={m.accuracy(Xte, te_y):.3f} "
              f"({time.time() - t0:.1f}s)")

print("--- MLP（样本少，重点是压容量 + 加正则）---")
for hidden in (16, 32, 64):
    for l2 in (1e-3, 1e-2):
        t0 = time.time()
        m = MLP(dim, len(LABELS), hidden=hidden, l2=l2, lr=0.05, epochs=80).fit(Xtr, tr_y)
        print(f"h={hidden:<3} l2={l2:<7} train={m.accuracy(Xtr, tr_y):.3f} test={m.accuracy(Xte, te_y):.3f} "
              f"({time.time() - t0:.1f}s)")
