"""分阶段计时（开发时用）：找出训练流程里真正的耗时大头。

用法：python bench.py
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "02_intent_classifier")]

from classifier import MLP, SoftmaxRegression, _features, build_dataset  # noqa: E402
from common.data_intent import LABELS  # noqa: E402


def timed(name: str, fn):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    print(f"{name:<34} {dt:7.2f}s")
    return out, dt


def main() -> None:
    (tr_t, tr_y, te_t, te_y), _ = timed("build_dataset(augment)", build_dataset)
    print(f"    n_train={len(tr_t)} n_test={len(te_t)}")
    (vec, dim, Xtr, Xte), _ = timed("tfidf + transform", lambda: _features(tr_t, te_t))
    print(f"    dim={dim}")

    lr_model, t_lr = timed("SoftmaxRegression(100 ep)", lambda: SoftmaxRegression(dim, len(LABELS)).fit(Xtr, tr_y))
    print(f"    train={lr_model.accuracy(Xtr, tr_y):.3f} test={lr_model.accuracy(Xte, te_y):.3f}")

    for hidden, epochs, lr in ((32, 60, 0.05), (64, 120, 0.05)):
        m, dt = timed(f"MLP(h={hidden}, {epochs} ep)", lambda: MLP(dim, len(LABELS), hidden=hidden, epochs=epochs, lr=lr).fit(Xtr, tr_y))
        print(f"    train={m.accuracy(Xtr, tr_y):.3f} test={m.accuracy(Xte, te_y):.3f} loss={m.history[-1]['loss']:.3f} "
              f"({dt / epochs * 1000:.0f} ms/epoch)")


if __name__ == "__main__":
    main()
