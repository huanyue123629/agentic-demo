"""检索排序的评测指标：Recall@K / MRR / nDCG@K。

不用第三方库，公式全部手写，便于在方案文档里逐条解释。
约定：相关性为二值（命中 / 未命中），命中按"答案组"判断。
"""

from __future__ import annotations

import math


def _hit(item: dict, gold: set[str]) -> bool:
    """一个检索结果是否算命中：答案组相交即可。"""
    groups = set(item.get("gold_groups") or [])
    if not groups and item.get("group"):
        groups = {item["group"]}
    return bool(groups & gold)


def recall_at_k(ranked: list[dict], gold: set[str], k: int) -> float:
    """前 k 条里有没有至少一条命中（这里是 group recall，同一组算同一答案）。"""
    return 1.0 if any(_hit(it, gold) for it in ranked[:k]) else 0.0


def mrr(ranked: list[dict], gold: set[str]) -> float:
    """第一条命中的排名的倒数；没有命中记 0。"""
    for i, it in enumerate(ranked, start=1):
        if _hit(it, gold):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[dict], gold: set[str], k: int) -> float:
    """二值相关性的 nDCG@K。

    同一答案组可能命中多条，这里按"组"去重：每个组只贡献一次增益，
    更贴近业务上"找到那套解决方案就算成功"的口径。
    """
    dcg = 0.0
    seen: set[str] = set()
    for i, it in enumerate(ranked[:k], start=1):
        groups = set(it.get("gold_groups") or ([it["group"]] if it.get("group") else []))
        new_groups = groups & gold - seen
        if not new_groups:
            continue
        seen |= new_groups
        dcg += 1.0 / math.log2(i + 1)
    # 理想情况：gold 组在前 |gold| 位各命中一次
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    if ideal <= 0:
        return 0.0
    return dcg / ideal


def evaluate(ranked_lists: list[tuple[list[dict], set[str]]], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """对一批查询做整体评测，返回各指标的均值。"""
    n = len(ranked_lists) or 1
    out: dict = {"n_queries": len(ranked_lists)}
    for k in ks:
        out[f"recall@{k}"] = round(sum(recall_at_k(r, g, k) for r, g in ranked_lists) / n, 4)
        out[f"ndcg@{k}"] = round(sum(ndcg_at_k(r, g, k) for r, g in ranked_lists) / n, 4)
    out["mrr"] = round(sum(mrr(r, g) for r, g in ranked_lists) / n, 4)
    return out
