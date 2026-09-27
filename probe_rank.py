"""对比不同重排策略（开发时用）。

用法：python probe_rank.py
结论会写回 pipeline.py 的默认实现。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "01_recall_rerank")]

from common.data_retrieval import EVAL_QUERIES, TICKETS  # noqa: E402
from common.metrics import evaluate  # noqa: E402
from pipeline import RetrievalPipeline  # noqa: E402


def ranked_from_scores(scores: list[float], k: int = 5) -> list[dict]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [TICKETS[i] for i in order]


def main() -> None:
    pipe = RetrievalPipeline()
    base_lists, rrf_lists, cur_lists = [], [], []
    detail = []

    for item in EVAL_QUERIES:
        q, gold = item["q"], set(item["gold"])
        pool = pipe.recall(q)
        idxs = [c["index"] for c in pool]

        # 1) 纯向量相似度
        sim_scores = [0.0] * len(TICKETS)
        for c in pool:
            sim_scores[c["index"]] = c["features"]["sim"]
        base = ranked_from_scores(sim_scores)

        # 2) 向量 + BM25 的 RRF 融合（只看名次，不看分数尺度）
        feats = {c["index"]: c["features"] for c in pool}
        sim_order = sorted(idxs, key=lambda i: -feats[i]["sim"])
        bm_order = sorted(idxs, key=lambda i: -feats[i]["bm25_raw"])
        rrf = {i: 0.0 for i in idxs}
        for lst in (sim_order, bm_order):
            for rank, i in enumerate(lst, start=1):
                rrf[i] += 1.0 / (60 + rank)
        rrf_scores = [rrf.get(i, 0.0) for i in range(len(TICKETS))]
        rrf_ranked = ranked_from_scores(rrf_scores)

        # 3) 当前实现
        cur = pipe.rerank(q, pool, top_k=5)

        base_lists.append((base, gold))
        rrf_lists.append((rrf_ranked, gold))
        cur_lists.append((cur, gold))

        r_cur = next((i for i, x in enumerate(cur, 1) if x["group"] in gold), None)
        r_base = next((i for i, x in enumerate(base, 1) if x["group"] in gold), None)
        detail.append((q, r_base, r_cur, base[0]["title"], cur[0]["title"]))

    print("纯相似度      :", evaluate(base_lists))
    print("RRF 融合      :", evaluate(rrf_lists))
    print("当前实现      :", evaluate(cur_lists))
    print()
    for item, (q, rb, rc, tb, tc) in zip(EVAL_QUERIES, detail):
        pool = pipe.recall(q)
        feats = {c["index"]: c["features"] for c in pool}
        idxs = [c["index"] for c in pool]
        top3 = sorted(idxs, key=lambda i: -feats[i]["sim"])[:3]
        top3_txt = " | ".join(f"{TICKETS[i]['id']}:{TICKETS[i]['title'][:14]}" for i in top3)
        flag = "OK " if rc == 1 else "MISS"
        print(f"[{flag}] {q[:26]:<28} gold={item['gold'][0]}")
        print(f"        相似度Top3: {top3_txt}")
        print(f"        最终Top1  : {tc}   (相似度排名#{rb} → 重排排名#{rc})")


if __name__ == "__main__":
    main()
