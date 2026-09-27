"""端到端自检：两个 demo 的核心逻辑都跑一遍，不需要装任何依赖。

用法（在 demo 目录下）：
    python smoke_test.py
"""

from __future__ import annotations

import os
import sys
import traceback

# 两个 demo 是独立目录，这里把它们的路径都加进来，
# 保证在任意工作目录下直接 `python smoke_test.py` 都能跑。
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "01_recall_rerank"), os.path.join(_HERE, "02_intent_classifier")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # Windows 控制台默认码页可能不是 UTF-8，先纠正，避免中文输出乱码
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

PASS, FAIL = [], []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        PASS.append((name, detail))
        print(f"  [PASS] {name}  {detail}")
    except Exception as exc:  # noqa: BLE001
        FAIL.append((name, exc))
        print(f"  [FAIL] {name}  {exc.__class__.__name__}: {exc}")
        traceback.print_exc()


def main() -> int:
    from common.data_intent import LABELS, TEST, TRAIN
    from common.data_retrieval import EVAL_QUERIES, TICKETS
    from common.text import tokenize
    from common.tfidf import TfidfVectorizer, sparse_dot
    from pipeline import RetrievalPipeline
    from classifier import train_all, cross_validate

    print("== 数据完整性 ==")

    def data_ok():
        ids = [t["id"] for t in TICKETS]
        assert len(ids) == len(set(ids)), "工单 id 重复"
        for item in EVAL_QUERIES:
            assert item["gold"], f"评测查询缺少 gold: {item['q']}"
        for label, sents in TRAIN.items():
            assert label in LABELS, label
            assert len(sents) >= 10, f"{label} 样本太少"
        for label, sents in TEST.items():
            assert label in LABELS, label
        train_all = {s for v in TRAIN.values() for s in v}
        test_all = {s for v in TEST.values() for s in v}
        overlap = train_all & test_all
        assert not overlap, f"训练集与测试集有重叠: {overlap}"
        return f"{len(TICKETS)} 条工单 / {len(EVAL_QUERIES)} 条评测查询 / {len(TRAIN)} 类意图"

    check("数据无重复、训练测试不重叠", data_ok)

    print("== 文本与向量工具 ==")

    def tfidf_ok():
        docs = ["充电宝充不进电", "耳机只有一边有声音", "怎么开发票"]
        vec = TfidfVectorizer().fit(docs)
        v0 = vec.transform("充电宝充不进电")
        v1 = vec.transform("耳机只有一边有声音")
        same = sparse_dot(v0, v0)
        cross = sparse_dot(v0, v1)
        assert abs(same - 1.0) < 1e-6, f"自相似度应为 1，实际 {same}"
        assert cross < 0.2, f"跨主题相似度应很低，实际 {cross}"
        return f"vocab={len(vec.vocab)} 自相似={same:.3f} 跨主题={cross:.3f}"

    check("TF-IDF 向量归一化与相似度", tfidf_ok)

    def tokenize_ok():
        toks = tokenize("充电宝插上充电器灯不亮")
        assert toks, "分词结果为空"
        assert len(toks) >= 4, f"分词过少: {toks}"
        return f"{len(toks)} tokens, 例: {toks[:5]}"

    check("中文分词", tokenize_ok)

    print("== Demo 1：召回 + 重排 ==")

    def pipeline_ok():
        pipe = RetrievalPipeline()
        res = pipe.search("我要退货，钱什么时候退给我", top_k=3)
        assert res["results"], "没有检索结果"
        assert len(res["results"]) == 3
        return f"top1={res['results'][0]['id']} {res['results'][0]['title']}"

    check("检索接口可用", pipeline_ok)

    def eval_ok():
        ev = RetrievalPipeline().evaluate()
        assert ev["rerank"]["recall@1"] >= ev["baseline"]["recall@1"], (
            f"重排后 Recall@1 反而下降: {ev['baseline']['recall@1']} -> {ev['rerank']['recall@1']}"
        )
        assert ev["rerank"]["recall@1"] >= 0.6, f"Recall@1 偏低: {ev['rerank']['recall@1']}"
        return (f"baseline R@1={ev['baseline']['recall@1']} MRR={ev['baseline']['mrr']} "
                f"-> rerank R@1={ev['rerank']['recall@1']} MRR={ev['rerank']['mrr']}")

    check("重排不劣于纯召回，且 Recall@1 达标", eval_ok)

    print("== Demo 2：意图分类 ==")

    def train_ok():
        res = train_all(quick=True)
        rep = res["report"]
        assert res["n_train"] > res["n_test"]
        for key in ("keyword", "logreg", "mlp"):
            assert key in rep, key
        assert rep["logreg"]["accuracy"] >= 0.6, f"逻辑回归准确率过低: {rep['logreg']['accuracy']}"
        for key, r in rep.items():
            for cls, m in r["per_class"].items():
                for f in ("precision", "recall", "f1"):
                    assert 0.0 <= m[f] <= 1.0, f"{key}.{cls}.{f}={m[f]}"
        return (f"keyword={rep['keyword']['accuracy']} "
                f"logreg={rep['logreg']['accuracy']} mlp={rep['mlp']['accuracy']}")

    check("三个模型训练与指标计算", train_ok)

    def cv_ok():
        cv = cross_validate(k=4)
        assert cv["logreg"]["mean"] > 0.4, f"交叉验证准确率异常: {cv['logreg']}"
        return f"4 折 CV logreg={cv['logreg']['mean']}±{cv['logreg']['std']} mlp={cv['mlp']['mean']}"

    check("分层交叉验证", cv_ok)

    print()
    print(f"结果：{len(PASS)} 项通过，{len(FAIL)} 项失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
