"""Demo 1 核心：向量召回 + 规则重排。

两段式结构（这也是工程上最常用、性价比最高的一种召回-排序方案）：

    第一阶段 召回（recall）
        用 TF-IDF 稀疏向量做余弦相似度，取 top-N 作为候选池。
        目标只有一个：别把正确答案漏掉，所以 N 开得比最终展示条数大得多。

    第二阶段 重排（rerank）
        对候选池里的每条工单，额外算几路"可解释的特征"，再加权融合。
        这一步的目标是把正确答案顶到第一位。

为什么不直接上向量模型？
    - 演示环境往往没有 GPU、也不方便下载模型权重，TF-IDF 是零依赖的 baseline；
    - 更重要的是：纯相似度会把"字面很像但答案不同"的工单排到前面（比如
      "充电宝充不进电" vs "充电宝给手机充不进电"），
      这属于排序问题，不是召回问题——靠业务规则修正比换更大的模型更直接。
"""

from __future__ import annotations

from collections import Counter

from common.data_retrieval import EVAL_QUERIES, TICKETS
from common.metrics import evaluate
from common.text import STOPWORDS, char_ngrams, tokenize
from common.tfidf import BM25, TfidfVectorizer, sparse_dot

# 转折连词：出现在"能给手机供电，但是自己充不进电"这类复合症状里。
# 用户描述故障时经常先说一个正常的部分，再接一个异常的部分，
# 而真正要解决的是转折之后那半句。
CONTRAST_MARKERS = ["但是", "但", "不过", "然而", "可是", "可", "却", "结果"]


def split_contrast(query: str) -> tuple[str, str] | None:
    """把查询按转折连词切成 (前半句, 后半句)；没有转折词返回 None。"""
    for mark in CONTRAST_MARKERS:
        idx = query.find(mark)
        if idx > 0 and idx + len(mark) < len(query) - 1:
            return query[:idx], query[idx + len(mark):]
    return None


DEFAULT_WEIGHTS: dict[str, float] = {
    # 基础相似度（TF-IDF 余弦），占大头但不能全给它
    "sim": 0.45,
    # BM25 关键词命中强度
    "bm25": 0.12,
    # 查询词在正文里的覆盖率：反过来说就是惩罚"只蹭到一个词"的文档
    "coverage": 0.13,
    # 标题词覆盖：客服工单标题是人工写的摘要，
    # "查到的词是否落在标题上"比"字符二元组整体像不像"更能反映主题一致
    "title": 0.16,
    # 分类亲和度：同分类加分
    "cat": 0.04,
    # 历史处理质量：被用户认可过的工单优先
    "helpful": 0.02,
    # 转折从句对齐：复合症状里"转折之后那半句"才是故障点，见 split_contrast()
    "clause": 0.12,
    # 同一答案组重复出现的惩罚，保证列表多样性
    "dup": 0.08,
}


class RetrievalPipeline:
    def __init__(self, tickets: list[dict] | None = None, weights: dict | None = None,
                 pool_size: int = 10):
        self.tickets = tickets if tickets is not None else TICKETS
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.pool_size = pool_size
        self._build()

    # ------------------------------------------------------------------
    # 建索引
    # ------------------------------------------------------------------
    def _build(self) -> None:
        # 正文 + 标题拼在一起做索引，标题天然获得更高词频
        self.docs = [f"{t['title']}。{t['text']}" for t in self.tickets]
        self.vectorizer = TfidfVectorizer(use_char_ngram=True).fit(self.docs)
        self.doc_vecs = self.vectorizer.transform_many(self.docs)
        self.bm25 = BM25().fit(self.docs)
        # 预存 token，重排时反复用
        self.doc_tokens = [set(tokenize(d)) for d in self.docs]
        self.doc_title_tokens = [set(tokenize(t["title"])) for t in self.tickets]

    # ------------------------------------------------------------------
    # 特征
    # ------------------------------------------------------------------
    def _features(self, query: str, q_tokens: list[str], q_vec: dict, doc_idx: int) -> dict:
        doc = self.tickets[doc_idx]
        toks = self.doc_tokens[doc_idx]

        sim = sparse_dot(q_vec, self.doc_vecs[doc_idx])
        bm25_raw = self.bm25.score(query, doc_idx)
        bm25_norm = min(bm25_raw / self.bm25.max_score(query), 1.0)

        # ---- 覆盖率：查询的"实词"有多少出现在这条工单里 ----
        # 停用词不直接丢弃，而是降权：中文短句里"钱""要"这类词看着像停用词，
        # 但往往正是它们区分了"寄回去要钱吗"和"寄回去怎么寄"。
        content = [t for t in q_tokens if not t.startswith("#")]
        weights = {t: (0.35 if t in STOPWORDS else 1.0) for t in set(content)}

        def weighted_cover(doc_toks: set[str]) -> float:
            num = den = 0.0
            for t, w in weights.items():
                den += w
                if t in doc_toks:
                    num += w
            return num / den if den else 0.0

        coverage = weighted_cover(toks)
        title_cover = weighted_cover(self.doc_title_tokens[doc_idx])
        # 再叠一层字符二元组相似度，抗错别字和口语变体（"充不进电" vs "无法充电"）
        q_grams = set(char_ngrams(query, 2))
        t_grams = set(char_ngrams(doc["title"], 2))
        title_gram = len(q_grams & t_grams) / len(q_grams) if q_grams else 0.0
        title_hit = 0.6 * title_cover + 0.4 * title_gram

        # 分类亲和度：查询里直接出现了分类名（如"物流""发票"）就加分
        cat_hit = 1.0 if doc["cat"] in query else 0.0

        # ---- 转折从句对齐 ----
        # "能给手机充电，但是自己充不进去电"里，前半句只是背景、后半句才是故障。
        # 只看整体相似度的话，一条"只讲了前半句"的工单会拿到虚高的分数。
        clause = 0.0
        parts = split_contrast(query)
        if parts:
            head_cov = self._weighted_cover(parts[0], toks)
            tail_cov = self._weighted_cover(parts[1], toks)
            # 后半句覆盖得比前半句好 → 加分；只覆盖前半句而漏掉后半句 → 减分
            clause = tail_cov - head_cov

        missed = sorted([t for t in set(content) if t not in toks and t not in STOPWORDS])
        return {
            "sim": round(sim, 4),
            "bm25": round(bm25_norm, 4),
            "coverage": round(coverage, 4),
            "title": round(title_hit, 4),
            "cat": cat_hit,
            "helpful": 1.0 if doc.get("helpful") else 0.0,
            "clause": round(clause, 4),
            "bm25_raw": round(bm25_raw, 3),
            "covered_terms": sorted({t for t in set(content) if t in toks and t not in STOPWORDS}),
            "missed_terms": missed[:6],
        }

    @staticmethod
    def _weighted_cover(text: str, doc_toks: set[str]) -> float:
        """一小段文本里的实词有多少出现在文档中（停用词降权而非丢弃）。"""
        toks = [t for t in tokenize(text) if not t.startswith("#")]
        weights = {t: (0.35 if t in STOPWORDS else 1.0) for t in set(toks)}
        num = den = 0.0
        for t, w in weights.items():
            den += w
            if t in doc_toks:
                num += w
        return num / den if den else 0.0

    # ------------------------------------------------------------------
    # 召回
    # ------------------------------------------------------------------
    def recall(self, query: str, top_n: int | None = None) -> list[dict]:
        top_n = top_n or self.pool_size
        q_vec = self.vectorizer.transform(query)
        q_tokens = tokenize(query)
        scored = []
        for i in range(len(self.tickets)):
            feats = self._features(query, q_tokens, q_vec, i)
            base = self.weights["sim"] * feats["sim"] + self.weights["bm25"] * feats["bm25"]
            scored.append({"index": i, "recall_score": round(base, 4), "features": feats})
        scored.sort(key=lambda x: (-x["recall_score"], x["index"]))
        return scored[:top_n]

    # ------------------------------------------------------------------
    # 重排
    # ------------------------------------------------------------------
    def rerank(self, query: str, candidates: list[dict] | None = None,
               top_k: int = 5) -> list[dict]:
        candidates = candidates if candidates is not None else self.recall(query)
        seen_groups: dict[str, int] = {}
        out: list[dict] = []
        # 先按基础分过一遍，再按最终分排序；dup 惩罚依赖"已出现过哪些答案组"，
        # 所以要在打分过程中动态统计，这里先按基础分粗排以保证惩罚顺序稳定。
        for cand in sorted(candidates, key=lambda c: -c["recall_score"]):
            i = cand["index"]
            doc = self.tickets[i]
            f = cand["features"]
            dup_count = seen_groups.get(doc["group"], 0)
            seen_groups[doc["group"]] = dup_count + 1

            score = (
                self.weights["sim"] * f["sim"]
                + self.weights["bm25"] * f["bm25"]
                + self.weights["coverage"] * f["coverage"]
                + self.weights["title"] * f["title"]
                + self.weights["cat"] * f["cat"]
                + self.weights["helpful"] * f["helpful"]
                + self.weights["clause"] * f["clause"]
                - self.weights["dup"] * dup_count
            )
            out.append({
                "_index": i,
                "id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "cat": doc["cat"],
                "group": doc["group"],
                "helpful": bool(doc.get("helpful")),
                "final_score": round(score, 4),
                "recall_score": cand["recall_score"],
                "rank_delta": None,  # 下面统一算
                "features": f,
                "contrib": {
                    k: round(self.weights[k] * f[k], 4)
                    for k in ("sim", "bm25", "coverage", "title", "cat", "helpful", "clause")
                },
            })

        out.sort(key=lambda x: (-x["final_score"], x["id"]))
        # 记录重排相对召回的排名变化，界面上一眼能看出重排干了什么
        recall_order = {c["index"]: pos + 1 for pos, c in enumerate(candidates)}
        for pos, item in enumerate(out, start=1):
            item["rank_delta"] = recall_order.get(item.pop("_index"), 0) - pos
        return out[:top_k]

    # ------------------------------------------------------------------
    # 对外一步到位
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, use_rerank: bool = True) -> dict:
        pool = self.recall(query)
        if use_rerank:
            ranked = self.rerank(query, pool, top_k=top_k)
            ranked_ids = {r["id"] for r in ranked}
            # 界面需要展示"重排前是什么样"，这里把纯召回的 top_k 也带上
            recall_topk = []
            for c in pool[:top_k]:
                d = self.tickets[c["index"]]
                recall_topk.append({
                    "id": d["id"], "title": d["title"], "group": d["group"],
                    "score": c["recall_score"],
                })
        else:
            ranked = []
            for c in pool[:top_k]:
                d = self.tickets[c["index"]]
                ranked.append({
                    "id": d["id"], "title": d["title"], "text": d["text"], "cat": d["cat"],
                    "group": d["group"], "helpful": bool(d.get("helpful")),
                    "final_score": c["recall_score"], "recall_score": c["recall_score"],
                    "rank_delta": 0, "features": c["features"], "contrib": {},
                })
            ranked_ids = set()
            recall_topk = []

        return {
            "query": query,
            "tokens": tokenize(query),
            "pool_size": len(pool),
            "results": ranked,
            "recall_topk": recall_topk,
            "changed": [r["id"] for r in ranked if r.get("rank_delta")],
            "reranked_ids": sorted(ranked_ids),
        }

    # ------------------------------------------------------------------
    # 离线评测
    # ------------------------------------------------------------------
    def evaluate(self, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
        """对比"只召回"和"召回+重排"两套排序的指标。"""
        recall_only, reranked = [], []
        per_query = []
        for item in EVAL_QUERIES:
            q, gold = item["q"], set(item["gold"])
            pool = self.recall(q)
            base_ranked = [
                {**self.tickets[c["index"]], "score": c["recall_score"]} for c in pool
            ]
            rr = self.rerank(q, pool, top_k=5)
            recall_only.append((base_ranked, gold))
            reranked.append((rr, gold))

            rk = next((i for i, r in enumerate(rr, 1) if r["group"] in gold), None)
            per_query.append({
                "query": q,
                "gold": sorted(gold),
                "difficulty": item.get("difficulty"),
                "top1": rr[0]["title"] if rr else "",
                "top1_hit": bool(rr and rr[0]["group"] in gold),
                "rank_of_gold": rk,
                "baseline_top1": base_ranked[0]["title"] if base_ranked else "",
                "baseline_top1_hit": bool(base_ranked and base_ranked[0]["group"] in gold),
            })
        changed_top1 = [p for p in per_query if p["top1"] != p["baseline_top1"]]
        improved = sum(1 for p in changed_top1 if p["top1_hit"] and not p["baseline_top1_hit"])
        worsened = sum(1 for p in changed_top1 if p["baseline_top1_hit"] and not p["top1_hit"])
        failed = [p["query"] for p in per_query if not p["top1_hit"]]
        return {
            "baseline": evaluate(recall_only, ks=ks),
            "rerank": evaluate(reranked, ks=ks),
            "per_query": per_query,
            "weights": self.weights,
            "n_docs": len(self.tickets),
            "flip": {
                "n": len(changed_top1),
                "improved": improved,
                "worsened": worsened,
                "unchanged": len(per_query) - len(changed_top1),
            },
            "failed_queries": failed,
            "limitation": (
                "重排只改变候选池内部的顺序，救不回召回阶段就漏掉的答案；"
                "而且它靠的是可解释的词汇级规则，遇到'扫地机在地上转圈圈不干活'"
                "这种需要常识才能对应到'边刷缠绕'的说法仍然会排错——"
                "这类查询才是真正的语义鸿沟，也是下一步该引入向量模型或 LLM 重排的地方。"
            ),
        }


if __name__ == "__main__":  # 手动跑一下看效果
    import json

    print(json.dumps(RetrievalPipeline().evaluate(), ensure_ascii=False, indent=2))
