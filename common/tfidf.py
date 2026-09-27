"""从零实现的 TF-IDF 向量化 + BM25 词权重。

只用标准库，行数不多，但每一步都写清楚，方便在方案文档里解释"为什么有效"。

TF-IDF:  tf(t,d) = 该 token 在文档 d 中出现的次数
         idf(t)  = ln((1 + N) / (1 + df(t))) + 1        （平滑，避免除零）
         w(t,d)  = (1 + ln tf) * idf(t)                  （对数饱和，抑制长文档）
BM25:    用于给"关键词完全命中"额外加权，和向量相似度互补。
"""

from __future__ import annotations

import math
from collections import Counter

from .text import l2_normalize, tokenize


class TfidfVectorizer:
    """把文本转成稀疏向量（dict 形式），并做 L2 归一化。"""

    def __init__(self, use_char_ngram: bool = True, min_df: int = 1):
        self.use_char_ngram = use_char_ngram
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.doc_freq: Counter = Counter()
        self.n_docs = 0

    # ---------- 内部工具 ----------
    def _analyze(self, text: str) -> list[str]:
        from .text import char_ngrams

        toks = tokenize(text)
        if self.use_char_ngram:
            # 少量字符二元组即可覆盖错别字/口语变体，比例过高会淹没词级信息
            toks = toks + char_ngrams(text, 2)
        return toks

    # ---------- 训练 ----------
    def fit(self, corpus: list[str]) -> "TfidfVectorizer":
        self.n_docs = len(corpus)
        self.doc_freq = Counter()
        for doc in corpus:
            for tok in set(self._analyze(doc)):
                self.doc_freq[tok] += 1
        kept = [t for t, df in self.doc_freq.items() if df >= self.min_df]
        # 按 token 排序保证多次运行结果完全一致（可复现）
        self.vocab = {t: i for i, t in enumerate(sorted(kept))}
        n = self.n_docs
        self.idf = [0.0] * len(self.vocab)
        for tok, idx in self.vocab.items():
            df = self.doc_freq[tok]
            self.idf[idx] = math.log((1.0 + n) / (1.0 + df)) + 1.0
        return self

    # ---------- 转换 ----------
    def transform(self, text: str) -> dict[int, float]:
        toks = self._analyze(text)
        if not toks:
            return {}
        tf = Counter(toks)
        raw: dict[int, float] = {}
        for tok, cnt in tf.items():
            idx = self.vocab.get(tok)
            if idx is None:
                continue
            raw[idx] = (1.0 + math.log(cnt)) * self.idf[idx]
        return _normalize_sparse(raw)

    def fit_transform(self, corpus: list[str]) -> list[dict[int, float]]:
        return self.fit(corpus).transform_many(corpus)

    def transform_many(self, corpus: list[str]) -> list[dict[int, float]]:
        return [self.transform(doc) for doc in corpus]


def _normalize_sparse(vec: dict[int, float]) -> dict[int, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 1e-12:
        return vec
    return {k: v / norm for k, v in vec.items()}


def sparse_dot(a: dict[int, float], b: dict[int, float]) -> float:
    """稀疏点积：两个向量都已 L2 归一化时，结果就是余弦相似度。"""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def dense_from_sparse(vec: dict[int, float], dim: int) -> list[float]:
    out = [0.0] * dim
    for k, v in vec.items():
        out[k] = v
    return out


def normalize_dense(vec: list[float]) -> list[float]:
    return l2_normalize(vec)


class BM25:
    """标准 BM25，用于给重排阶段提供"关键词命中"信号。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avg_len: float = 1.0
        self.idf: dict[str, float] = {}

    def fit(self, corpus: list[str]) -> "BM25":
        self.doc_tokens = [tokenize(doc) for doc in corpus]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        n = len(self.doc_tokens)
        df: Counter = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        self.idf = {
            t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }
        return self

    def score(self, query: str, doc_idx: int) -> float:
        if not self.doc_tokens:
            return 0.0
        toks = tokenize(query)
        if not toks:
            return 0.0
        dl = self.doc_len[doc_idx] or 1
        tf = Counter(self.doc_tokens[doc_idx])
        total = 0.0
        for t in toks:
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            total += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg_len))
        return total

    def max_score(self, query: str) -> float:
        """当前语料下的近似上界，用于把 BM25 归一到 0~1。"""
        if not self.doc_tokens:
            return 1.0
        toks = tokenize(query)
        if not toks:
            return 1.0
        upper = sum(self.idf.get(t, 0.0) for t in set(toks)) * (self.k1 + 1)
        return max(upper, 1e-6)
