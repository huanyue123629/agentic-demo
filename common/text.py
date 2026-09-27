"""极简中文文本处理工具。

设计原则：只依赖 Python 标准库，保证任何人 clone 下来直接能跑，
不需要装任何第三方包。如果环境里有 jieba，会自动启用更好的分词。

对外提供：
    tokenize(text)            -> list[str]   分词
    char_ngrams(text, n)      -> list[str]   字符 n-gram
    l2_normalize(vec)         -> list[float] L2 归一化
    cosine(a, b)              -> float       余弦相似度
"""

from __future__ import annotations

import math
import re
from collections import Counter

# 有 jieba 就用 jieba，没有就退回纯字符 n-gram，行为保持一致
try:  # pragma: no cover - 取决于运行环境
    import jieba  # type: ignore

    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    jieba = None  # type: ignore
    _HAS_JIEBA = False

_PUNCT_RE = re.compile(r"[\s,，。！？!?、；;：:（）()\[\]【】\"'“”‘’—\-_/\\|~`@#$%^&*+=<>《》…·]+")
_NUM_RE = re.compile(r"\d+")

# 常见疑问词与口水词：它们对"这条问题和哪条历史工单像"几乎没有区分度，
# 但会稀释向量，所以单独维护一张停用词表。
STOPWORDS = {
    "的", "了", "是", "我", "我们", "你", "你们", "他", "她", "它", "这", "那",
    "这个", "那个", "一下", "怎么", "怎样", "如何", "什么", "为什么", "哪里",
    "可以", "能否", "请问", "请", "帮", "帮我", "想要", "想", "要", "有", "没有",
    "吗", "呢", "啊", "吧", "呀", "哦", "嗯", "和", "跟", "与", "就", "都", "还",
    "在", "到", "给", "让", "被", "把", "对", "从", "为", "会", "能", "需要",
}


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """把一句话切成 token 列表（已小写化、去标点）。

    中文部分优先用 jieba；英文/数字用正则单独切出来。
    """
    if not text:
        return []
    text = text.strip()
    tokens: list[str] = []

    if _HAS_JIEBA:
        for tok in jieba.lcut(text):
            tok = tok.strip().lower()
            tok = _PUNCT_RE.sub("", tok)
            if not tok:
                continue
            if not keep_stopwords and tok in STOPWORDS:
                continue
            tokens.append(tok)
    else:
        # 无 jieba 时的退化方案：把连续汉字按 1~2 字切，英文数字按整词切
        for chunk in _PUNCT_RE.split(text.lower()):
            if not chunk:
                continue
            if re.fullmatch(r"[a-z0-9.]+", chunk):
                if keep_stopwords or chunk not in STOPWORDS:
                    tokens.append(chunk)
                continue
            # 汉字串：单字 + 相邻双字，兼顾召回与顺序信息
            chars = [c for c in chunk if not c.isspace()]
            for i, c in enumerate(chars):
                if keep_stopwords or c not in STOPWORDS:
                    tokens.append(c)
                if i + 1 < len(chars):
                    bigram = chars[i] + chars[i + 1]
                    if keep_stopwords or bigram not in STOPWORDS:
                        tokens.append(bigram)

    # 抽出数字，比如 "7 天" "48 小时"，这类信息在售后场景区分度很高
    for num in _NUM_RE.findall(text):
        tokens.append(f"#{num}")

    return tokens


def char_ngrams(text: str, n: int = 2) -> list[str]:
    """字符级 n-gram，用于抗错别字（"充不进电" vs "充不上电"）。"""
    chars = [c for c in _PUNCT_RE.sub("", text) if not c.isspace()]
    if len(chars) < n:
        return ["".join(chars)] if chars else []
    return ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    # 向量都已 L2 归一化，点积即余弦相似度
    return sum(x * y for x, y in zip(a, b))


def token_f1(pred: list[str], gold: list[str]) -> float:
    """分词结果的 F1，用来做分词质量的小体检（可选）。"""
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p = same / len(pred)
    r = same / len(gold)
    return 2 * p * r / (p + r)
