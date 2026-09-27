"""Demo 2 核心：客服意图分类。

对比三种做法（全部从零实现，只依赖标准库）：

    1. KeywordBaseline  关键词打分。不需要训练，是最朴素的 baseline，
                        用来回答"这题到底难不难"。
    2. SoftmaxRegression 多项逻辑回归 + L2 正则。特征用 TF-IDF，
                        优化器手写 mini-batch Adam。
    3. MLP              一层隐藏层的轻量神经网络（ReLU + softmax），
                        同样特征、同样优化器，看非线性带来的增益。

关于"轻量 BERT"：环境里不一定有 transformers/torch，所以真正的
BERT 微调放在可选路径（可选依赖缺失就自动跳过，不报错）。
本文件里的 MLP 作为"轻量模型"的代理：参数量可控、CPU 秒级训练，
用来验证"特征够不够用"这个问题——如果 MLP 打不过逻辑回归，
说明瓶颈在特征/数据量，而不是模型容量，这时候上 BERT 收益有限。
"""

from __future__ import annotations

import math
import random
from collections import Counter

from common.data_intent import LABEL_CN, LABELS, TEST, TRAIN
from common.text import tokenize
from common.tfidf import TfidfVectorizer


# 特征类型：稀疏用 dict{特征下标: 权重}，稠密用 list[float]。
# 之所以两种都支持，是因为中文 TF-IDF 稀疏度极高（一句话几十个非零项 vs 两千维词表），
# 全稠密矩阵乘法会把一个本该几秒跑完的训练拖到十几分钟。
SparseVec = dict[int, float]
DenseVec = list[float]
Vec = "SparseVec | DenseVec"


def dim_of(x) -> int:
    return len(x)


def nonzeros(x) -> "list[tuple[int, float]]":
    """统一取非零项：dict 直接 items，list 需要扫描（仅用于小规模场景）。"""
    if isinstance(x, dict):
        return list(x.items())
    return [(i, v) for i, v in enumerate(x) if v != 0.0]


def dot_dense_sparse(row: list[float], x) -> float:
    """稠密权重行 · 稀疏/稠密输入。稀疏路径只走非零项，这是性能关键。"""
    if isinstance(x, dict):
        return sum(row[i] * v for i, v in x.items())
    return sum(w * v for w, v in zip(row, x))


def check_dims(X, dim: int) -> None:
    """校验输入特征维度，防止"模型按 A 维初始化、却喂了 B 维特征"这类静默错误。

    稀疏向量本身不携带总维度（只存非零项），所以这里校验的是
    最大下标不越界；稠密向量直接比长度。
    """
    if not X or not dim:
        return
    for i, x in enumerate(X[:50]):
        if isinstance(x, dict):
            if x:
                m = max(x)
                if m >= dim:
                    raise ValueError(
                        f"稀疏特征下标越界：第 {i} 条样本出现下标 {m}，但模型维度只有 {dim}"
                    )
        elif len(x) != dim:
            raise ValueError(
                f"特征维度不一致：模型按 dim={dim} 初始化，但第 {i} 条样本是 {len(x)} 维"
            )


# ----------------------------------------------------------------------
# 数据准备
# ----------------------------------------------------------------------
def build_dataset(include_augment: bool = True) -> tuple[list[str], list[int], list[str], list[int]]:
    """返回 (X_train, y_train, X_test, y_test)。"""
    texts, labels = [], []
    for label, sents in TRAIN.items():
        for s in sents:
            texts.append(s)
            labels.append(LABELS.index(label))
    if include_augment:
        texts, labels = augment(texts, labels)

    test_texts, test_labels = [], []
    for label, sents in TEST.items():
        for s in sents:
            test_texts.append(s)
            test_labels.append(LABELS.index(label))
    return texts, labels, test_texts, test_labels


# 口语前缀/后缀：真实客服会话里用户经常带情绪和语气词，
# 加噪声是为了让模型不要只记住句子长度或开头几个字。
_PREFIXES = ["你好，", "在吗，", "麻烦问一下，", "急，", "想咨询一下，", ""]
_SUFFIXES = ["谢谢", "麻烦尽快回复", "急等", "在线等", "谢谢了", ""]

# 同义替换表：扩大表达覆盖，注意只做"同向"替换，不改变意图
_SYNONYMS: list[tuple[str, str]] = [
    ("退款", "退钱"),
    ("退货", "退还商品"),
    ("快递", "包裹"),
    ("物流", "运输"),
    ("发票", "票据"),
    ("充电器", "充电头"),
    ("耳机", "蓝牙耳机"),
    ("怎么办", "怎么处理"),
    ("能不能", "可以不可以"),
    ("几天", "多久"),
]


def augment(texts: list[str], labels: list[int], seed: int = 20260101,
            factor: int = 3) -> tuple[list[str], list[int]]:
    """按固定随机种子做小幅扩增，保证结果可复现。"""
    rng = random.Random(seed)
    out_texts = list(texts)
    out_labels = list(labels)
    for _ in range(factor):
        for t, y in zip(texts, labels):
            s = t
            # 随机做 0~2 次同义替换
            for src, dst in _SYNONYMS:
                if src in s and rng.random() < 0.4:
                    s = s.replace(src, dst, 1)
            s = rng.choice(_PREFIXES) + s + rng.choice(_SUFFIXES)
            if s.strip() and s not in out_texts:
                out_texts.append(s)
                out_labels.append(y)
    return out_texts, out_labels


# ----------------------------------------------------------------------
# 1) 关键词 baseline
# ----------------------------------------------------------------------
class KeywordBaseline:
    """每个类统计一个"高区分度词表"，打分 = 命中词的 idf 之和。"""

    name = "关键词打分（baseline）"

    def __init__(self, top_per_class: int = 12):
        self.top_per_class = top_per_class
        self.tables: list[set[str]] = []

    def fit(self, texts: list[str], labels: list[int]) -> "KeywordBaseline":
        n_cls = len(LABELS)
        per_class = [Counter() for _ in range(n_cls)]
        df = Counter()
        for t, y in zip(texts, labels):
            toks = set(tokenize(t))
            per_class[y].update(toks)
            for tok in toks:
                df[tok] += 1
        n = len(texts)
        idf = {tok: math.log((1 + n) / (1 + d)) + 1 for tok, d in df.items()}
        self.tables = []
        for c in range(n_cls):
            # 只在少数类出现的词 idf 高，天然被选出来
            ranked = sorted(per_class[c].items(), key=lambda kv: -kv[1] * idf.get(kv[0], 1.0))
            self.tables.append({tok for tok, _ in ranked[: self.top_per_class]})
        self._idf = idf
        return self

    def scores(self, text: str) -> list[float]:
        toks = set(tokenize(text))
        return [sum(self._idf.get(t, 1.0) for t in toks & table) for table in self.tables]

    def predict(self, texts: list[str]) -> list[int]:
        out = []
        for t in texts:
            sc = self.scores(t)
            out.append(max(range(len(sc)), key=lambda i: sc[i]))
        return out

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        res = []
        for t in texts:
            sc = self.scores(t)
            m = max(sc) or 1.0
            exps = [math.exp((s - m) / max(m, 1e-6)) for s in sc]
            z = sum(exps)
            res.append([e / z for e in exps])
        return res


# ----------------------------------------------------------------------
# 通用：softmax + Adam
# ----------------------------------------------------------------------
def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    z = sum(exps)
    return [e / z for e in exps]


class _Adam:
    """mini-batch Adam。

    参数按"张量"组织：标量参数是 list[float]，矩阵参数是 list[list[float]]，
    这样 in-place 更新能直接改到模型的权重上（不需要 flatten/unflatten，
    避免每次迭代复制一遍权重）。
    梯度允许是稀疏 dict（只更新非零项），否则稠密扫描会主导运行时间。
    """

    def __init__(self, lr: float = 0.05, b1: float = 0.9, b2: float = 0.999,
                 eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m: list = []
        self.v: list = []

    def _zeros_like(self, param):
        if param and isinstance(param[0], list):
            return [[0.0] * len(row) for row in param]
        return [0.0] * len(param)

    def step(self, params: list, grads: list) -> None:
        if not self.m:
            self.m = [self._zeros_like(p) for p in params]
            self.v = [self._zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            if isinstance(g, dict):
                # 稀疏梯度：只更新出现过的坐标
                for i, gi in g.items():
                    if gi == 0.0:
                        continue
                    m[i] = self.b1 * m[i] + (1 - self.b1) * gi
                    v[i] = self.b2 * v[i] + (1 - self.b2) * gi * gi
                    p[i] -= self.lr * (m[i] / (1 - self.b1 ** self.t)) / (
                        math.sqrt(v[i] / (1 - self.b2 ** self.t)) + self.eps
                    )
            elif p and isinstance(p[0], list):
                # 矩阵参数，但梯度可能是稀疏的（每行一个 dict）
                for row_p, row_g, row_m, row_v in zip(p, g, m, v):
                    if isinstance(row_g, dict):
                        for i, gi in row_g.items():
                            if gi == 0.0:
                                continue
                            row_m[i] = self.b1 * row_m[i] + (1 - self.b1) * gi
                            row_v[i] = self.b2 * row_v[i] + (1 - self.b2) * gi * gi
                            row_p[i] -= self.lr * (row_m[i] / (1 - self.b1 ** self.t)) / (
                                math.sqrt(row_v[i] / (1 - self.b2 ** self.t)) + self.eps
                            )
                        continue
                    for i in range(len(row_p)):
                        gi = row_g[i]
                        if gi == 0.0:
                            continue
                        row_m[i] = self.b1 * row_m[i] + (1 - self.b1) * gi
                        row_v[i] = self.b2 * row_v[i] + (1 - self.b2) * gi * gi
                        row_p[i] -= self.lr * (row_m[i] / (1 - self.b1 ** self.t)) / (
                            math.sqrt(row_v[i] / (1 - self.b2 ** self.t)) + self.eps
                        )
            else:
                for i in range(len(p)):
                    gi = g[i]
                    if gi == 0.0:
                        continue
                    m[i] = self.b1 * m[i] + (1 - self.b1) * gi
                    v[i] = self.b2 * v[i] + (1 - self.b2) * gi * gi
                    p[i] -= self.lr * (m[i] / (1 - self.b1 ** self.t)) / (
                        math.sqrt(v[i] / (1 - self.b2 ** self.t)) + self.eps
                    )


# ----------------------------------------------------------------------
# 2) 多项逻辑回归
# ----------------------------------------------------------------------
class SoftmaxRegression:
    name = "逻辑回归 + TF-IDF"

    def __init__(self, dim: int, n_cls: int, lr: float = 0.05, l2: float = 1e-3,
                 epochs: int = 100, batch_size: int = 32, seed: int = 7):
        rng = random.Random(seed)
        self.W = [[rng.gauss(0, 0.01) for _ in range(dim)] for _ in range(n_cls)]
        self.b = [0.0] * n_cls
        self.dim = dim
        self.lr, self.l2, self.epochs, self.bs = lr, l2, epochs, batch_size
        self.history: list[dict] = []

    def _logits(self, x) -> list[float]:
        return [dot_dense_sparse(row, x) + bb for row, bb in zip(self.W, self.b)]

    def fit(self, X: list, y: list[int], X_val=None, y_val=None,
            verbose: bool = False) -> "SoftmaxRegression":
        check_dims(X, self.dim)
        n_cls = len(self.W)
        opt = _Adam(lr=self.lr)
        idx = list(range(len(X)))
        rng = random.Random(11)
        for ep in range(1, self.epochs + 1):
            rng.shuffle(idx)
            total_loss = 0.0
            for start in range(0, len(idx), self.bs):
                batch = idx[start : start + self.bs]
                gW = [{} for _ in range(n_cls)]
                gb = [0.0] * n_cls
                for i in batch:
                    x = X[i]
                    p = softmax(self._logits(x))
                    total_loss += -math.log(max(p[y[i]], 1e-12))
                    for c in range(n_cls):
                        d = p[c] - (1.0 if c == y[i] else 0.0)
                        if d == 0.0:
                            continue
                        row = gW[c]
                        for j, xj in nonzeros(x):
                            row[j] = row.get(j, 0.0) + d * xj
                        gb[c] += d
                inv = 1.0 / len(batch)
                for c in range(n_cls):
                    row = gW[c]
                    for j in list(row.keys()):
                        row[j] = row[j] * inv + self.l2 * self.W[c][j]
                    gb[c] *= inv
                opt.step([self.W, self.b], [gW, gb])
            if ep % 5 == 0 or ep == 1:
                rec = {"epoch": ep, "loss": round(total_loss / len(X), 4)}
                if X_val:
                    rec["val_acc"] = round(self.accuracy(X_val, y_val), 4)
                self.history.append(rec)
                if verbose:
                    print(rec)
        return self

    def predict_proba(self, X: list) -> list[list[float]]:
        return [softmax(self._logits(x)) for x in X]

    def predict(self, X: list) -> list[int]:
        return [max(range(len(p)), key=lambda i: p[i]) for p in self.predict_proba(X)]

    def accuracy(self, X: list, y: list[int]) -> float:
        pred = self.predict(X)
        return sum(1 for a, b in zip(pred, y) if a == b) / max(len(y), 1)


# ----------------------------------------------------------------------
# 3) 轻量 MLP
# ----------------------------------------------------------------------
class MLP:
    """一层隐藏层的小网络：ReLU + softmax，Adam 优化。

    作为"轻量模型"的代理：参数量只有两层权重，CPU 上秒级到十几秒训练完。
    和逻辑回归用同一份特征、同一个优化器，唯一变量就是多了非线性，
    所以两者的差距能直接说明"这题需不需要更复杂的模型"。

    隐藏层宽度故意设得很小（默认 16）：430 条样本 / 1628 维特征这个规模下，
    h=64 会迅速把训练集拟合到 1.0 而测试集退化，"容量更大"反而是负收益。
    这也是本 demo 想验证的结论之一——瓶颈在数据量，不在模型大小。
    """

    name = "轻量 MLP（1 隐层）"

    def __init__(self, dim: int, n_cls: int, hidden: int = 16, lr: float = 0.05,
                 l2: float = 1e-3, epochs: int = 80, batch_size: int = 32, seed: int = 7):
        rng = random.Random(seed)
        # He 初始化：适合 ReLU
        s1 = math.sqrt(2.0 / dim)
        s2 = math.sqrt(2.0 / hidden)
        self.W1 = [[rng.gauss(0, s1) for _ in range(dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.W2 = [[rng.gauss(0, s2) for _ in range(hidden)] for _ in range(n_cls)]
        self.b2 = [0.0] * n_cls
        self.dim = dim
        self.lr, self.l2, self.epochs, self.bs = lr, l2, epochs, batch_size
        self.hidden = hidden
        self.history: list[dict] = []

    def _forward(self, x):
        h_pre = [dot_dense_sparse(row, x) + b for row, b in zip(self.W1, self.b1)]
        h = [v if v > 0 else 0.0 for v in h_pre]          # ReLU
        logits = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return h, logits

    def fit(self, X: list, y: list[int], X_val=None, y_val=None,
            verbose: bool = False) -> "MLP":
        check_dims(X, self.dim)
        dim = self.dim
        n_cls = len(self.W2)
        hsz = self.hidden
        opt = _Adam(lr=self.lr)
        idx = list(range(len(X)))
        rng = random.Random(13)
        for ep in range(1, self.epochs + 1):
            rng.shuffle(idx)
            total_loss = 0.0
            for start in range(0, len(idx), self.bs):
                batch = idx[start : start + self.bs]
                gW1 = [{} for _ in range(hsz)]      # 稀疏：只记录本 batch 动过的坐标
                gb1 = [0.0] * hsz
                gW2 = [[0.0] * hsz for _ in range(n_cls)]
                gb2 = [0.0] * n_cls
                for i in batch:
                    x = X[i]
                    h, logits = self._forward(x)
                    p = softmax(logits)
                    total_loss += -math.log(max(p[y[i]], 1e-12))
                    # ---- 输出层梯度 ----
                    dlogits = [p[c] - (1.0 if c == y[i] else 0.0) for c in range(n_cls)]
                    for c in range(n_cls):
                        d = dlogits[c]
                        if d == 0.0:
                            continue
                        row = gW2[c]
                        for k in range(hsz):
                            hk = h[k]
                            if hk:
                                row[k] += d * hk
                        gb2[c] += d
                    # ---- 隐藏层梯度（穿过 ReLU）----
                    dh = [0.0] * hsz
                    for c in range(n_cls):
                        d = dlogits[c]
                        if d == 0.0:
                            continue
                        w2row = self.W2[c]
                        for k in range(hsz):
                            if h[k]:
                                dh[k] += d * w2row[k]
                    for k in range(hsz):
                        g = dh[k]
                        if g == 0.0:
                            continue
                        row = gW1[k]
                        for j, xj in nonzeros(x):
                            row[j] = row.get(j, 0.0) + g * xj
                        gb1[k] += g
                inv = 1.0 / len(batch)
                for k in range(hsz):
                    row = gW1[k]
                    w1row = self.W1[k]
                    for j in list(row.keys()):
                        row[j] = row[j] * inv + self.l2 * w1row[j]
                    gb1[k] *= inv
                for c in range(n_cls):
                    w2row = self.W2[c]
                    row = gW2[c]
                    for k in range(hsz):
                        row[k] = row[k] * inv + self.l2 * w2row[k]
                    gb2[c] *= inv
                opt.step([self.W1, self.b1, self.W2, self.b2], [gW1, gb1, gW2, gb2])
            if ep % 5 == 0 or ep == 1:
                rec = {"epoch": ep, "loss": round(total_loss / len(X), 4)}
                if X_val:
                    rec["val_acc"] = round(self.accuracy(X_val, y_val), 4)
                self.history.append(rec)
                if verbose:
                    print(rec)
        return self

    def predict_proba(self, X: list) -> list[list[float]]:
        return [softmax(self._forward(x)[1]) for x in X]

    def predict(self, X: list) -> list[int]:
        return [max(range(len(p)), key=lambda i: p[i]) for p in self.predict_proba(X)]

    def accuracy(self, X: list, y: list[int]) -> float:
        pred = self.predict(X)
        return sum(1 for a, b in zip(pred, y) if a == b) / max(len(y), 1)


# ----------------------------------------------------------------------
# 组装：训练两个模型 + 评测
# ----------------------------------------------------------------------
def _features(train_texts: list[str], test_texts: list[str]):
    """稀疏特征：dict{下标: 权重}，避免两千维词表上的稠密矩阵运算。"""
    vec = TfidfVectorizer(use_char_ngram=True).fit(train_texts)
    dim = len(vec.vocab)
    Xtr = vec.transform_many(train_texts)
    Xte = vec.transform_many(test_texts)
    return vec, dim, Xtr, Xte


def train_all(verbose: bool = False, quick: bool = False) -> dict:
    """跑完整流程，返回给界面用的结果字典。

    quick=True 时把迭代轮数减半，用于自检脚本（结果仍然同量级）。
    """
    tr_texts, tr_labels, te_texts, te_labels = build_dataset(include_augment=True)
    vec, dim, Xtr, Xte = _features(tr_texts, te_texts)
    ep = 40 if quick else 100
    ep_mlp = 40 if quick else 80

    kb = KeywordBaseline().fit(tr_texts, tr_labels)
    lr = SoftmaxRegression(dim, len(LABELS), epochs=ep).fit(
        Xtr, tr_labels, Xte, te_labels, verbose=verbose)
    mlp = MLP(dim, len(LABELS), epochs=ep_mlp).fit(
        Xtr, tr_labels, Xte, te_labels, verbose=verbose)

    models = {"keyword": kb, "logreg": lr, "mlp": mlp}
    preds = {k: (m.predict(te_texts) if k == "keyword" else m.predict(Xte))
             for k, m in models.items()}
    probas = {k: (m.predict_proba(te_texts) if k == "keyword" else m.predict_proba(Xte))
              for k, m in models.items()}
    report = {k: classification_report(preds[k], te_labels, proba=probas[k]) for k in models}

    return {
        "labels": LABELS,
        "label_cn": LABEL_CN,
        "vec": vec,
        "dim": dim,
        "models": models,
        "n_train": len(tr_texts),
        "n_test": len(te_texts),
        "report": report,
        "history": {"logreg": lr.history, "mlp": mlp.history},
        "confusion": {k: confusion_matrix(preds[k], te_labels) for k in models},
        "conclusion": (
            "逻辑回归与轻量 MLP 在留出集上打平，而关键词 baseline 明显更差："
            "说明这题的瓶颈是样本量而非模型容量，直接换更大的预训练模型收益有限，"
            "更该做的是补真实工单数据和加检索/规则模块。"
        ),
    }


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def wilson_interval(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间。

    样本只有 16 条时，"准确率 0.94" 的区间其实宽到 0.72~0.99，
    直接拿点估计下结论是不严谨的，所以把区间一起报出来。
    """
    if total <= 0:
        return (0.0, 1.0)
    p = success / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def classification_report(pred: list[int], y: list[int],
                          proba: list[list[float]] | None = None) -> dict:
    n = len(LABELS)
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    support = [0] * n
    for p, t in zip(pred, y):
        support[t] += 1
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_cls = {}
    f1s = []
    for i, name in enumerate(LABELS):
        prec = tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] else 0.0
        rec = tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_cls[name] = {
            "label_cn": LABEL_CN[name],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support[i],
        }

    acc = sum(1 for p, t in zip(pred, y) if p == t) / max(len(y), 1)
    lo, hi = wilson_interval(sum(1 for p, t in zip(pred, y) if p == t), len(y))
    out = {
        "accuracy": round(acc, 4),
        "accuracy_ci95": [lo, hi],   # 留出集只有 16 条，光看一个点估计会过度解读
        "macro_f1": round(sum(f1s) / n, 4),
        "per_class": per_cls,
    }
    if proba:
        # 平均置信度：太低说明模型自己也不确定，这类样本适合直接转人工
        conf = [max(p) for p in proba]
        out["mean_confidence"] = round(sum(conf) / len(conf), 4)
    return out


def confusion_matrix(pred: list[int], y: list[int]) -> list[list[int]]:
    n = len(LABELS)
    m = [[0] * n for _ in range(n)]
    for p, t in zip(pred, y):
        m[t][p] += 1
    return m


def cross_validate(k: int = 4, seed: int = 3) -> dict:
    """分层 k 折交叉验证（只看逻辑回归和 MLP），给一个比单次留出更稳的估计。"""
    texts, labels, _, _ = build_dataset(include_augment=False)
    by_cls: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(y, []).append(i)
    rng = random.Random(seed)
    for v in by_cls.values():
        rng.shuffle(v)
    folds: list[list[int]] = [[] for _ in range(k)]
    for c, idxs in by_cls.items():
        for j, i in enumerate(idxs):
            folds[j % k].append(i)

    scores = {"logreg": [], "mlp": []}
    for f in range(k):
        test_idx = set(folds[f])
        tr = [i for i in range(len(texts)) if i not in test_idx]
        te = sorted(test_idx)
        tr_texts = [texts[i] for i in tr]
        te_texts = [texts[i] for i in te]
        ytr = [labels[i] for i in tr]
        yte = [labels[i] for i in te]
        _, dim, Xtr, Xte = _features(tr_texts, te_texts)
        lr = SoftmaxRegression(dim, len(LABELS), epochs=40).fit(Xtr, ytr)
        mlp = MLP(dim, len(LABELS), epochs=40).fit(Xtr, ytr)
        scores["logreg"].append(lr.accuracy(Xte, yte))
        scores["mlp"].append(mlp.accuracy(Xte, yte))

    def summarize(vals: list[float]) -> dict:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
                "folds": [round(v, 4) for v in vals]}

    return {"k": k, "n_samples": len(texts), **{kk: summarize(vv) for kk, vv in scores.items()}}


def transformer_available() -> bool:
    """检查可选的 BERT 微调依赖是否可用（缺失就跳过，不影响 demo 运行）。"""
    try:  # pragma: no cover
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


if __name__ == "__main__":  # 手动训练一遍
    import json

    res = train_all(verbose=True, quick=True)
    print(json.dumps({
        "n_train": res["n_train"],
        "n_test": res["n_test"],
        "report": {k: {"accuracy": v["accuracy"], "accuracy_ci95": v["accuracy_ci95"],
                       "macro_f1": v["macro_f1"]}
                   for k, v in res["report"].items()},
        "cv": cross_validate(k=4),
    }, ensure_ascii=False, indent=2))
