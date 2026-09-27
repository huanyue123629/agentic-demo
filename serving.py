"""本地 HTTP 服务：给两个 demo 套一个网页界面。

只用标准库 http.server，不需要 Flask/FastAPI，也不用 npm。
启动后浏览器打开 http://127.0.0.1:8000 即可。

接口一览（都是 JSON）：
    GET  /api/health                        健康检查
    GET  /api/retrieval/tickets             工单语料
    GET  /api/retrieval/search?q=...&k=5     检索 + 重排（含特征明细）
    GET  /api/retrieval/eval                 Recall/MRR/nDCG 对比
    GET  /api/classify/stats                 模型指标（第一次调用会训练，约 15s）
    POST /api/classify                       单条文本的意图预测
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "01_recall_rerank"), os.path.join(_HERE, "02_intent_classifier")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.data_intent import LABEL_CN, LABELS  # noqa: E402
from common.data_retrieval import TICKETS  # noqa: E402
from pipeline import RetrievalPipeline  # noqa: E402

DEFAULT_PORT = 8000

# ---- 懒加载 + 训练缓存 -------------------------------------------------
_state: dict = {"pipe": None, "pipe_eval": None, "clf": None}
_lock = threading.Lock()


def get_pipeline() -> RetrievalPipeline:
    if _state["pipe"] is None:
        with _lock:
            if _state["pipe"] is None:
                _state["pipe"] = RetrievalPipeline()
    return _state["pipe"]


def get_pipeline_eval() -> dict:
    if _state["pipe_eval"] is None:
        with _lock:
            if _state["pipe_eval"] is None:
                _state["pipe_eval"] = get_pipeline().evaluate()
    return _state["pipe_eval"]


def get_classifier() -> dict:
    """训练一次就缓存住；模型训练是 CPU 密集的，别每次请求都重跑。"""
    if _state["clf"] is None:
        with _lock:
            if _state["clf"] is None:
                from classifier import train_all

                _state["clf"] = train_all()
    return _state["clf"]


def classify_text(text: str) -> dict:
    res = get_classifier()
    vec = res["vec"]
    x = vec.transform(text)
    out = {}
    for key in ("keyword", "logreg", "mlp"):
        model = res["models"][key]
        proba = model.predict_proba([text] if key == "keyword" else [x])[0]
        order = sorted(range(len(proba)), key=lambda i: -proba[i])
        top = order[0]
        out[key] = {
            "label": LABELS[top],
            "label_cn": LABEL_CN[LABELS[top]],
            "confidence": round(proba[top], 4),
            "top3": [
                {"label": LABELS[i], "label_cn": LABEL_CN[LABELS[i]], "p": round(proba[i], 4)}
                for i in order[:3]
            ],
            "margin": round(proba[order[0]] - proba[order[1]], 4),
        }
    # 业务动作建议：置信度低就建议转人工，这是客服系统里最常见的兜底设计
    conf = out["logreg"]["confidence"]
    margin = out["logreg"]["margin"]
    if conf >= 0.7 and margin >= 0.35:
        action = {"level": "auto", "text": "可直接自动答复 / 自动分派"}
    elif conf >= 0.45:
        action = {"level": "assist", "text": "建议坐席确认后发送（置信度中等）"}
    else:
        action = {"level": "human", "text": "置信度不足，建议转人工处理"}
    return {"text": text, "models": out, "action": action, "features": {"n_nonzero": len(x), "dim": res["dim"]}}


# ---- 静态页面 ---------------------------------------------------------
def _static_file(name: str) -> str:
    return os.path.join(_HERE, "web", name)


class Handler(BaseHTTPRequestHandler):
    server_version = "AnkerDemo/1.0"

    # ---------- 工具 ----------
    def _send(self, payload, status: int = 200, content_type: str = "application/json; charset=utf-8"):
        if not isinstance(payload, (bytes, bytearray)):
            payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: str, content_type: str):
        if not os.path.isfile(path):
            self._send({"error": f"not found: {os.path.basename(path)}"}, 404)
            return
        with open(path, "rb") as f:
            self._send(f.read(), 200, content_type)

    def log_message(self, fmt, *args):  # 精简日志
        sys.stderr.write("[server] %s\n" % (fmt % args))

    # ---------- 路由 ----------
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send_file(_static_file("index.html"), "text/html; charset=utf-8")
            elif path == "/api/health":
                self._send({"ok": True, "tickets": len(TICKETS), "intents": len(LABELS)})
            elif path == "/api/retrieval/tickets":
                self._send({"tickets": [
                    {k: t[k] for k in ("id", "title", "text", "cat", "group", "helpful")}
                    for t in TICKETS
                ]})
            elif path == "/api/retrieval/search":
                q = (qs.get("q") or [""])[0].strip()
                if not q:
                    self._send({"error": "缺少参数 q"}, 400)
                    return
                k = int((qs.get("k") or ["5"])[0])
                use_rerank = (qs.get("rerank") or ["1"])[0] not in ("0", "false", "no")
                self._send(get_pipeline().search(q, top_k=k, use_rerank=use_rerank))
            elif path == "/api/retrieval/eval":
                self._send(get_pipeline_eval())
            elif path == "/api/classify/stats":
                res = get_classifier()
                self._send({
                    "labels": LABELS,
                    "label_cn": LABEL_CN,
                    "dim": res["dim"],
                    "n_train": res["n_train"],
                    "n_test": res["n_test"],
                    "report": res["report"],
                    "confusion": res["confusion"],
                    "history": res["history"],
                    "conclusion": res["conclusion"],
                })
            else:
                self._send({"error": "unknown path", "path": path}, 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send({"error": f"{exc.__class__.__name__}: {exc}"}, 500)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            self._send({"error": "请求体不是合法 JSON"}, 400)
            return
        try:
            if parsed.path == "/api/classify":
                text = (body.get("text") or "").strip()
                if not text:
                    self._send({"error": "缺少 text 字段"}, 400)
                    return
                self._send(classify_text(text))
            elif parsed.path == "/api/classify/cv":
                from classifier import cross_validate

                self._send(cross_validate(k=4))
            elif parsed.path == "/api/classify/retrain":
                with _lock:
                    _state["clf"] = None
                self._send({"ok": True, "msg": "已清空缓存，下次请求会重新训练"})
            else:
                self._send({"error": "unknown path", "path": parsed.path}, 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send({"error": f"{exc.__class__.__name__}: {exc}"}, 500)


def main() -> None:
    port = int(os.environ.get("DEMO_PORT", DEFAULT_PORT))
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"demo 已启动：http://127.0.0.1:{port}")
    print("  - 召回+重排：/api/retrieval/search?q=充电宝充不进电")
    print("  - 意图分类：/api/classify/stats（首次调用会训练，约 15 秒）")
    print("按 Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
