"""HTTP 接口自检：起一个本地服务（或用已有服务），逐个打接口并校验。

用法：
    python api_test.py            # 自动起临时服务（端口 8199），测完关闭
    python api_test.py 8124       # 测已经在跑的指定端口
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

PASS, FAIL = [], []


def get(path: str, timeout: int = 240):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def post(path: str, payload: dict, timeout: int = 240):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def check(name: str, fn) -> None:
    t0 = time.time()
    try:
        detail = fn()
        PASS.append(name)
        print(f"  [PASS] {name}  ({time.time() - t0:.1f}s)  {detail}")
    except Exception as exc:  # noqa: BLE001
        FAIL.append((name, exc))
        print(f"  [FAIL] {name}  ({time.time() - t0:.1f}s)  {exc.__class__.__name__}: {exc}")


def wait_ready(proc=None, tries: int = 60) -> None:
    for _ in range(tries):
        try:
            get("/api/health", timeout=3)
            return
        except Exception:  # noqa: BLE001
            if proc is not None and proc.poll() is not None:
                out = proc.stdout.read().decode("utf-8", "ignore") if proc.stdout else ""
                raise RuntimeError(f"服务提前退出：\n{out}")
            time.sleep(0.5)
    raise RuntimeError("服务未在预期时间内就绪")


def main() -> int:
    proc = None
    if len(sys.argv) == 1:
        env = dict(os.environ, DEMO_PORT=str(PORT), PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen([sys.executable, "serving.py"], cwd=_HERE, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"已启动临时服务 端口 {PORT}，pid={proc.pid}")
    wait_ready(proc)

    print("== 接口自检 ==")

    check("GET / 静态页面", lambda: (lambda s, b: f"{len(b)} bytes")(
        *(lambda r: (r.status, r.read()))(urllib.request.urlopen(BASE + "/", timeout=30))))

    def health():
        s, j = get("/api/health", timeout=10)
        assert j["ok"] and j["tickets"] == 20 and j["intents"] == 8, j
        return f"ok={j['ok']} 工单={j['tickets']} 意图={j['intents']}"
    check("GET /api/health", health)

    def tickets():
        s, j = get("/api/retrieval/tickets", timeout=20)
        assert len(j["tickets"]) == 20
        return f"{len(j['tickets'])} 条工单"
    check("GET /api/retrieval/tickets", tickets)

    def search():
        q = urllib.parse.quote("充电宝能给手机充电，但是自己充不进去电")
        s, j = get(f"/api/retrieval/search?q={q}&k=3", timeout=30)
        assert len(j["results"]) == 3, len(j["results"])
        assert "contrib" in j["results"][0] and "clause" in j["results"][0]["features"]
        top = j["results"][0]
        return f"top1={top['id']} {top['title'][:16]} 分数={top['final_score']}"
    check("GET /api/retrieval/search", search)

    def search_bad():
        try:
            get("/api/retrieval/search?q=", timeout=10)
        except urllib.error.HTTPError as e:
            assert e.code == 400, e.code
            return "空查询返回 400"
        raise AssertionError("空查询应当返回 400")
    check("GET /api/retrieval/search 参数校验", search_bad)

    def eval_ep():
        s, j = get("/api/retrieval/eval", timeout=60)
        assert set(j) >= {"baseline", "rerank", "per_query", "flip", "limitation"}, list(j)
        assert len(j["per_query"]) == 20
        return (f"R@1 {j['baseline']['recall@1']}→{j['rerank']['recall@1']} · "
                f"MRR {j['baseline']['mrr']}→{j['rerank']['mrr']} · "
                f"Top1 被改写 {j['flip']['n']}/{len(j['per_query'])}")
    check("GET /api/retrieval/eval", eval_ep)

    def stats():
        s, j = get("/api/classify/stats", timeout=600)
        assert j["n_test"] == 16 and j["dim"] > 100
        assert all(k in j["report"] for k in ("keyword", "logreg", "mlp"))
        assert len(j["confusion"]["logreg"]) == 8
        return " · ".join(f"{k}={v['accuracy']}" for k, v in j["report"].items())
    check("GET /api/classify/stats（含首次训练）", stats)

    def classify():
        s, j = post("/api/classify", {"text": "耳塞丢了，哪能买到"}, timeout=60)
        assert j["models"]["logreg"]["label_cn"], j
        assert j["action"]["level"] in ("auto", "assist", "human")
        return (f"logreg={j['models']['logreg']['label_cn']}"
                f"({j['models']['logreg']['confidence']}) "
                f"mlp={j['models']['mlp']['label_cn']} 建议={j['action']['level']}")
    check("POST /api/classify", classify)

    def classify_bad():
        try:
            post("/api/classify", {"text": "  "}, timeout=10)
        except urllib.error.HTTPError as e:
            assert e.code == 400, e.code
            return "空文本返回 400"
        raise AssertionError("空文本应当返回 400")
    check("POST /api/classify 参数校验", classify_bad)

    def cv():
        s, j = post("/api/classify/cv", {}, timeout=600)
        assert j["k"] == 4 and "logreg" in j and "mlp" in j
        return f"logreg={j['logreg']['mean']}±{j['logreg']['std']} mlp={j['mlp']['mean']}±{j['mlp']['std']}"
    check("POST /api/classify/cv（4 折）", cv)

    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("临时服务已关闭")

    print()
    print(f"结果：{len(PASS)} 项通过，{len(FAIL)} 项失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
