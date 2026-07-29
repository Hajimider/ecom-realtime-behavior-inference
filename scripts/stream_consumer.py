"""
stream_consumer.py — 消费 Kafka user_actions，实时推理 3 个模型，写 MySQL
"""
from __future__ import annotations

import json, sys, time, warnings
from collections import Counter
from pathlib import Path

import joblib, numpy as np, pymysql
from kafka import KafkaConsumer
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import OUTPUT_DIR

MODEL_DIR = OUTPUT_DIR
TOPIC = "user_actions"
BATCH_SECONDS = 5
KAFKA_TIMEOUT_MS = 60000
CONSUME_LIMIT = 50000


def _extract_features(events: list[dict]) -> list[float]:
    types = [e.get("event_type", "") for e in events]
    prices = [e.get("price", 0) for e in events if e.get("price", 0) > 0]
    products = set(e.get("product_id") for e in events if e.get("product_id"))
    cats = set(e.get("category_code", "") for e in events if e.get("category_code"))
    brands = set(e.get("brand", "") for e in events if e.get("brand"))
    cnt = Counter(types)
    n_total = len(events)
    return [
        n_total, cnt.get("view", 0), cnt.get("cart", 0),
        cnt.get("cart", 0) / n_total if n_total else 0,
        len(products), len(cats), len(brands),
        max(prices) if prices else 0, min(prices) if prices else 0,
    ]


def _infer_batch(buffer, xgb_model, mlp_model, iforest, cursor, conn):
    if not buffer:
        return
    sids = list(buffer.keys())
    events_list = [buffer[sid] for sid in sids]
    X_batch = np.array([_extract_features(evts) for evts in events_list])

    probs_xgb = xgb_model.predict_proba(X_batch)[:, 1]
    probs_mlp = mlp_model.predict_proba(X_batch)[:, 1]
    scores = iforest.score_samples(X_batch)

    rows = []
    for i, sid in enumerate(sids):
        prob_xgb = float(probs_xgb[i])
        prob_mlp = float(probs_mlp[i])
        s = float(scores[i])
        if not np.isfinite(s):
            s = 0.0
        rows.append((sid, events_list[i][-1]["event_type"],
                     round(prob_xgb, 5), round(prob_mlp, 5), round(s, 5)))
    try:
        cursor.executemany(
            "INSERT INTO stream_events_log "
            "(session_id, event_type, buy_prob_xgb, buy_prob_mlp, anomaly_score) "
            "VALUES (%s, %s, %s, %s, %s)", rows)
        conn.commit()
    except Exception as e:
        print(f"  MySQL 写入错误: {e}")


def main():
    print("加载 3 个模型 ...")
    t0 = time.time()
    xgb_model = joblib.load(str(MODEL_DIR / "stream_xgb.pkl"))
    mlp_model = joblib.load(str(MODEL_DIR / "stream_mlp.pkl"))
    iforest = joblib.load(str(MODEL_DIR / "stream_iforest.pkl"))
    print(f"  模型加载完成, 耗时 {time.time()-t0:.1f}s")

    conn = pymysql.connect(host="localhost", user="root", password="123456",
                           database="user_analysis", charset="utf8mb4")
    cursor = conn.cursor()

    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers="localhost:9092",
        auto_offset_reset="latest", enable_auto_commit=False,
        group_id=f"stream-infer-{int(time.time())}",
        consumer_timeout_ms=KAFKA_TIMEOUT_MS,
    )

    max_consume = CONSUME_LIMIT
    buffer: dict[str, list] = {}
    last_t = time.time()
    total = 0
    batch_t = time.time()

    bar = tqdm(total=max_consume, desc="消费 Kafka", unit="条", mininterval=0.5, smoothing=0.1)

    try:
        for msg in consumer:
            try:
                event = json.loads(msg.value.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not event:
                continue
            sid = event.get("user_session", "")
            if not sid:
                continue
            buffer.setdefault(sid, []).append(event)
            total += 1
            bar.update(1)
            bar.set_postfix_str(f"sessions={len(buffer)}")

            if max_consume and total >= max_consume:
                if buffer:
                    _infer_batch(buffer, xgb_model, mlp_model, iforest, cursor, conn)
                break

            now = time.time()
            if now - last_t >= BATCH_SECONDS:
                _infer_batch(buffer, xgb_model, mlp_model, iforest, cursor, conn)
                infer_elapsed = time.time() - batch_t
                bar.set_postfix_str(f"sessions={len(buffer)},推理={infer_elapsed:.1f}s")
                buffer.clear()
                last_t = now
                batch_t = now

    except KeyboardInterrupt:
        if buffer:
            _infer_batch(buffer, xgb_model, mlp_model, iforest, cursor, conn)
        bar.set_postfix_str("手动停止")
    finally:
        if buffer:
            _infer_batch(buffer, xgb_model, mlp_model, iforest, cursor, conn)
        bar.close()
        consumer.close()
        conn.close()
        print(f"  完成, 共处理 {total:,} 事件, 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
