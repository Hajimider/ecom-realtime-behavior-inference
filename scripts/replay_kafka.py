"""
replay_kafka.py — 读取 2019-Oct.csv，按真实时间戳加速回放到 Kafka
"""
from __future__ import annotations

import csv, json, sys, time, warnings
from datetime import datetime
from pathlib import Path

from kafka import KafkaProducer
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = str(PROJECT_ROOT / "data" / "2019-Oct.csv")
SPEEDUP = 300
TOPIC = "user_actions"

SEND_LIMIT = 50000


def main():
    try:
        import pymysql as _pm
        _c = _pm.connect(host="localhost", user="root", password="123456", database="user_analysis")
        _c.cursor().execute("TRUNCATE stream_events_log")
        _c.commit(); _c.close()
    except Exception as e:
        print(f"清空数据失败（忽略）: {e}")

    print(f"读取 CSV: {CSV_PATH}")
    print(f"加速倍数: {SPEEDUP}x")
    row_count = 42448764
    print(f"    CSV: 5.28GB，总数据量: {row_count:,} 行")
    print(f"    当前 SEND_LIMIT = {SEND_LIMIT if SEND_LIMIT else 'None(全部)'}")
    limit = SEND_LIMIT if SEND_LIMIT is not None else row_count

    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        linger_ms=100, batch_size=65536,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    t_start = time.time()
    pbar = tqdm(total=limit, desc="  发送 Kafka", unit="条", mininterval=0.5, ncols=80)
    last_time = None
    sent = 0

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if sent >= limit:
                break
            event_time = datetime.strptime(row["event_time"].replace(" UTC", ""), "%Y-%m-%d %H:%M:%S")
            if last_time is None:
                last_time = event_time
            else:
                elapsed = (event_time - last_time).total_seconds() / SPEEDUP
                if elapsed > 0:
                    time.sleep(min(elapsed, 0.05))
                last_time = event_time

            msg = {
                "event_time": row["event_time"][:19],
                "event_type": row["event_type"],
                "product_id": row["product_id"],
                "category_code": row.get("category_code", "") or "",
                "brand": row.get("brand", "") or "",
                "price": float(row["price"]) if row["price"] else 0,
                "user_id": row["user_id"],
                "user_session": row["user_session"],
            }
            producer.send(TOPIC, value=msg)
            sent += 1
            pbar.update(1)

    producer.flush()
    producer.close()
    pbar.close()
    print(f"  读取 {sent:,} 条, 发送 {sent:,} 条, 耗时 {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
