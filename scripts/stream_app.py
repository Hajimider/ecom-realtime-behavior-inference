"""
stream_app.py — Flask SSE 实时看板
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import pymysql
from flask import Flask, render_template, Response
import logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import mysql

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))


def get_db():
    return pymysql.connect(**mysql.pymysql_kwargs, charset="utf8mb4")


def load_metrics() -> dict:
    conn = get_db()
    cursor = conn.cursor()
    metrics = {}

    cursor.execute("SELECT COUNT(*) FROM stream_events_log")
    metrics["total_events"] = int(cursor.fetchone()[0])

    cursor.execute("SELECT COUNT(*) FROM stream_events_log WHERE event_type='purchase'")
    metrics["total_purchases"] = int(cursor.fetchone()[0])

    cursor.execute("SELECT COUNT(DISTINCT session_id) FROM stream_events_log")
    metrics["total_sessions"] = int(cursor.fetchone()[0])

    cursor.execute("""
        SELECT event_type, COUNT(*) FROM stream_events_log
        WHERE created_at > NOW() - INTERVAL 1 MINUTE
        GROUP BY event_type
    """)
    recent = {r[0]: int(r[1]) for r in cursor.fetchall()}
    metrics["recent_views"] = recent.get("view", 0)
    metrics["recent_carts"] = recent.get("cart", 0)
    metrics["recent_purchases"] = recent.get("purchase", 0)

    cursor.execute("SELECT AVG(buy_prob_xgb) FROM stream_events_log")
    val = cursor.fetchone()[0]
    metrics["avg_buy_prob"] = round(float(val), 4) if val is not None else 0

    cursor.execute("SELECT AVG(anomaly_score) FROM stream_events_log")
    val = cursor.fetchone()[0]
    metrics["avg_anomaly"] = round(float(val), 4) if val is not None else 0

    cursor.execute("SELECT COUNT(*) FROM stream_events_log WHERE buy_prob_xgb > 0.5")
    high_prob = int(cursor.fetchone()[0])
    metrics["high_prob_pct"] = round(high_prob / metrics["total_sessions"] * 100, 1) if metrics["total_sessions"] > 0 else 0

    cursor.execute("""
        SELECT session_id, event_type, buy_prob_xgb, buy_prob_mlp, anomaly_score, created_at
        FROM stream_events_log ORDER BY id DESC LIMIT 5
    """)
    metrics["recent_rows"] = [
        {
            "session_id": r[0][:12] + "...",
            "event_type": r[1],
            "xgb": f"{r[2]*100:.1f}%" if r[2] else "N/A",
            "mlp": f"{r[3]*100:.1f}%" if r[3] else "N/A",
            "anomaly": f"{r[4]:.2f}",
            "time": r[5].strftime("%H:%M:%S") if r[5] else "",
        }
        for r in cursor.fetchall()
    ]

    cursor.close()
    conn.close()
    return metrics


@app.route("/")
def index():
    return render_template("stream_dashboard.html")


@app.route("/stream")
def stream():
    def generate():
        while True:
            m = load_metrics()
            yield f"data: {json.dumps(m, ensure_ascii=False)}\n\n"
            time.sleep(3)
    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    print(f"\n  -> http://localhost:5001\n")
    app.run(debug=False, host="0.0.0.0", port=5001)
