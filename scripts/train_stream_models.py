"""
train_stream_models.py — 读取 200 万行提取 Session 特征，训练 3 个模型
"""
from __future__ import annotations

import sys, warnings, csv
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
import joblib, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RANDOM_SEED, OUTPUT_DIR

CSV_PATH = PROJECT_ROOT / "data" / "2019-Oct.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_ROWS = 2_000_000

print("[1/4] 提取 Session 特征 ...", flush=True)
print(f"  读取 {MAX_ROWS:,} 行...", flush=True)

session_stats = {}
with open(str(CSV_PATH), "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    with tqdm(total=MAX_ROWS, desc="  读取并分组", unit="行", mininterval=0.3, ncols=80) as pbar:
        for i, r in enumerate(reader):
            if i >= MAX_ROWS:
                break
            pbar.update(1)
            sid = r.get("user_session", "")
            if not sid:
                continue
            et = r["event_type"]
            price = float(r["price"]) if r["price"] else 0
            if sid not in session_stats:
                if et == "purchase":
                    session_stats[sid] = {"n": 0, "view": 0, "cart": 0, "purchase": 1,
                                          "products": set(), "cats": set(), "brands": set(),
                                          "prices": [], "last_type": ""}
                else:
                    session_stats[sid] = {"n": 1, "view": 1 if et == "view" else 0,
                                          "cart": 1 if et == "cart" else 0, "purchase": 0,
                                          "products": {r["product_id"]},
                                          "cats": set() if not r.get("category_code") else {r["category_code"]},
                                          "brands": set() if not r.get("brand") else {r["brand"]},
                                          "prices": [price] if price > 0 else [],
                                          "last_type": et}
            else:
                s = session_stats[sid]; s["n"] += 1
                if et == "view": s["view"] += 1
                elif et == "cart": s["cart"] += 1
                elif et == "purchase":
                    s["purchase"] += 1
                    continue
                s["products"].add(r["product_id"])
                if r.get("category_code"): s["cats"].add(r["category_code"])
                if r.get("brand"): s["brands"].add(r["brand"])
                if price > 0: s["prices"].append(price)
                s["last_type"] = et

print(f"  {i:,} 条事件 → {len(session_stats):,} 个 Session", flush=True)

X_data, y_data = [], []
n_sessions = len(session_stats)
pbar = tqdm(total=n_sessions, desc="  特征矩阵", unit="session", ncols=80, delay=0.5)
for s in session_stats.values():
    pbar.update(1)
    prices = s["prices"]
    X_data.append([s["n"], s["view"], s["cart"],
                   s["cart"] / s["n"] if s["n"] else 0,
                   len(s["products"]), len(s["cats"]), len(s["brands"]),
                   max(prices) if prices else 0, min(prices) if prices else 0])
    y_data.append(1 if s["purchase"] > 0 else 0)
pbar.close()

del session_stats
X = np.array(X_data); y = np.array(y_data)
print(f"  特征: {X.shape}, 购买率: {y.mean()*100:.1f}%", flush=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
print(f"  训练/{X_train.shape[0]:,} 测试/{X_test.shape[0]:,}", flush=True)

print("[2/4] XGBoost ...", flush=True)
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=RANDOM_SEED, eval_metric="logloss", use_label_encoder=False)
xgb_model.fit(X_train, y_train)
print(f"  AUC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]):.4f}", flush=True)

print("[3/4] MLP 神经网络（平衡采样）...", flush=True)
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]
rng = np.random.RandomState(RANDOM_SEED)
neg_sampled = rng.choice(neg_idx, size=len(pos_idx), replace=False)
train_idx = np.concatenate([pos_idx, neg_sampled])
rng.shuffle(train_idx)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=500,
                    random_state=RANDOM_SEED, early_stopping=True, validation_fraction=0.15)
mlp.fit(X_train[train_idx], y_train[train_idx])
test_pos_idx = np.where(y_test == 1)[0]
test_neg_idx = np.where(y_test == 0)[0]
test_sampled = rng.choice(test_neg_idx, size=len(test_pos_idx), replace=False)
test_balanced_idx = np.concatenate([test_pos_idx, test_sampled])
rng.shuffle(test_balanced_idx)
auc_mlp = roc_auc_score(y_test[test_balanced_idx], mlp.predict_proba(X_test[test_balanced_idx])[:, 1])
print(f"  AUC (平衡测试集): {auc_mlp:.4f}", flush=True)

print("[4/4] IsolationForest ...", flush=True)
iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=RANDOM_SEED)
iforest.fit(X)
print(f"  异常占比: {(iforest.predict(X) == -1).mean() * 100:.1f}%", flush=True)

joblib.dump(xgb_model, str(OUTPUT_DIR / "stream_xgb.pkl"))
joblib.dump(mlp, str(OUTPUT_DIR / "stream_mlp.pkl"))
joblib.dump(iforest, str(OUTPUT_DIR / "stream_iforest.pkl"))
print(f"  ✓ 模型已保存到 {OUTPUT_DIR}/stream_*.pkl", flush=True)
