"""
train_stream_models_spark.py — Spark 读取全部 42M 行，Session 特征 + 3 个模型训练
"""
from __future__ import annotations

import os, sys, warnings, time, gc, subprocess
from pathlib import Path

warnings.filterwarnings("ignore")
import joblib, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
import xgboost as xgb

os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g pyspark-shell"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import findspark; findspark.init()
except ImportError:
    pass

from pyspark.sql import SparkSession, functions as F, types as T
from config.settings import RANDOM_SEED, OUTPUT_DIR, RAW_DATA_FILE
from config.settings import spark as scfg


def _check_nvidia_smi() -> bool:
    try:
        o = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return o.returncode == 0
    except Exception:
        return False


def _detect_and_guide():
    print("─" * 50)
    print("🔍 检测硬件加速器 ...")
    found = []

    try:
        import torch
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            found.append(f"NVIDIA GPU ({name} × {n_gpus}) — CUDA {torch.version.cuda}")
    except ImportError:
        if _check_nvidia_smi():
            found.append("NVIDIA GPU（nvidia-smi 检测到，未装 torch，无法获取详情）")
        else:
            try:
                import pynvml
                pynvml.nvmlInit()
                n_gpus = pynvml.nvmlDeviceGetCount()
                if n_gpus > 0:
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    name = pynvml.nvmlDeviceGetName(h)
                    found.append(f"NVIDIA GPU ({name} × {n_gpus}) — pynvml")
                pynvml.nvmlShutdown()
            except Exception:
                pass

    try:
        import torch
        if torch.backends.mps.is_available():
            found.append("Apple Silicon GPU — MPS")
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name(0):
            found.append(f"AMD GPU — ROCm ({torch.cuda.get_device_name(0)})")
    except ImportError:
        pass

    if found:
        for line in found:
            print(f"  ✅ {line}")
        print(f"\n🚀 检测到加速器，可按以下路线加速训练：")
        print(f"  • XGBoost:  添加参数 tree_method='gpu_hist', device='cuda'")
        print(f"  • MLP:      将 sklearn MLP 替换为 torch nn.Module + .to('cuda')")
        print(f"  • IsolationForest: sklearn 无 GPU 实现，保持 CPU 即可")
    else:
        print("  ⚠️  未检测到 GPU / 加速器 → CPU 正常训练")
    print("─" * 50)


def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _detect_and_guide()

    schema = T.StructType([
        T.StructField("event_time", T.TimestampType(), True),
        T.StructField("event_type", T.StringType(), True),
        T.StructField("product_id", T.DoubleType(), True),
        T.StructField("price", T.DoubleType(), True),
        T.StructField("category_code", T.StringType(), True),
        T.StructField("brand", T.StringType(), True),
        T.StructField("user_session", T.StringType(), True),
    ])

    ss = SparkSession.builder \
        .appName("StreamFeatureSpark") \
        .master(scfg.master) \
        .config("spark.sql.shuffle.partitions", scfg.shuffle_partitions) \
        .config("spark.local.dir", scfg.local_dir) \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")

    print("[1/3] Spark 读取全部 CSV ...", flush=True)
    t0 = time.time()
    raw = ss.read.option("header", True).schema(schema).csv(RAW_DATA_FILE) \
        .withColumn("price", F.coalesce("price", F.lit(0.0))) \
        .drop("event_time").cache()
    total = raw.count()
    print(f"  {total:,} 行 | 耗时 {time.time()-t0:.0f}s", flush=True)

    print("[2/3] Session 特征聚合 ...", flush=True)
    t0 = time.time()
    non_buy = raw.filter(F.col("event_type") != "purchase").cache()
    buy_label = raw.filter(F.col("event_type") == "purchase") \
        .groupBy("user_session").agg(F.lit(1).alias("label"))

    feat = non_buy.groupBy("user_session").agg(
        F.count("*").alias("n"),
        F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("v"),
        F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("c"),
        F.countDistinct("product_id").alias("np"),
        F.countDistinct(F.when(F.col("category_code") != "", F.col("category_code"))).alias("nc"),
        F.countDistinct(F.when(F.col("brand") != "", F.col("brand"))).alias("nb"),
        F.max(F.when(F.col("price") > 0, F.col("price"))).alias("pmax"),
        F.min(F.when(F.col("price") > 0, F.col("price"))).alias("pmin"),
    ).cache()

    merged = feat.join(buy_label, "user_session", "left").fillna({"label": 0})
    n_sessions = merged.count()

    pdf = merged.select("n", "v", "c", "np", "nc", "nb", "pmax", "pmin", "label").toPandas()
    pdf["cr"] = (pdf["c"] / pdf["n"]).fillna(0)

    X = pdf[["n", "v", "c", "cr", "np", "nc", "nb", "pmax", "pmin"]].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    y = pdf["label"].values.astype(np.int32)

    del pdf, feat, non_buy, buy_label, merged, raw; gc.collect()
    ss.stop()
    print(f"  {n_sessions:,} Session | X.shape={X.shape} | 购买率 {y.mean()*100:.1f}% | 耗时 {time.time()-t0:.0f}s", flush=True)

    print("[3/3] 训练模型 ...", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    rng = np.random.RandomState(RANDOM_SEED)

    t0 = time.time()
    xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                              random_state=RANDOM_SEED, eval_metric="logloss", use_label_encoder=False)
    xgb_m.fit(X_tr, y_tr)
    print(f"  XGBoost AUC: {roc_auc_score(y_te, xgb_m.predict_proba(X_te)[:,1]):.4f} | {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    pi = np.where(y_tr == 1)[0]; ni = np.where(y_tr == 0)[0]
    ns = rng.choice(ni, len(pi), replace=False)
    ti = np.concatenate([pi, ns]); rng.shuffle(ti)
    mlp = MLPClassifier((128, 64, 32), activation="relu", max_iter=500,
                        random_state=RANDOM_SEED, early_stopping=True, validation_fraction=0.15)
    mlp.fit(X_tr[ti], y_tr[ti])
    tp = np.where(y_te == 1)[0]; tn = np.where(y_te == 0)[0]
    ts = rng.choice(tn, len(tp), replace=False)
    tbi = np.concatenate([tp, ts]); rng.shuffle(tbi)
    print(f"  MLP AUC (平衡): {roc_auc_score(y_te[tbi], mlp.predict_proba(X_te[tbi])[:,1]):.4f} | {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    ifo = IsolationForest(n_estimators=100, contamination=0.05, random_state=RANDOM_SEED).fit(X)
    print(f"  IForest 异常: {(ifo.predict(X)==-1).mean()*100:.1f}% | {time.time()-t0:.1f}s", flush=True)

    for n, m in [("stream_xgb", xgb_m), ("stream_mlp", mlp), ("stream_iforest", ifo)]:
        joblib.dump(m, str(OUTPUT_DIR / f"{n}.pkl"))
    print(f"  ✓ 模型已保存 | 总耗时 {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
