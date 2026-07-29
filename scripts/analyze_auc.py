"""
analyze_auc.py — 验证 AUC 是否因 cart 特征导致，对比完整 vs 去掉 cart 后 AUC 差异
"""
from __future__ import annotations

import os, sys, warnings, gc
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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
from config.settings import RANDOM_SEED, RAW_DATA_FILE
from config.settings import spark as scfg


def main():
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
        .appName("AucAnalysis") \
        .master(scfg.master) \
        .config("spark.sql.shuffle.partitions", scfg.shuffle_partitions) \
        .config("spark.local.dir", scfg.local_dir) \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")

    print("读取全部 CSV ...", flush=True)
    raw = ss.read.option("header", True).schema(schema).csv(RAW_DATA_FILE) \
        .withColumn("price", F.coalesce("price", F.lit(0.0))) \
        .drop("event_time").cache()
    raw.count()

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
    pdf = merged.select("n", "v", "c", "np", "nc", "nb", "pmax", "pmin", "label").toPandas()
    pdf["cr"] = (pdf["c"] / pdf["n"]).fillna(0)

    del feat, non_buy, buy_label, merged, raw; gc.collect()
    ss.stop()
    print(f"数据加载完成: {len(pdf):,} sessions", flush=True)

    full_cols = ["n", "v", "c", "cr", "np", "nc", "nb", "pmax", "pmin"]
    X_full = np.nan_to_num(pdf[full_cols].values.astype(np.float64), nan=0.0)
    y = pdf["label"].values.astype(np.int32)

    no_cart_cols = ["n", "v", "np", "nc", "nb", "pmax", "pmin"]
    X_nocart = np.nan_to_num(pdf[no_cart_cols].values.astype(np.float64), nan=0.0)

    cart_only_cols = ["c", "cr"]
    X_cart = pdf[cart_only_cols].values.astype(np.float64)

    stats_cols = ["n", "v", "np", "nc", "nb"]
    X_stats = pdf[stats_cols].values.astype(np.float64)

    print("\n=== cart 特征与 label 的关系 ===", flush=True)
    has_cart = (pdf["c"] > 0)
    for cart_val, label_val, desc in [
        (has_cart, pdf["label"], "有加购 vs 购买"),
        (~has_cart, pdf["label"], "无加购 vs 购买"),
    ]:
        n = cart_val.sum()
        buy = label_val[cart_val].mean()
        print(f"  {desc}: {n:>6,} sessions, 购买率 {buy*100:.1f}%", flush=True)

    cart_and_buy = (has_cart & (pdf["label"] == 1)).sum()
    cart_no_buy = (has_cart & (pdf["label"] == 0)).sum()
    no_cart_buy = (~has_cart & (pdf["label"] == 1)).sum()
    no_cart_no_buy = (~has_cart & (pdf["label"] == 0)).sum()
    print(f"\n  混淆矩阵:", flush=True)
    print(f"                 label=1     label=0")
    print(f"   有加购      {cart_and_buy:>8,}  {cart_no_buy:>8,}")
    print(f"   无加购      {no_cart_buy:>8,}  {no_cart_no_buy:>8,}")

    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2,
                                            random_state=RANDOM_SEED, stratify=y)
    results = []
    for name, X in [
        ("完整 9 维特征", X_full),
        ("去掉 cart (c,cr)", X_nocart),
        ("只用 cart (c,cr)", X_cart),
        ("仅基础统计 (n,v,np,nc,nb)", X_stats),
    ]:
        print(f"\n{'='*50}\n训练: {name}\n  特征维度: {X.shape[1]}", flush=True)
        m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              scale_pos_weight=(y[train_idx]==0).sum()/max((y[train_idx]==1).sum(), 1),
                              random_state=RANDOM_SEED, eval_metric="logloss", use_label_encoder=False,
                              verbosity=0)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], pred)
        pred_bin = (pred >= 0.5).astype(int)
        prec, rec = precision_score(y[test_idx], pred_bin), recall_score(y[test_idx], pred_bin)
        print(f"  AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}", flush=True)
        results.append((name, auc, prec, rec))

    print(f"\n{'='*50}\n结果汇总:", flush=True)
    print(f"  {'特征组合':<30} {'AUC':<8} {'Precision':<12} {'Recall':<8}")
    print(f"  {'-'*58}")
    for name, auc, prec, rec in results:
        print(f"  {name:<30} {auc:<8.4f} {prec:<12.4f} {rec:<8.4f}", flush=True)


if __name__ == "__main__":
    main()
