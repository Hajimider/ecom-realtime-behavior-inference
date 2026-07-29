# 电商实时用户行为流分析与在线推理

用 [Kaggle E-Commerce Behavior Data from Multi-Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) 全部 42M 行数据，Spark 加速特征提取训练 3 个模型，Kafka 实时回放用户行为，在线推理每个会话的购买概率与异常分数。

---

## 数据

[E-Commerce Behavior Data from Multi-Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)

| | |
|---|---|
| 大小 | 5.3 GB，42,448,765 行 |
| 时间 | 2019-10-01 ~ 2019-10-31 |
| 字段 | event_time, event_type, product_id, category_code, brand, price, user_id, user_session |
| 事件类型 | view（93.8%）、cart（3.3%）、purchase（2.9%） |

---

## 数据处理流程

```
                         ┌── Spark 直读 ──► Session 特征 ──► XGBoost (AUC 0.958)
2019-Oct.csv ───┬──► 训练 ──┤                                 MLP    (AUC 0.597)
 (5.3 GB, 42M)  │         └── 全部 42M 行                     IForest (异常 5.0%)
                │         36s + 31s = 67s 特征提取
                │
                └──► replay_kafka.py ──► Kafka ──► stream_consumer.py ──► MySQL ──► stream_app.py
                     加速回放 (300x)      user_actions   3模型在线推理              SSE 实时看板
                                                                                  localhost:5001
```

### 1. 模型训练

`scripts/train_stream_models_spark.py`

Spark 直读全部 42,448,764 行，groupBy 聚合 Session 级 9 维特征（约 65,298 个 Session），3 个模型训练总耗时约 1.7 分钟。

| 维度 | 特征 | 说明 |
|------|------|------|
| 事件量级 | n_total / view / cart | 当前会话的事件计数 |
| 转化率 | cart_ratio | 加购事件占比 |
| 多样性 | n_products / n_categories / n_brands | 商品、品类、品牌覆盖 |
| 价格 | price_max / price_min | 已浏览商品的价格区间 |

训练 3 个模型：

| 模型 | 说明 | 指标 |
|------|------|------|
| XGBoost | 树模型，`scale_pos_weight` 处理不平衡 | AUC **0.958** |
| MLPClassifier | 神经网络（平衡采样 1:1 训练，平衡测试集评估） | AUC **0.597** |
| IsolationForest | 无监督异常检测，`contamination=0.05` | **5.0%** 异常 |

**数据泄露验证：已修复 ✅** purchase 事件通过 Spark filter 完全排除在特征计算之外，只用于打标签，从代码层面杜绝了数据泄露。

**AUC 0.958 分析：** cart 特征并非 AUC 高的主因。对比试验结果：

| 特征组合 | AUC |
|----------|-----|
| 完整 9 维特征 | **0.9581** |
| 去掉 cart (c, cr) | **0.9506** |
| 只用 cart (c, cr) | 0.8101 |
| 仅基础统计 (n, v, np, nc, nb) | 0.9503 |

去掉 cart 后 AUC 仅降 0.0075，说明核心区分力来自事件量级和多样性特征（n_total、view_count、product/category/brand 覆盖），而非加购信息。cart 特征虽有区分力（有加购 session 购买率 90.7%，无加购 18.6%），但整体贡献有限。

MLP 在平衡测试集上 AUC 偏低（0.597），说明全量数据下简单 3 层全连接网络表达能力不足。

运行前自动检测 GPU / 加速器，检测到时输出加速路线引导（XGBoost 启用 `gpu_hist`、MLP 改用 torch）。无 GPU 则 CPU 正常训练。

### 2. Kafka 回放

`scripts/replay_kafka.py`

- 按 CSV 中的真实时间戳以 300 倍速回放
- 通过 `SEND_LIMIT` 控制发送数量（默认 50000，设为 `None` 发送全部 42M 行）
- 启动前自动清空上一轮的 `stream_events_log`

### 3. 在线推理

`scripts/stream_consumer.py`

- 消费 Kafka `user_actions` topic
- 每 5 秒批量推理一次，将整个 batch 拼接成 (N, 9) 矩阵统一调用模型，避免逐条推理的性能开销
- 3 个模型输出：XGBoost 购买概率、MLP 购买概率、IsolationForest 异常得分
- 写入 MySQL `stream_events_log`

### 4. Flask SSE 实时看板

`scripts/stream_app.py` + `templates/stream_dashboard.html`

- `http://localhost:5001`
- 累计事件数 / Session 数 / 购买数
- 近 1 分钟浏览 / 加购 / 购买实时计数
- 平均购买概率与异常分
- 最新 5 条推理结果（每条含 XGBoost 概率、MLP 概率、异常分）

---

## 运行

### 前置条件

| 依赖 | 说明 |
|------|------|
| Python 3.10+ | 项目运行基础 |
| Spark 3.4+ | 训练阶段特征提取（配合 findspark） |
| Kafka | 启动在 `localhost:9092` |
| MySQL 8.0+ | 存储推理结果 |
| 2019-Oct.csv | 放入项目 `data/` 目录 |

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化 MySQL

```bash
mysql -u root -p < sql/stream_init.sql
```

### 3. 训练模型

```bash
python scripts/train_stream_models_spark.py
```

Spark 直读全部 42M 行，特征提取约 67 秒，训练约 31 秒，总耗时约 1.7 分钟。模型保存到 `output/stream_*.pkl`。

如需原版（Pandas 读取 200 万行）训练，使用 `python scripts/train_stream_models.py`。

### 4. 启动 Kafka 回放 + 消费 + 看板

需要 3 个终端同时运行：

```bash
# 终端 1：回放数据到 Kafka
python scripts/replay_kafka.py

# 终端 2：消费 Kafka 并推理
python scripts/stream_consumer.py

# 终端 3：打开实时看板
python scripts/stream_app.py
```

访问 `http://localhost:5001`。

> 默认每端只处理 50000 条（演示模式）。如需处理全部 42M 行，修改 `replay_kafka.py` 的 `SEND_LIMIT = None` 和 `stream_consumer.py` 的 `CONSUME_LIMIT = None`。

---

## 项目结构

```
data/
  2019-Oct.csv                        # 数据集（5.3 GB）
scripts/
  train_stream_models.py              # 原版：Pandas 读 200 万行
  train_stream_models_spark.py        # Spark 版：读全部 42M 行
  analyze_auc.py                      # AUC 分析（验证数据泄露 + 特征贡献）
  replay_kafka.py                     # CSV → Kafka 回放
  stream_consumer.py                  # Kafka 消费 + 在线推理
  stream_app.py                       # Flask SSE 实时看板
templates/
  stream_dashboard.html               # 暗色风格看板页面
sql/
  stream_init.sql                     # MySQL 建表
config/
  settings.py                         # 集中配置（Spark / MySQL / 路径）
output/
  stream_xgb.pkl                      # 训练好的模型文件
  stream_mlp.pkl
  stream_iforest.pkl
requirements.txt                      # 完整依赖
```

---

## 踩过的坑

**数据泄露。** 初始版本将整个 Session（包括 purchase 事件本身）都用于计算特征，XGBoost AUC 虚高到 0.9975。修复：purchase 事件只用于标签，不进入特征计算。经 `analyze_auc.py` 交叉验证确认修复。

**AUC 0.958 是否正常。** 经 `analyze_auc.py` 对比试验验证：去掉 cart 特征后 AUC 仅从 0.9581 降至 0.9506，说明核心区分力来自事件量级和多样性特征，不存在数据泄露。AUC 0.958 反映了 9 维特征在 6.5 万 Session 上的真实区分能力。

**MLP 在平衡测试集上 AUC 偏低（0.597）。** 全量数据类别分布复杂，简单 MLP（3 层全连接）表达能力不足以拟合全部模式，可换 torch + 深层次结构 + GPU 加速。

**进度条 0% 闪现。** `print()` 与 `tqdm` 同时输出时，tqdm 的 0% 行会出现在 print 文字之前。修复：print 语句加 `flush=True`，tqdm 设 `delay=0.5` 延迟首行显示。

**Kafka flush 导致消息丢失。** 每 5000 条手动 `flush()` 反而降低吞吐。改为 `linger_ms=100, batch_size=65536` 让 Kafka 异步批量发送，只用末尾一次 `flush()`。

**MySQL Decimal 无法 JSON 序列化。** `AVG()` 返回 `Decimal` 类型，`json.dumps` 报错。用 `float()` 包裹后再序列化。

**模型推理 batch 优化。** 逐个 session 调用 `predict_proba` 在 Python 层有巨大开销。改为拼接 (N, 9) 矩阵一次推理，batch size 1000+ 时提速约 1000 倍。

---

## 依赖

kafka-python, tqdm, pandas, numpy, scikit-learn, xgboost, flask, pymysql, pyspark, findspark

Kafka 依赖本地运行的 Kafka 服务（`localhost:9092`）。训练阶段需要 Spark 环境（配合 findspark 自动发现）。
