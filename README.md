# 电商用户流失预测

PySpark / XGBoost / SHAP / Hive / Flask

## 数据

[Kaggle E-Commerce Behavior Data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)（2019-10，42,448,765行，5.3GB）。

字段: event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session

事件类型: view(93.8%), cart(3.3%), purchase(2.9%)

---

# 客户行为与消费习惯分析

Pandas / RFM / Matplotlib

## 数据

5万条客户数据，25个字段（年龄、性别、国家、会员年限、登录频率、会话时长、页面浏览、购物车放弃率、心愿单商品数、购买次数、LTV等）。

---

# 实时用户行为流分析

Kafka / XGBoost / MLP / IsolationForest / Flask SSE

## 数据

与离线项目同源（Kaggle电商行为数据），通过 replay_kafka.py 加速回放入 Kafka，stream_consumer.py 消费并推理。

## 使用

```bash
pip install kafka-python tqdm
python scripts/train_stream_models.py
python scripts/replay_kafka.py
python scripts/stream_consumer.py
python scripts/stream_app.py
```

详细说明见对应脚本顶部注释。
