"""
Centralized configuration for the e-commerce churn prediction project.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JARS_DIR = PROJECT_ROOT / "jars"
OUTPUT_DIR = PROJECT_ROOT / "output"

RAW_DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = str(RAW_DATA_DIR / "2019-Oct.csv")

MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "user_analysis")
MYSQL_URL = f"jdbc:mysql://{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=Asia/Shanghai"
MYSQL_JDBC_DRIVER = "com.mysql.cj.jdbc.Driver"

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", str(PROJECT_ROOT / "spark_temp"))


def spark_jars() -> str:
    jars = list(JARS_DIR.glob("mysql-connector-j-*.jar"))
    if not jars:
        raise FileNotFoundError(f"No MySQL JDBC jar found under {JARS_DIR}")
    return str(jars[0])


OBSERVATION_END = "2019-10-22"


@dataclass
class MySQLConfig:
    user: str = MYSQL_USER
    password: str = MYSQL_PASSWORD
    host: str = MYSQL_HOST
    port: int = MYSQL_PORT
    database: str = MYSQL_DATABASE

    @property
    def sqlalchemy_url(self) -> str:
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def pymysql_kwargs(self) -> dict:
        return {"host": self.host, "user": self.user, "password": self.password, "database": self.database}

    @property
    def jdbc_url(self) -> str:
        return MYSQL_URL


@dataclass
class SparkConfig:
    app_name: str = "ChurnFeatureEngineering"
    master: str = SPARK_MASTER
    shuffle_partitions: int = SPARK_SHUFFLE_PARTITIONS
    local_dir: str = field(default_factory=lambda: SPARK_LOCAL_DIR)
    jdbc_driver: str = MYSQL_JDBC_DRIVER


RANDOM_SEED = 42
TEST_SIZE = 0.2
XGB_PARAMS = {
    "n_estimators": 150, "max_depth": 5, "learning_rate": 0.08,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.5,
    "random_state": RANDOM_SEED, "use_label_encoder": False, "eval_metric": "logloss",
}

FEATURE_COLUMNS = [
    "total_events", "view_count", "cart_count", "purchase_count",
    "view_to_cart_rate", "view_to_purchase_rate", "cart_to_purchase_rate",
    "unique_products", "unique_categories", "unique_brands",
    "avg_price", "total_spend",
    "n_sessions", "avg_events_per_session",
    "recency_days", "active_days",
    "weekend_ratio", "night_ratio",
    "top1_category",
]

mysql = MySQLConfig()
spark = SparkConfig()
