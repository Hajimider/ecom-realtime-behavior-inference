DROP TABLE IF EXISTS stream_metrics;
CREATE TABLE stream_metrics (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    metric_key  VARCHAR(64)   NOT NULL,
    metric_value VARCHAR(128) NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_metric (metric_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS stream_events_log;
CREATE TABLE stream_events_log (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id      VARCHAR(100),
    event_type      VARCHAR(20),
    buy_prob_xgb    DECIMAL(6,5),
    buy_prob_mlp    DECIMAL(6,5),
    anomaly_score   DECIMAL(6,5),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
