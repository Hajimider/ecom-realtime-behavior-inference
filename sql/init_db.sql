-- ============================================================================
-- Database Schema — E-Commerce User Churn Prediction & Profiling System
--  Dataset: Kaggle "E-Commerce Behavior Data from Multi-Category Store"
-- ============================================================================
-- Execution: mysql -u root -p < sql/init_db.sql
-- ============================================================================

CREATE DATABASE IF NOT EXISTS user_analysis
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE user_analysis;

-- --------------------------------------------------------------------------
-- 1. User features — Spark 特征计算结果（Flask Web 读，不用于训练）
-- --------------------------------------------------------------------------
DROP TABLE IF EXISTS user_features;
CREATE TABLE user_features (
    user_id              BIGINT        PRIMARY KEY,
    total_events         INT           COMMENT "All events in observation window",
    view_count           INT,
    cart_count           INT,
    purchase_count       INT,
    view_to_cart_rate    DECIMAL(6,5)  COMMENT "cart / view",
    view_to_purchase_rate DECIMAL(6,5) COMMENT "purchase / view",
    cart_to_purchase_rate DECIMAL(6,5) COMMENT "purchase / cart",
    unique_products      INT           COMMENT "Distinct products interacted",
    unique_categories    INT           COMMENT "Distinct categories",
    unique_brands        INT           COMMENT "Distinct brands",
    avg_price            DECIMAL(10,2) COMMENT "Average product price seen",
    total_spend          DECIMAL(12,2) COMMENT "Sum of purchase prices",
    n_sessions           INT           COMMENT "Distinct sessions",
    avg_events_per_session DECIMAL(8,2),
    recency_days         INT           COMMENT "Days since last event",
    active_days          INT           COMMENT "Unique active days",
    weekend_ratio        DECIMAL(6,5)  COMMENT "Weekend events / total",
    night_ratio          DECIMAL(6,5)  COMMENT "Night events (22-6h) / total",
    top1_category        VARCHAR(200)  COMMENT "Most-frequent category_code",
    r_score              INT           COMMENT "Recency quintile (1-5)",
    f_score              INT           COMMENT "Frequency quintile",
    m_score              INT           COMMENT "Monetary quintile",
    cluster              INT           COMMENT "KMeans segment label",
    is_churn             TINYINT       COMMENT "1 = no purchase in label window",
    feature_dt           DATE          COMMENT "Feature computation date"
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------------------------
-- 3. Churn predictions — inference results
-- --------------------------------------------------------------------------
DROP TABLE IF EXISTS churn_predictions;
CREATE TABLE churn_predictions (
    id                INT           AUTO_INCREMENT PRIMARY KEY,
    user_id           BIGINT,
    churn_probability DECIMAL(6,5)  COMMENT "XGBoost probability",
    prediction_label  VARCHAR(10)   COMMENT "Yes / No",
    predict_dt        DATE          COMMENT "Inference date",
    top_factor1       VARCHAR(50),
    top_factor2       VARCHAR(50),
    top_factor3       VARCHAR(50)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
