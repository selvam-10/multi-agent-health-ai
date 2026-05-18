"""
Configuration parameters for the multi-agent framework.
Adjust these for different clinical scenarios or datasets.
"""

# Monitoring Agent Thresholds
MONITORING_CONFIG = {
    "heart_rate": {
        "critical_high": 140,  # bpm
        "alert_high": 110,
        "alert_low": 50,
        "critical_low": 40,
    },
    "systolic_bp": {
        "critical_high": 180,  # mmHg
        "alert_high": 160,
        "alert_low": 90,
        "critical_low": 70,
    },
    "cholesterol": {
        "alert_threshold": 300,  # mg/dL
    },
    "temperature": {
        "fever_threshold": 38.0,  # Celsius
    }
}

# Anomaly Detection Config
ANOMALY_CONFIG = {
    "heart_rate_deviation": 30,  # bpm
    "systolic_bp_deviation": 25,  # mmHg
    "cholesterol_deviation": 50,  # mg/dL
}

# Risk Prediction Scoring
RISK_SCORING = {
    "high_heart_rate": 20,  # +20 points
    "high_blood_pressure": 25,
    "high_cholesterol": 20,
    "advanced_age": 15,  # age > 55
}

RISK_LEVELS = {
    "LOW": (0, 30),
    "MEDIUM": (30, 60),
    "HIGH": (60, 100),
}

# Generative Model Config
GENERATIVE_CONFIG = {
    "model_name": "gpt2",
    "max_length": 100,
    "temperature": 0.7,
    "do_sample": True,
    "device": -1,  # -1 for CPU, 0+ for GPU
}

# Data Loading
DATA_CONFIG = {
    "default_path": "dataset.csv",
    "timestamp_format": "%Y-%m-%d %H:%M:%S",
}
