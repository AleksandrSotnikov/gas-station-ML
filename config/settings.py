"""Configuration settings for the gas station ML project."""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ML Model settings
ML_CONFIG = {
    "clustering": {
        "kmeans": {
            "n_clusters": 5,
            "random_state": 42,
            "max_iter": 300
        },
        "dbscan": {
            "eps": 0.5,
            "min_samples": 5
        },
        "hierarchical": {
            "n_clusters": 5,
            "linkage": "ward"
        }
    },
    "feature_engineering": {
        "scaler": "StandardScaler",
        "pca_components": 0.95
    },
    "validation": {
        "test_size": 0.2,
        "cv_folds": 5,
        "random_state": 42
    }
}

# Customer persona settings
PERSONA_CONFIG = {
    "min_cluster_size": 20,  # lowered from 50 to generate personas on smaller datasets
    "max_personas": 10,
    "behavioral_features": [
        "visit_frequency",
        "avg_purchase_amount",
        "time_of_day_preference",
        "fuel_type_preference",
        "additional_services_usage",
        "price_sensitivity",
        "loyalty_score"
    ],
    "contextual_triggers": [
        "weather_condition",
        "traffic_level",
        "queue_length",
        "distance_to_station",
        "day_of_week",
        "season"
    ]
}

# Experiment design settings
EXPERIMENT_CONFIG = {
    "ab_test": {
        "min_sample_size": 1000,
        "confidence_level": 0.95,
        "statistical_power": 0.8,
        "min_effect_size": 0.05
    },
    "metrics": [
        "conversion_rate",
        "average_purchase_value",
        "customer_satisfaction",
        "retention_rate",
        "cross_selling_rate"
    ]
}

# Database settings
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "gas_station_ml"),
    "username": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "pool_size": 10,
    "max_overflow": 20
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": os.getenv("DEBUG", "False").lower() == "true",
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "cors_origins": ["*"],
    "rate_limit": 100  # requests per minute
}

# Logging settings
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR / "app.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    API_CONFIG["debug"] = False
    LOGGING_CONFIG["handlers"]["default"]["level"] = "WARNING"
elif os.getenv("ENVIRONMENT") == "development":
    API_CONFIG["debug"] = True
    ML_CONFIG["validation"]["cv_folds"] = 3  # Faster for development
