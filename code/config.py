from pathlib import Path
import os


# ---------------------------------------------------------
# Project directories
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_FILE = PROJECT_ROOT / "output.csv"


# ---------------------------------------------------------
# Dataset files
# ---------------------------------------------------------

REQUESTS_FILE = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_FILE = DATASET_DIR / "sample_requests.csv"

FINANCIAL_PROFILES_FILE = DATASET_DIR / "financial_profiles.csv"
FINANCIAL_EVENTS_FILE = DATASET_DIR / "financial_events.csv"
PAYMENT_OPTIONS_FILE = DATASET_DIR / "request_payment_options.csv"

EXCHANGE_RATES_FILE = DATASET_DIR / "exchange_rates.csv"
MESSAGES_FILE = DATASET_DIR / "messages.csv"
IMAGES_FILE = DATASET_DIR / "images.csv"

MEDIA_DIR = DATASET_DIR / "media"
IMAGE_DIR = MEDIA_DIR / "images"


# ---------------------------------------------------------
# Forecast configuration
# ---------------------------------------------------------

FORECAST_DAYS = 90


# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

AI_PROVIDER = os.getenv("AI_PROVIDER", "")
AI_MODEL = os.getenv("AI_MODEL", "")