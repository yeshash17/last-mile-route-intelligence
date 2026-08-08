"""
config.py — central settings loaded from environment variables.
Copy .env.example to .env and fill in your values.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- External APIs ---
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
TWILIO_SID          = os.getenv("TWILIO_SID", "")
TWILIO_TOKEN        = os.getenv("TWILIO_TOKEN", "")
TWILIO_FROM_NUMBER  = os.getenv("TWILIO_FROM_NUMBER", "")

# --- Data storage ---
DATABASE_URL  = os.getenv("DATABASE_URL", "sqlite:///./route_intelligence.db")
MODEL_DIR     = os.getenv("MODEL_DIR", "./saved_models")

# --- Operational thresholds ---
SHIFT_DURATION_MINS        = int(os.getenv("SHIFT_DURATION_MINS", "480"))    # 8 hrs
MAX_VEHICLE_CAPACITY_KG    = float(os.getenv("MAX_VEHICLE_CAPACITY_KG", "500"))
REOPT_DELAY_THRESHOLD_MINS = int(os.getenv("REOPT_DELAY_THRESHOLD_MINS", "10"))
HIGH_RISK_THRESHOLD        = float(os.getenv("HIGH_RISK_THRESHOLD", "0.70"))  # P(fail) > 0.7 → redirect
MEDIUM_RISK_THRESHOLD      = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.40"))
ETA_ALERT_BUFFER_MINS      = int(os.getenv("ETA_ALERT_BUFFER_MINS", "30"))   # SMS this many mins before arrival

# --- VRP solver ---
VRP_TIME_LIMIT_SECONDS = int(os.getenv("VRP_TIME_LIMIT_SECONDS", "30"))
VRP_WAIT_SLACK_MINS    = int(os.getenv("VRP_WAIT_SLACK_MINS", "30"))         # max driver wait at a stop
