from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


FEATURE_NAMES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

KMEANS_PATH = BASE_DIR / "models" / "kmeans_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"