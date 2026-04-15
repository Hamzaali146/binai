from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "CV Testing Framework"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/cv_testing.db"

    # Storage paths
    DATA_DIR: Path = BASE_DIR / "data"
    DATASETS_DIR: Path = BASE_DIR / "data" / "datasets"
    RESULTS_DIR: Path = BASE_DIR / "data" / "results"
    REPORTS_DIR: Path = BASE_DIR / "data" / "reports"
    FAILURES_DIR: Path = BASE_DIR / "data" / "failures"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:8501", "http://localhost:3000"]

    # Evaluation defaults
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    DEFAULT_IOU_THRESHOLD: float = 0.5
    MAX_FAILURE_IMAGES: int = 100

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist
for d in [
    settings.DATA_DIR,
    settings.DATASETS_DIR,
    settings.RESULTS_DIR,
    settings.REPORTS_DIR,
    settings.FAILURES_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
