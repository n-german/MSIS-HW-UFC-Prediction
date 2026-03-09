from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)