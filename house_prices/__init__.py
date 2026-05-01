"""House prices prediction package."""
from pathlib import Path

CONTINUOUS_FEATURES = ["GrLivArea", "TotalBsmtSF"]
CATEGORICAL_FEATURES = ["Neighborhood", "HouseStyle"]
FEATURE_COLUMNS = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
LABEL_COLUMN = "SalePrice"

MODELS_DIR = Path(__file__).parent.parent / "models"
