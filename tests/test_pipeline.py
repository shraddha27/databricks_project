import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Minimal stubs allow the validation to run in a lightweight local environment
# without Databricks-specific dependencies installed.
for module_name in [
    "mlflow",
    "mlflow.sklearn",
    "mlflow.models",
    "pyspark",
    "pyspark.sql",
    "pyspark.sql.functions",
    "sklearn",
    "sklearn.compose",
    "sklearn.ensemble",
    "sklearn.impute",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.pipeline",
    "sklearn.preprocessing",
]:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

sys.modules["mlflow"].set_registry_uri = lambda *args, **kwargs: None
sys.modules["mlflow"].set_experiment = lambda *args, **kwargs: None
sys.modules["mlflow"].start_run = lambda *args, **kwargs: types.SimpleNamespace(
    __enter__=lambda self: self,
    __exit__=lambda self, exc_type, exc, tb: False,
    info=types.SimpleNamespace(run_id="test-run"),
)
sys.modules["mlflow"].log_metrics = lambda *args, **kwargs: None
sys.modules["mlflow"].log_params = lambda *args, **kwargs: None
sys.modules["mlflow"].set_tag = lambda *args, **kwargs: None
sys.modules["mlflow"].register_model = lambda *args, **kwargs: types.SimpleNamespace(version="1")

sys.modules["mlflow.sklearn"].log_model = lambda *args, **kwargs: types.SimpleNamespace(
    model_uri="test-model"
)
sys.modules["mlflow.models"].infer_signature = lambda *args, **kwargs: None

sys.modules["pyspark.sql"].DataFrame = object
sys.modules["pyspark.sql"].SparkSession = object
sys.modules["pyspark.sql"].functions = sys.modules["pyspark.sql.functions"]

setattr(sys.modules["sklearn.compose"], "ColumnTransformer", type("ColumnTransformer", (), {}))
setattr(sys.modules["sklearn.compose"], "TransformedTargetRegressor", type("TransformedTargetRegressor", (), {}))
setattr(sys.modules["sklearn.ensemble"], "GradientBoostingRegressor", type("GradientBoostingRegressor", (), {}))
setattr(sys.modules["sklearn.impute"], "SimpleImputer", type("SimpleImputer", (), {}))
setattr(sys.modules["sklearn.metrics"], "mean_squared_error", lambda *args, **kwargs: 0.0)
setattr(sys.modules["sklearn.metrics"], "r2_score", lambda *args, **kwargs: 0.0)
setattr(sys.modules["sklearn.model_selection"], "train_test_split", lambda *args, **kwargs: ([], []))
setattr(sys.modules["sklearn.pipeline"], "Pipeline", type("Pipeline", (), {}))
setattr(sys.modules["sklearn.pipeline"], "make_pipeline", lambda *args, **kwargs: None)
setattr(sys.modules["sklearn.preprocessing"], "OneHotEncoder", type("OneHotEncoder", (), {}))
setattr(sys.modules["sklearn"], "compose", sys.modules["sklearn.compose"])
setattr(sys.modules["sklearn"], "ensemble", sys.modules["sklearn.ensemble"])
setattr(sys.modules["sklearn"], "impute", sys.modules["sklearn.impute"])
setattr(sys.modules["sklearn"], "metrics", sys.modules["sklearn.metrics"])
setattr(sys.modules["sklearn"], "model_selection", sys.modules["sklearn.model_selection"])
setattr(sys.modules["sklearn"], "pipeline", sys.modules["sklearn.pipeline"])
setattr(sys.modules["sklearn"], "preprocessing", sys.modules["sklearn.preprocessing"])

from column_names import normalize_column_name
from pipeline import compute_risk_label


class NormalizeColumnNameTests(unittest.TestCase):
    def test_converts_mixed_headers_to_snake_case(self):
        self.assertEqual(normalize_column_name("Review Scores Rating"), "review_scores_rating")

    def test_removes_punctuation_and_duplicate_separators(self):
        self.assertEqual(normalize_column_name("  Price ($)  "), "price")


class RiskLabelTests(unittest.TestCase):
    def test_highlights_risky_listings_when_signal_is_strong(self):
        self.assertEqual(compute_risk_label(550, 65, 4, 12, 18), 1)
        self.assertEqual(compute_risk_label(260, 92, 2, 4, 90), 0)


if __name__ == "__main__":
    unittest.main()