"""Build an Airbnb valuation and rental-income Delta Lakehouse."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import mlflow
import mlflow.sklearn
from column_names import normalize_column_name
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


MODEL_FEATURES = (
    "accommodates",
    "bedrooms",
    "bathrooms",
    "beds",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "availability_365",
    "latitude",
    "longitude",
)

MODEL_CATEGORICAL_FEATURES = (
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
)

SEARCH_TEXT_FIELDS = (
    "name",
    "summary",
    "description",
    "neighborhood_overview",
    "amenities",
    "property_type",
    "room_type",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "city",
)


def compute_risk_label(
    nightly_price: float | None,
    review_score_rating: float | None,
    bedrooms: float | None,
    minimum_nights: float | None,
    number_of_reviews: float | None,
) -> int:
    """Return a simple binary risk label for a listing.

    The rule intentionally emphasizes the kinds of signals a risk team would review:
    unusually high nightly-price anchors, weak guest sentiment, low review volume,
    large minimum-stay requirements, and larger property footprints.
    """
    risk_score = 0.0
    if nightly_price is not None and nightly_price >= 450:
        risk_score += 0.45
    elif nightly_price is not None and nightly_price >= 300:
        risk_score += 0.20

    if review_score_rating is not None:
        if review_score_rating < 70:
            risk_score += 0.35
        elif review_score_rating < 85:
            risk_score += 0.15

    if bedrooms is not None and bedrooms >= 4:
        risk_score += 0.10
    if minimum_nights is not None and minimum_nights >= 10:
        risk_score += 0.20
    if number_of_reviews is not None and number_of_reviews < 20:
        risk_score += 0.25

    return 1 if risk_score >= 0.75 else 0


def read_bronze(spark: SparkSession, source_path: str) -> DataFrame:
    raw = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .option("recursiveFileLookup", "true")
        .option("pathGlobFilter", "*.csv")
        .load(source_path)
    )
    return raw.withColumn("_source_file", F.col("_metadata.file_path")).withColumn(
        "_ingested_at", F.current_timestamp()
    )


def build_silver(bronze: DataFrame) -> tuple[DataFrame, dict[str, float]]:
    renamed = bronze
    for column_name in bronze.columns:
        if not column_name.startswith("_"):
            renamed = renamed.withColumnRenamed(
                column_name, normalize_column_name(column_name)
            )

    if "price" not in renamed.columns:
        raise ValueError("Listings data must contain a price column.")

    price_text = F.regexp_replace(F.col("price").cast("string"), r"[^0-9.-]", "")
    cleaned = renamed.withColumn("nightly_price", price_text.cast("double"))
    if "id" in cleaned.columns:
        cleaned = cleaned.withColumn("listing_id", F.col("id").cast("string"))
    elif "listing_id" in cleaned.columns:
        cleaned = cleaned.withColumn("listing_id", F.col("listing_id").cast("string"))
    else:
        cleaned = cleaned.withColumn("listing_id", F.monotonically_increasing_id().cast("string"))

    if "bathrooms" not in cleaned.columns and "bathrooms_text" in cleaned.columns:
        cleaned = cleaned.withColumn(
            "bathrooms",
            F.regexp_extract(F.col("bathrooms_text"), r"([0-9]+(?:\.[0-9]+)?)", 1).cast("double"),
        )

    for feature in MODEL_FEATURES:
        if feature in cleaned.columns:
            cleaned = cleaned.withColumn(feature, F.col(feature).cast("double"))

    if "name" not in cleaned.columns:
        cleaned = cleaned.withColumn("name", F.lit(None).cast("string"))
    if "neighbourhood_cleansed" not in cleaned.columns:
        cleaned = cleaned.withColumn("neighbourhood_cleansed", F.lit(None).cast("string"))

    search_fields = [name for name in SEARCH_TEXT_FIELDS if name in cleaned.columns]
    if not search_fields:
        raise ValueError("No listing text fields are available for semantic search.")
    cleaned = cleaned.withColumn(
        "semantic_search_text",
        F.concat_ws(
            " | ",
            *[
                F.when(
                    F.col(field_name).isNotNull()
                    & (F.length(F.trim(F.col(field_name).cast("string"))) > 0),
                    F.concat(
                        F.lit(f"{field_name.replace('_', ' ')}: "),
                        F.trim(F.col(field_name).cast("string")),
                    ),
                )
                for field_name in search_fields
            ],
        ),
    )

    summary = cleaned.agg(
        F.count(F.lit(1)).alias("source_rows"),
        F.sum(F.when(F.col("nightly_price").isNull(), 1).otherwise(0)).alias(
            "missing_price_rows"
        ),
        F.sum(F.when(F.col("nightly_price") <= 0, 1).otherwise(0)).alias(
            "nonpositive_price_rows"
        ),
        F.expr("percentile_approx(nightly_price, 0.25, 10000)").alias("price_q1"),
        F.expr("percentile_approx(nightly_price, 0.75, 10000)").alias("price_q3"),
    ).first()
    duplicate_rows = (
        cleaned.groupBy("listing_id")
        .count()
        .agg(
            F.sum(
                F.when(F.col("count") > 1, F.col("count") - 1).otherwise(0)
            ).alias("duplicate_rows")
        )
        .first()["duplicate_rows"]
        or 0
    )
    q1, q3 = summary["price_q1"], summary["price_q3"]
    lower_fence = float(q1 - 1.5 * (q3 - q1)) if q1 is not None and q3 is not None else None
    upper_fence = float(q3 + 1.5 * (q3 - q1)) if q1 is not None and q3 is not None else None
    unusual_price_rows = 0
    if lower_fence is not None and upper_fence is not None:
        unusual_price_rows = cleaned.filter(
            F.col("nightly_price").isNotNull()
            & ((F.col("nightly_price") < lower_fence) | (F.col("nightly_price") > upper_fence))
        ).count()

    quality = {
        "source_rows": float(summary["source_rows"] or 0),
        "missing_price_rows": float(summary["missing_price_rows"] or 0),
        "nonpositive_price_rows": float(summary["nonpositive_price_rows"] or 0),
        "duplicate_listing_rows": float(duplicate_rows),
        "unusual_price_rows": float(unusual_price_rows),
        "price_lower_fence": lower_fence,
        "price_upper_fence": upper_fence,
    }
    silver = (
        cleaned.filter(F.col("nightly_price").isNotNull() & (F.col("nightly_price") > 0))
        .dropDuplicates(["listing_id"])
        .withColumn("processed_at", F.current_timestamp())
    )
    return silver, quality


def write_quality_audit(
    spark: SparkSession,
    pipeline_run_id: str,
    source_path: str,
    quality: dict[str, float],
    catalog: str,
    schema: str,
) -> tuple[str, list[str]]:
    warnings = []
    if quality["missing_price_rows"]:
        warnings.append("missing_or_unparseable_prices")
    if quality["nonpositive_price_rows"]:
        warnings.append("nonpositive_prices")
    if quality["duplicate_listing_rows"]:
        warnings.append("duplicate_listing_ids")
    if quality["unusual_price_rows"]:
        warnings.append("nightly_price_iqr_outliers")
    status = "WARN" if warnings else "PASS"

    audit = spark.createDataFrame(
        [(
            pipeline_run_id,
            source_path,
            int(quality["source_rows"]),
            int(quality["missing_price_rows"]),
            int(quality["nonpositive_price_rows"]),
            int(quality["duplicate_listing_rows"]),
            int(quality["unusual_price_rows"]),
            quality["price_lower_fence"],
            quality["price_upper_fence"],
            status,
            ",".join(warnings),
        )],
        "pipeline_run_id STRING, source_path STRING, source_rows LONG, "
        "missing_price_rows LONG, nonpositive_price_rows LONG, "
        "duplicate_listing_rows LONG, unusual_price_rows LONG, "
        "price_lower_fence DOUBLE, price_upper_fence DOUBLE, "
        "quality_status STRING, warning_flags STRING",
    ).withColumn("quality_checked_at", F.current_timestamp())
    audit.write.format("delta").mode("append").saveAsTable(
        f"{catalog}.{schema}.data_quality_audit"
    )
    print(
        f"Data quality {status}: source_rows={quality['source_rows']:.0f}, "
        f"missing_prices={quality['missing_price_rows']:.0f}, "
        f"nonpositive_prices={quality['nonpositive_price_rows']:.0f}, "
        f"duplicate_rows={quality['duplicate_listing_rows']:.0f}, "
        f"unusual_price_rows={quality['unusual_price_rows']:.0f}, "
        f"flags={','.join(warnings) or 'none'}"
    )
    return status, warnings


def build_gold(
    silver: DataFrame,
    occupancy_rate: float,
    expense_ratio: float,
    cap_rate: float,
) -> tuple[DataFrame, dict[str, float], Any, list[str], Any]:
    numeric_features = [name for name in MODEL_FEATURES if name in silver.columns]
    categorical_features = [
        name for name in MODEL_CATEGORICAL_FEATURES if name in silver.columns
    ]
    features = numeric_features + categorical_features
    if not features:
        raise ValueError("No supported numeric listing features are available for training.")
    if not 0 < occupancy_rate <= 1:
        raise ValueError("occupancy_rate must be greater than 0 and at most 1.")
    if not 0 <= expense_ratio < 1:
        raise ValueError("expense_ratio must be at least 0 and less than 1.")
    if cap_rate <= 0:
        raise ValueError("cap_rate must be greater than 0.")

    training = silver.select("listing_id", "nightly_price", *features).toPandas()
    if len(training) < 10:
        raise ValueError("At least 10 priced listings are required to train the valuation model.")
    for feature in categorical_features:
        training[feature] = training[feature].where(
            training[feature].notna(), "__missing__"
        ).astype(str)

    train_set, test_set = train_test_split(training, test_size=0.2, random_state=42)
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median", add_indicator=True)),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(
                        handle_unknown="ignore", min_frequency=10, sparse_output=False
                    ),
                ),
                categorical_features,
            )
        )

    regressor = Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            (
                "gradient_boosting",
                GradientBoostingRegressor(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=3,
                    min_samples_leaf=10,
                    loss="huber",
                    random_state=42,
                ),
            ),
        ]
    )
    model = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )
    model.fit(train_set[features], train_set["nightly_price"])
    test_predictions = model.predict(test_set[features])
    baseline_price = float(train_set["nightly_price"].mean())
    baseline_predictions = np.full(len(test_set), baseline_price)
    baseline_rmse = float(
        mean_squared_error(test_set["nightly_price"], baseline_predictions) ** 0.5
    )
    metrics = {
        "rmse": float(mean_squared_error(test_set["nightly_price"], test_predictions) ** 0.5),
        "r2": float(r2_score(test_set["nightly_price"], test_predictions)),
        "baseline_rmse": baseline_rmse,
        "baseline_r2": float(r2_score(test_set["nightly_price"], baseline_predictions)),
        "rmse_improvement_pct": float(
            (baseline_rmse - mean_squared_error(test_set["nightly_price"], test_predictions) ** 0.5)
            / baseline_rmse
            * 100
        ),
        "train_rows": float(len(train_set)),
        "test_rows": float(len(test_set)),
    }
    print(
        f"Model features: numeric={numeric_features}, "
        f"categorical={categorical_features}"
    )

    scored = silver.sparkSession.createDataFrame(
        [
            (str(listing_id), float(prediction), baseline_price)
            for listing_id, prediction in zip(test_set["listing_id"], test_predictions)
        ],
        "listing_id STRING, predicted_nightly_price DOUBLE, baseline_nightly_price DOUBLE",
    )
    risk_label_udf = F.udf(compute_risk_label, "int")
    gold = (
        silver.drop("predicted_nightly_price")
        .join(scored, on="listing_id", how="inner")
        .withColumn(
            "prediction_error",
            F.col("predicted_nightly_price") - F.col("nightly_price"),
        )
        .withColumn("absolute_prediction_error", F.abs(F.col("prediction_error")))
        .withColumn(
            "baseline_prediction_error",
            F.col("baseline_nightly_price") - F.col("nightly_price"),
        )
        .withColumn(
            "absolute_baseline_prediction_error",
            F.abs(F.col("baseline_prediction_error")),
        )
        .withColumn(
            "risk_label",
            risk_label_udf(
                F.col("nightly_price"),
                F.col("review_scores_rating"),
                F.col("bedrooms"),
                F.col("minimum_nights"),
                F.col("number_of_reviews"),
            ),
        )
        .withColumn(
            "estimated_monthly_rental_income",
            F.col("predicted_nightly_price") * F.lit(30 * occupancy_rate),
        )
        .withColumn(
            "estimated_annual_rental_income",
            F.col("estimated_monthly_rental_income") * F.lit(12),
        )
        .withColumn(
            "estimated_annual_noi",
            F.col("estimated_annual_rental_income") * F.lit(1 - expense_ratio),
        )
        .withColumn(
            "estimated_property_value",
            F.col("estimated_annual_noi") / F.lit(cap_rate),
        )
        .withColumn("assumed_occupancy_rate", F.lit(occupancy_rate))
        .withColumn("assumed_expense_ratio", F.lit(expense_ratio))
        .withColumn("assumed_cap_rate", F.lit(cap_rate))
        .withColumn("valuation_created_at", F.current_timestamp())
    )
    return gold, metrics, model, features, train_set[features].head(5)


def run_pipeline(args: argparse.Namespace) -> None:
    spark = SparkSession.builder.appName("airbnb-real-estate-valuation").getOrCreate()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{args.catalog}`.`{args.schema}`")
    model_name = f"{args.catalog}.{args.schema}.airbnb_nightly_price_model"
    mlflow.set_registry_uri("databricks-uc")
    current_user = spark.sql("SELECT current_user() AS username").first()["username"]
    experiment_path = f"/Users/{current_user}/airbnb-real-estate-valuation"
    mlflow.set_experiment(experiment_path)

    with mlflow.start_run(run_name="airbnb-nightly-price") as run:
        bronze = read_bronze(spark, args.source_path)
        silver, quality = build_silver(bronze)
        quality_status, warnings = write_quality_audit(
            spark, run.info.run_id, args.source_path, quality, args.catalog, args.schema
        )
        quality_metrics = {
            f"data_quality_{name}": float(value)
            for name, value in quality.items()
            if value is not None
        }
        mlflow.log_metrics(quality_metrics)
        mlflow.set_tag("data_quality_status", quality_status)
        mlflow.set_tag("data_quality_flags", ",".join(warnings))

        gold, metrics, model, features, input_example = build_gold(
            silver, args.occupancy_rate, args.expense_ratio, args.cap_rate
        )
        mlflow.log_params(
            {
                "source_path": args.source_path,
                "model_type": "GradientBoostingRegressor with log1p target",
                "numeric_features": ",".join(name for name in features if name in MODEL_FEATURES),
                "categorical_features": ",".join(
                    name for name in features if name in MODEL_CATEGORICAL_FEATURES
                ),
                "train_test_split": "80/20",
                "random_state": 42,
                "n_estimators": 120,
                "learning_rate": 0.05,
                "max_depth": 3,
                "min_samples_leaf": 10,
                "occupancy_rate": args.occupancy_rate,
                "expense_ratio": args.expense_ratio,
                "cap_rate": args.cap_rate,
            }
        )
        mlflow.log_metrics(
            {
                "model_rmse_usd": metrics["rmse"],
                "model_r2": metrics["r2"],
                "baseline_rmse_usd": metrics["baseline_rmse"],
                "baseline_r2": metrics["baseline_r2"],
                "rmse_improvement_pct": metrics["rmse_improvement_pct"],
                "train_rows": metrics["train_rows"],
                "test_rows": metrics["test_rows"],
            }
        )
        signature = infer_signature(input_example, model.predict(input_example))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        bronze.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{args.catalog}.{args.schema}.bronze_airbnb_listings"
        )
        silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{args.catalog}.{args.schema}.silver_airbnb_listings"
        )
        spark.sql(
            f"ALTER TABLE `{args.catalog}`.`{args.schema}`.`silver_airbnb_listings` "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
        )
        gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{args.catalog}.{args.schema}.gold_airbnb_valuation"
        )

        registered_model = mlflow.register_model(model_info.model_uri, model_name)
        mlflow.set_tag("registered_model_name", model_name)
        mlflow.set_tag("registered_model_version", str(registered_model.version))

    print(f"Loaded listing CSV files from {args.source_path}")
    print(
        f"Holdout rows: train={metrics['train_rows']:.0f}, "
        f"test={metrics['test_rows']:.0f}"
    )
    print(
        f"Mean-price baseline: RMSE={metrics['baseline_rmse']:.2f}, "
        f"R2={metrics['baseline_r2']:.3f}"
    )
    print(
        f"Improved model: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}, "
        f"RMSE improvement={metrics['rmse_improvement_pct']:.1f}%"
    )
    print(f"Registered model version: {model_name} v{registered_model.version}")
    print(f"Wrote tables to {args.catalog}.{args.schema}; MLflow run_id={run.info.run_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-path",
        default="/Volumes/databricks_airbnb_sample_data/v01/sf-listings/sf-airbnb.csv",
    )
    parser.add_argument("--catalog", default="workspace")
    parser.add_argument("--schema", default="airbnb_lakehouse")
    parser.add_argument("--occupancy-rate", type=float, default=0.65)
    parser.add_argument("--expense-ratio", type=float, default=0.35)
    parser.add_argument("--cap-rate", type=float, default=0.06)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())