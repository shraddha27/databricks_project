# Airbnb Real Estate Valuation & Rental Income Lakehouse

A Databricks Asset Bundle for Databricks Free Edition serverless compute. It ingests the `sf-airbnb.csv` sample from the shared Unity Catalog volume, creates a three-layer Delta Lakehouse, trains a scikit-learn nightly-price model, and estimates rental income and income-based property value for each listing.

## Tables

- `bronze_airbnb_listings`: source CSV rows with source file and ingestion timestamp.
- `silver_airbnb_listings`: normalized headers, parsed nightly price, stable listing ID, and deduplicated priced listings.
- `gold_airbnb_valuation`: held-out listings only, with actual, model-predicted, and mean-baseline nightly prices, both prediction errors, gross rental-income estimates, estimated NOI, and income-approach property value.
- `data_quality_audit`: append-only per-run counts for missing/invalid prices, duplicate IDs, IQR-based unusual prices, warning flags, and audit status.

The Gold estimates use editable defaults of 65% occupancy, 35% operating expenses, and a 6% capitalization rate. These are modeling assumptions, not facts supplied by the Airbnb dataset. Property value is estimated as annual NOI divided by the cap rate; it is not a market appraisal.

## Deploy and run

Prerequisites: Databricks CLI with Asset Bundles support, an authenticated Databricks profile, and Unity Catalog write access to the `workspace` catalog.

1. Validate the bundle: `databricks bundle validate -t dev --profile dev`
2. Deploy the job: `databricks bundle deploy -t dev --profile dev`
3. Start the pipeline: `databricks bundle run airbnb_valuation_pipeline -t dev --profile dev`

The job uses a serverless environment and needs no cluster ID. The source defaults to `/Volumes/databricks_airbnb_sample_data/v01/sf-listings/sf-airbnb.csv`. Override it with `--var="source_path=/Volumes/.../your-listings.csv"` if needed, or override the output catalog with `--var="catalog=..."` if your workspace uses a different writable catalog.

The model trains on 80% of listings and evaluates predictions on the remaining 20%. It uses available numeric features plus room type, property type, and neighborhood categories, and predicts log-transformed prices to reduce outlier impact. Gold contains only held-out listings, with model predictions compared against a baseline that always predicts the training-set average price. Data-quality checks run before model fitting; invalid prices are excluded and duplicates are deduplicated, while IQR price outliers are flagged and retained. Each run logs metrics and a model artifact to the `airbnb-real-estate-valuation` MLflow experiment and registers a new Unity Catalog version at `workspace.airbnb_lakehouse.airbnb_nightly_price_model`.

The job overwrites Bronze, Silver, and Gold on each run; quality audit rows append to preserve history. A daily 06:00 UTC schedule is configured but paused until model quality is approved. The current source is a managed volume, while Databricks file-arrival triggers require an external-location-backed path. After approval, unpause the daily schedule to refresh once per day. For event-only refresh, first move the incoming CSV to an eligible external location and configure a file-arrival trigger.

By default, results are written to `workspace.airbnb_lakehouse.gold_airbnb_valuation`. RMSE, R2, and the model's RMSE improvement over baseline are printed in the job output.

## Hosted Airbnb Analyst

The hosted Databricks App is [Airbnb Valuation Analyst](https://agent-airbnb-valuation-7474652903609181.aws.databricksapps.com). Sign in with your Databricks workspace account to use it. The app uses the `databricks-gpt-oss-120b` serving endpoint, Databricks-managed SQL MCP for fixed analytics, and Databricks-managed AI Search MCP for semantic listing retrieval. The listing index is `workspace.airbnb_lakehouse.airbnb_listing_search`, backed by `silver_airbnb_listings.semantic_search_text` and `databricks-gte-large-en`. SQL access is restricted to `execute_sql_read_only` and `poll_sql_result`; the write-capable SQL MCP tool is explicitly blocked.

The app runs under its own Databricks service principal. Its bundle grants only `CAN_QUERY` on the model endpoint, `CAN_USE` on the SQL warehouse, and `SELECT` on the Gold valuation and quality-audit tables. Redeploy updates with `databricks bundle deploy -t dev --profile dev` and `databricks bundle run airbnb_valuation_agent -t dev --profile dev`.

## Local checks

The pipeline itself runs on Databricks serverless compute, which supplies PySpark. Run local unit checks with `python -m unittest discover -s tests`.