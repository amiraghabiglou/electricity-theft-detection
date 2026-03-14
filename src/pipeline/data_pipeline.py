"""
End-to-end data pipeline for Electricity Theft Detection.
Handles raw SGCC data ingestion, cleaning, and TSFRESH feature extraction.
"""
import argparse
import logging
import multiprocessing
from pathlib import Path

import pandas as pd

from src.features.extractors import ElectricityFeatureExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_data_pipeline(
    input_path: str, output_path: str, extract_tsfresh: bool = True, sample_size: int = None
):
    """
    Executes the data processing pipeline.

    1. Loads raw CSV (SGCC format)
    2. Cleans and handles missing values
    3. (Optional) Extracts TSFRESH features
    4. Saves to high-performance Parquet format for training
    """
    logger.info(f"🚀 Starting data pipeline. Input: {input_path}")

    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)

    df_raw = pd.read_csv(input_path)

    # 1. Standardize Kaggle schema to internal schema
    logger.info("Standardizing raw dataset columns...")
    df_raw = df_raw.rename(columns={"CONS_NO": "consumer_id", "FLAG": "label"})

    # 2. Optional Subsampling for local testing
    if sample_size and sample_size < len(df_raw):
        logger.warning(f"Subsampling dataset to {sample_size} rows for rapid testing!")
        df_raw = df_raw.sample(n=sample_size, random_state=42)

    logger.info(f"Processing data with shape: {df_raw.shape}")

    # 3. Initialize Extractor with Multiprocessing
    cores_to_use = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Initializing Extractor with {cores_to_use} CPU cores...")
    extractor = ElectricityFeatureExtractor(n_jobs=cores_to_use)

    logger.info("Cleaning data and handling missing values...")
    df_long = extractor.prepare_data(df_raw)

    if extract_tsfresh:
        logger.info("Extracting TSFRESH features...")
        features = extractor.extract_features(df_long)
        logger.info("Adding domain-specific theft features...")
        final_df = extractor.add_domain_features(features, df_raw)
    else:
        final_df = df_long

    if "label" in df_raw.columns:
        labels = df_raw.set_index("consumer_id")["label"]
        final_df = final_df.join(labels, how="left")
        final_df = final_df.dropna(subset=["label"])

    if final_df.empty:
        raise ValueError("Processed features dataframe is empty.")

    final_df = final_df.fillna(0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path)
    logger.info(f"✅ Pipeline complete. Processed data saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridGuard Data Pipeline")
    parser.add_argument("--input", default="data/raw/sgcc.csv")
    parser.add_argument("--output", default="data/processed/features.parquet")
    parser.add_argument("--extract-tsfresh", action="store_true", default=True)
    # ADD THE NEW ARGUMENT HERE
    parser.add_argument(
        "--sample", type=int, default=None, help="Number of rows to sample for quick testing"
    )

    args = parser.parse_args()

    try:
        run_data_pipeline(
            input_path=args.input,
            output_path=args.output,
            extract_tsfresh=args.extract_tsfresh,
            sample_size=args.sample,  # Pass it to the function
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)
