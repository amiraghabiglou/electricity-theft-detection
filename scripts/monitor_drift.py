import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from src.monitoring.drift_detector import ElectricityDriftMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_drift_check(reference_path: str, current_path: str, output_path: str):
    logger.info(f"Loading reference data from {reference_path}")

    # In a real scenario, this might load multiple files from a directory
    # For this script, we assume a single parquet file or CSV
    if Path(reference_path).is_dir():
        # Grab the first parquet file in the reference directory
        ref_file = list(Path(reference_path).glob("*.parquet"))[0]
        ref_df = pd.read_parquet(ref_file)
    else:
        ref_df = pd.read_parquet(reference_path)

    logger.info(f"Loading current production data from {current_path}")
    curr_df = pd.read_parquet(current_path)

    logger.info("Initializing Drift Monitor...")
    monitor = ElectricityDriftMonitor(reference_data=ref_df)

    logger.info("Detecting drift...")
    reports = monitor.detect_drift(curr_df)

    # Convert reports to dictionaries for JSON serialization
    report_dicts = [
        {
            "feature_name": r.feature_name,
            "drift_detected": r.drift_detected,
            "statistic": r.statistic,
            "p_value": r.p_value,
            "percent_change": r.percent_change,
        }
        for r in reports
    ]

    # Save the detailed report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_dicts, f, indent=4)
    logger.info(f"Detailed drift report saved to {output_path}")

    # Check if an alert needs to be generated
    alert = monitor.generate_alert(reports)
    if alert:
        logger.error(f"Significant drift detected: {alert['summary']}")
        # Exit with code 1 to trigger the failure block in GitHub Actions
        sys.exit(1)
    else:
        logger.info("No significant drift detected. Model is stable.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data Drift Detection")
    parser.add_argument(
        "--reference", required=True, help="Path to reference data directory or file"
    )
    parser.add_argument("--current", required=True, help="Path to current production features")
    parser.add_argument("--output", required=True, help="Path to save the JSON report")

    args = parser.parse_args()

    try:
        run_drift_check(args.reference, args.current, args.output)
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        sys.exit(2)
