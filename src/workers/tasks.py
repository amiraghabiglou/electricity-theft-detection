# src/workers/tasks.py
from dataclasses import asdict
import os
import pandas as pd
from celery import Celery

from src.features.extractors import ElectricityFeatureExtractor
from src.llm.report_generator import TheftReportGenerator
from src.models.ensemble import HybridTheftDetector


broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery("theft_detection", broker=broker_url, backend=result_backend)


@celery_app.task(bind=True, name="process_theft_analysis")
def process_theft_analysis(self, consumer_data_batch):
    """
    Background task to handle the CPU-intensive pipeline.
    """
    # 1. Feature Extraction
    extractor = ElectricityFeatureExtractor(n_jobs=1)  # Reduce parallel overhead in worker

    # --- DATA SHAPING FIX ---
    rows = []
    for consumer in consumer_data_batch:
        c_id = consumer["consumer_id"]
        for time_step, val in enumerate(consumer["consumption_data"]):
            rows.append({
                "id": c_id,  # ❌ Changed from "consumer_id" to "id"
                "time": time_step,
                "value": float(val)
            })

    df_long = pd.DataFrame(rows)

    # 1. Extract base statistical features (tsfresh expects "id" in df_long)
    features = extractor.extract_features(df_long)

    df_raw = pd.DataFrame(consumer_data_batch)

    consumption_df = pd.DataFrame(df_raw['consumption_data'].tolist())

    df_raw_wide = pd.concat([df_raw[['consumer_id']], consumption_df], axis=1)

    final_features = extractor.add_domain_features(features, df_raw_wide)
    # ---------------------------

    # 2. Prediction
    detector = HybridTheftDetector.load("models/hybrid_detector.joblib")
    results = detector.predict(final_features)

    # 3. Reasoning (Lazy Loading & Error Handling)
    final_output = []
    for res in results:
        report = None

        # RESTORE TO PRODUCTION THRESHOLD
        if res.fraud_probability > 0.6:
            try:
                report_gen = TheftReportGenerator(model_path="models/phi-3-q4.gguf")
                report = report_gen.generate_report(res)
            except Exception as e:
                report = f"Fraud detected, but LLM failed to load: {str(e)}"

        final_output.append({**asdict(res), "report": report})
