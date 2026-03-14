import time

import pandas as pd
import requests
import streamlit as st

# Configuration
API_URL = "http://api:8000"  # Uses Docker internal network name

st.set_page_config(page_title="GridGuard SLM", page_icon="⚡", layout="wide")

st.title("⚡ GridGuard: AI Electricity Theft Detection")
st.markdown("Analyze smart meter data using Hybrid ML (XGBoost) and Generative AI (Phi-3).")

st.sidebar.header("Input Parameters")
consumer_id = st.sidebar.text_input("Consumer ID", value="THEFT_999")

# Default anomalous data (sudden drop)
default_data = "6.1, 6.3, 5.9, 6.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
consumption_input = st.sidebar.text_area(
    "Daily Consumption (kWh, comma-separated)", value=default_data
)

if st.sidebar.button("Run Analysis", type="primary"):
    try:
        # 1. Parse Input
        consumption_list = [float(x.strip()) for x in consumption_input.split(",")]

        payload = {
            "consumers": [{"consumer_id": consumer_id, "consumption_data": consumption_list}]
        }

        # 2. Display the Input Data
        st.subheader(f"📊 Analyzing Meter: {consumer_id}")
        chart_data = pd.DataFrame(
            {"Days": range(1, len(consumption_list) + 1), "kWh": consumption_list}
        )
        st.line_chart(chart_data.set_index("Days"))

        # 3. Trigger the Backend API
        with st.spinner("Submitting task to ML workers..."):
            post_response = requests.post(f"{API_URL}/detect", json=payload)
            post_response.raise_for_status()
            job_id = post_response.json().get("job_id")

        st.info(f"Task queued successfully! Job ID: `{job_id}`")

        # 4. Poll for Results
        with st.spinner("AI models are crunching the numbers... (This takes a moment)"):
            status = "Processing"
            while status in ["Processing", "PENDING"]:
                time.sleep(3)  # Wait 3 seconds between polls
                get_response = requests.get(f"{API_URL}/results/{job_id}")
                get_response.raise_for_status()
                result_data = get_response.json()
                status = result_data.get("status")

        # 5. Display Results
        if status == "Completed":
            st.success("Analysis Complete!")

            analysis = result_data["result"][0]
            prob = analysis["fraud_probability"] * 100

            # Metrics Row
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Tier", analysis["risk_tier"])
            col2.metric("Fraud Probability", f"{prob:.1f}%")
            col3.metric("Anomaly Score", f"{analysis['anomaly_score']:.3f}")

            # SHAP Explanation
            st.warning(f"**Mathematical Driver (SHAP):** {analysis['explanation']}")

            # LLM Report
            st.subheader("📝 Generative AI Investigation Report")
            if analysis["report"]:
                st.write(analysis["report"])
            else:
                st.write("*No report generated (Risk below threshold).*")

        else:
            st.error(f"Pipeline failed or returned unknown status: {status}")

    except Exception as e:
        st.error(f"Error communicating with backend: {e}")
