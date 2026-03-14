import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def sample_payload():
    return {
        "consumers": [
            {"consumer_id": "SGCC_999", "consumption_values": [10.5, 10.2, 0.0, 0.0, 0.0, 8.5, 9.0]}
        ]
    }


# @patch("src.api.main.process_theft_analysis.delay")
def test_async_detection_submission(sample_payload):
    """Test that the API correctly hands off tasks to the worker queue."""
    # We no longer need to mock the return value to a specific string
    response = client.post("/detect", json=sample_payload)

    assert response.status_code == 202

    job_id = response.json().get("job_id")

    # Validate that job_id is not None
    assert job_id is not None

    # Validate that job_id is a valid UUID (this is the professional way)
    try:
        uuid.UUID(str(job_id))
    except ValueError:
        pytest.fail(f"job_id {job_id} is not a valid UUID")


@patch("src.workers.tasks.process_theft_analysis.AsyncResult")
def test_results_polling(mock_async_result):
    """Test the polling mechanism for retrieving completed analysis."""
    mock_job = MagicMock()
    mock_job.ready.return_value = True
    mock_job.result = [{"consumer_id": "SGCC_999", "risk_tier": "HIGH"}]
    mock_async_result.return_value = mock_job

    response = client.get("/results/test-job-uuid")

    assert response.status_code == 200
    assert response.json()["status"] == "Completed"
    assert response.json()["result"][0]["consumer_id"] == "SGCC_999"
