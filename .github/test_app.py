import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")
        self.assertEqual(
            response.get_json()["service"],
            "telecom-churn-prediction",
        )

    def test_high_risk_customer_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "tenure": 6,
                "monthly_charges": 95.0,
                "contract": "Month-to-month",
                "tech_support": "No",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn(result["prediction"], ["CHURN", "NO_CHURN"])
        self.assertIn(result["prediction_code"], [0, 1])

    def test_low_risk_customer_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "tenure": 60,
                "monthly_charges": 50.0,
                "contract": "Two year",
                "tech_support": "Yes",
                "internet_service": "DSL",
                "payment_method": "Bank transfer",
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn(result["prediction"], ["CHURN", "NO_CHURN"])
        self.assertIn(result["prediction_code"], [0, 1])

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "tenure": 6,
                "monthly_charges": 95.0,
                "contract": "Month-to-month",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_non_json_request_is_rejected(self):
        response = self.client.post("/predict", data="not json")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
