import unittest
from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_prediction_endpoint(self):
        response = self.client.post(
            "/predict",
            json={
                "tenure": 6,
                "monthly_charges": 95.0,
                "contract": "Month-to-month",
                "tech_support": "No",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.get_json()["prediction"],
            ["CHURN", "NO_CHURN"]
        )

    def test_missing_fields(self):
        response = self.client.post(
            "/predict",
            json={"tenure": 6}
        )
        self.assertEqual(response.status_code, 400)

    def test_invalid_json(self):
        response = self.client.post(
            "/predict",
            data="not json"
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
