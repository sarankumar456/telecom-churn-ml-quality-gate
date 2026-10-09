import json
import os
import unittest

import joblib
import pandas as pd


class TelecomMLPipelineTests(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("telecom_churn.csv"))

    def test_model_exists(self):
        self.assertTrue(os.path.exists("telecom_churn_model.pkl"))

    def test_metrics_exist(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_metrics_are_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_model_prediction(self):
        model = joblib.load("telecom_churn_model.pkl")

        sample_customer = pd.DataFrame([{
            "tenure": 6,
            "monthly_charges": 95.0,
            "contract": "Month-to-month",
            "tech_support": "No",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check"
        }])

        prediction = model.predict(sample_customer)

        self.assertEqual(len(prediction), 1)
        self.assertIn(int(prediction[0]), [0, 1])


if __name__ == "__main__":
    unittest.main()
