import os
import unittest

import pandas as pd

from src.preprocessing import build_wireless_features


class ExcelDatasetSupportTest(unittest.TestCase):
    def test_excel_dataset_has_real_wireless_features_and_target(self):
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "WiFi_Transmission_Rate_Recommendation_Dataset_5000.xlsx",
        )
        df = pd.read_excel(data_path)

        features, target = build_wireless_features(df)

        self.assertIn("rssi", features.columns)
        self.assertIn("channel_utilization", features.columns)
        self.assertIn("recommended_transmission_rate", target.name)
        self.assertEqual(len(features), len(df))
        self.assertGreater(len(target.unique()), 1)


if __name__ == "__main__":
    unittest.main()
