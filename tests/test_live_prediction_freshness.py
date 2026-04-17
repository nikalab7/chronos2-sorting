from __future__ import annotations

import unittest

import pandas as pd

from chronos_equity_prototype.pipeline import ZeroShotResearchPipeline


class LivePredictionFreshnessTests(unittest.TestCase):
    def test_build_v2_live_freshness_metadata_flags_newer_observed_session(self) -> None:
        v2_base_frame = pd.DataFrame(
            {
                "session_date": [
                    "2026-04-15T04:00:00+00:00",
                    "2026-04-16T04:00:00+00:00",
                ]
            }
        )

        metadata = ZeroShotResearchPipeline._build_v2_live_freshness_metadata(
            v2_base_frame,
            signal_session=pd.Timestamp("2026-04-15T04:00:00+00:00"),
        )

        self.assertTrue(bool(metadata["has_newer_observed_session_than_signal"]))
        self.assertEqual(metadata["prediction_freshness"], "newer_incomplete_session_present")
        self.assertEqual(pd.Timestamp(metadata["observed_latest_session"]), pd.Timestamp("2026-04-16T04:00:00+00:00"))

    def test_build_v2_live_freshness_metadata_marks_clean_next_session(self) -> None:
        v2_base_frame = pd.DataFrame(
            {
                "session_date": [
                    "2026-04-15T04:00:00+00:00",
                ]
            }
        )

        metadata = ZeroShotResearchPipeline._build_v2_live_freshness_metadata(
            v2_base_frame,
            signal_session=pd.Timestamp("2026-04-15T04:00:00+00:00"),
        )

        self.assertFalse(bool(metadata["has_newer_observed_session_than_signal"]))
        self.assertEqual(metadata["prediction_freshness"], "clean_next_session")


if __name__ == "__main__":
    unittest.main()
