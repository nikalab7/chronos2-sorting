from __future__ import annotations

import unittest

import pandas as pd

from chronos_equity_prototype.config import ForecasterConfig
from chronos_equity_prototype.pipeline import ZeroShotResearchPipeline
from chronos_equity_prototype.v2_basket import BasketPredictionInput


class _StaleLatestOnlyBuilder:
    def build_prediction_inputs(
        self,
        base_frame: pd.DataFrame,
        *,
        context_length: int,
        prediction_length: int,
        basket_mode: str = "full",
        latest_only: bool = False,
        use_covariates: bool = True,
        beta_mode: str = "unit_beta_1.0",
    ):
        del base_frame, context_length, prediction_length, basket_mode, latest_only, use_covariates, beta_mode
        prediction_input = BasketPredictionInput(
            basket_id="growth",
            item_id="growth|2026-04-15",
            signal_session=pd.Timestamp("2026-04-15T04:00:00+00:00"),
            prediction_session=pd.Timestamp("2026-04-16T04:00:00+00:00"),
            model_target_columns=("AAA", "SPY"),
            target_columns=("AAA",),
            history_df=pd.DataFrame(),
            future_df=pd.DataFrame(),
            beta_by_ticker={"AAA": 1.0},
            realized_future_paths={},
        )
        return [prediction_input], pd.DataFrame(), pd.DataFrame()


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

    def test_run_v2_next_session_prediction_refuses_started_target_session(self) -> None:
        pipeline = ZeroShotResearchPipeline(v2_builder=_StaleLatestOnlyBuilder())
        v2_base_frame = pd.DataFrame(
            {
                "session_date": [
                    "2026-04-15T04:00:00+00:00",
                    "2026-04-16T04:00:00+00:00",
                ]
            }
        )

        with self.assertRaisesRegex(ValueError, "newer observed session|not strictly after the current US market day"):
            pipeline.run_v2_next_session_prediction(
                v2_base_frame,
                ForecasterConfig(prediction_length=7),
                "live_test",
            )

    def test_run_v2_next_session_prediction_requires_seven_slot_horizon(self) -> None:
        pipeline = ZeroShotResearchPipeline()

        with self.assertRaisesRegex(ValueError, "one full next regular session = 7 slots"):
            pipeline.run_v2_next_session_prediction(
                pd.DataFrame(),
                ForecasterConfig(prediction_length=14),
                "live_test",
            )

    def test_run_v2_experiment_requires_seven_slot_horizon(self) -> None:
        pipeline = ZeroShotResearchPipeline()

        with self.assertRaisesRegex(ValueError, "one full next regular session = 7 slots"):
            pipeline.run_v2_experiment(
                pd.DataFrame(),
                ForecasterConfig(prediction_length=14),
                "backtest_test",
            )

    def test_assert_prediction_session_after_current_market_day_rejects_same_day_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "not strictly after the current US market day"):
            ZeroShotResearchPipeline._assert_prediction_session_after_current_market_day(
                pd.Timestamp("2026-04-17T04:00:00+00:00"),
                market_timezone="America/New_York",
                now=pd.Timestamp("2026-04-17 14:45:00", tz="America/New_York"),
            )

    def test_assert_prediction_session_after_current_market_day_accepts_next_day_target(self) -> None:
        ZeroShotResearchPipeline._assert_prediction_session_after_current_market_day(
            pd.Timestamp("2026-04-20T04:00:00+00:00"),
            market_timezone="America/New_York",
            now=pd.Timestamp("2026-04-17 14:45:00", tz="America/New_York"),
        )

    def test_reconciliation_readiness_rejects_future_prediction_session(self) -> None:
        is_ready, message = ZeroShotResearchPipeline.reconciliation_readiness(
            [pd.Timestamp("2026-04-20T04:00:00+00:00")],
            market_timezone="America/New_York",
            now=pd.Timestamp("2026-04-18 12:00:00", tz="America/New_York"),
        )

        self.assertFalse(is_ready)
        self.assertIn("still in the future", message)
        self.assertIn("2026-04-20", message)

    def test_reconciliation_readiness_rejects_same_day_before_close(self) -> None:
        is_ready, message = ZeroShotResearchPipeline.reconciliation_readiness(
            [pd.Timestamp("2026-04-20T04:00:00+00:00")],
            market_timezone="America/New_York",
            now=pd.Timestamp("2026-04-20 14:00:00", tz="America/New_York"),
        )

        self.assertFalse(is_ready)
        self.assertIn("not fully complete yet", message)
        self.assertIn("2026-04-20", message)

    def test_reconciliation_readiness_accepts_same_day_after_close(self) -> None:
        is_ready, message = ZeroShotResearchPipeline.reconciliation_readiness(
            [pd.Timestamp("2026-04-20T04:00:00+00:00")],
            market_timezone="America/New_York",
            now=pd.Timestamp("2026-04-20 16:05:00", tz="America/New_York"),
        )

        self.assertTrue(is_ready)
        self.assertEqual(message, "")


if __name__ == "__main__":
    unittest.main()
