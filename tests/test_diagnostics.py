from __future__ import annotations

import unittest

import pandas as pd

from chronos_equity_prototype.diagnostics import (
    EXACT_OPEN_ENTRY_FORMULA,
    FIRST_15M_ENTRY_FORMULA,
    FIRST_5M_ENTRY_FORMULA,
    STRICT_EXECUTION_MODE_LABEL,
    apply_unit_beta_to_forecasts,
    build_backtest_scorecard,
    generate_near_open_execution_analysis,
    generate_signal_execution_analysis,
    generate_v2_diagnostics,
)
from chronos_equity_prototype.portfolio import PortfolioBuilder


class DiagnosticsTests(unittest.TestCase):
    def _make_forecasts(self) -> pd.DataFrame:
        rows = []
        signal_sessions = [pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-04-02", tz="UTC")]
        prediction_sessions = [pd.Timestamp("2026-04-02", tz="UTC"), pd.Timestamp("2026-04-03", tz="UTC")]
        day_values = [
            {
                "AAA": (0.8, 0.2, 0.5, 0.03, 0.01),
                "BBB": (0.4, 1.0, 0.3, 0.01, -0.02),
                "CCC": (-0.3, -0.5, -0.2, -0.01, 0.01),
                "DDD": (-0.7, -1.2, -0.4, -0.03, -0.03),
            },
            {
                "AAA": (0.5, 0.1, 0.4, 0.02, 0.015),
                "BBB": (0.2, 0.8, 0.2, 0.00, -0.01),
                "CCC": (-0.1, -0.4, -0.1, -0.02, 0.005),
                "DDD": (-0.6, -1.0, -0.3, -0.01, -0.02),
            },
        ]
        for day_index, (signal_session, prediction_session) in enumerate(zip(signal_sessions, prediction_sessions)):
            for ticker, values in day_values[day_index].items():
                score_plain, score_z, score_risk, realized_day_return, beta_used = values
                rows.append(
                    {
                        "ticker": ticker,
                        "basket_id": "growth" if ticker in {"AAA", "BBB"} else "defensive_value",
                        "signal_session": signal_session,
                        "prediction_session": prediction_session,
                        "stock_q10_path": [(score_plain / 2.0) - 0.1, (score_plain / 2.0) - 0.1],
                        "stock_q50_path": [score_plain / 2.0, score_plain / 2.0],
                        "stock_q90_path": [(score_plain / 2.0) + 0.1, (score_plain / 2.0) + 0.1],
                        "spy_q10_path": [-0.05, -0.05],
                        "spy_q50_path": [0.0, 0.0],
                        "spy_q90_path": [0.05, 0.05],
                        "score_plain": score_plain,
                        "score_z": score_z,
                        "score_risk": score_risk,
                        "score_q50": score_z,
                        "beta_used": beta_used,
                        "alpha_q50_path": [score_plain / 2.0, score_plain / 2.0],
                        "realized_alpha_path": [realized_day_return / 2.0, realized_day_return / 2.0],
                        "realized_day_return": realized_day_return,
                    }
                )
        return pd.DataFrame(rows)

    def test_portfolio_builder_can_rank_by_custom_score_column(self) -> None:
        forecasts = self._make_forecasts()

        default_book = PortfolioBuilder().rank_and_size(forecasts, n_long=1, n_short=1)
        plain_book = PortfolioBuilder().rank_and_size(forecasts, n_long=1, n_short=1, score_column="score_plain")

        self.assertEqual(default_book.frame.iloc[0]["ticker"], "BBB")
        self.assertEqual(plain_book.frame.iloc[0]["ticker"], "AAA")
        self.assertEqual(plain_book.frame.iloc[0]["ranking_score_column"], "score_plain")

    def test_generate_v2_diagnostics_compares_score_columns(self) -> None:
        forecasts = self._make_forecasts()
        labeled_frame = pd.DataFrame(
            [
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.015, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.005, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.03, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.0, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.0, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
            ]
        )

        outputs = generate_v2_diagnostics(
            forecasts,
            labeled_frame=labeled_frame,
            n_long=1,
            n_short=1,
            costs_bps=(10, 20, 40),
        )

        comparison = outputs["ranking_signal_comparison"]
        self.assertEqual(set(comparison["score_column"].tolist()), {"score_plain", "score_z", "score_risk"})
        self.assertIn("top1_capture_mean", comparison.columns)
        self.assertIn("mean_daily_long_short_spread", comparison.columns)
        self.assertIn("mean_beta", outputs["beta_summary"].columns)
        self.assertIn("variant", outputs["beta_effect_comparison"].columns)
        self.assertIn("prediction_days", outputs["data_quality_overview"]["metric"].tolist())

    def test_apply_unit_beta_to_forecasts_rebuilds_scores_and_realized_paths(self) -> None:
        forecasts = self._make_forecasts()
        forecasts["beta_used"] = 0.5
        labeled_frame = pd.DataFrame(
            [
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.02, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.015, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.005, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.005, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
            ]
        )

        adjusted = apply_unit_beta_to_forecasts(forecasts, labeled_frame)

        self.assertTrue((adjusted["beta_used"] == 1.0).all())
        self.assertIn("beta_mode", adjusted.columns)
        self.assertIn("realized_alpha_path", adjusted.columns)
        self.assertIn("score_z", adjusted.columns)

    def test_generate_signal_execution_analysis_returns_filter_and_execution_reports(self) -> None:
        forecasts = self._make_forecasts()
        labeled_frame = pd.DataFrame(
            [
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.02, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.015, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.005, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.005, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
            ]
        )
        adjusted = apply_unit_beta_to_forecasts(forecasts, labeled_frame)

        outputs = generate_signal_execution_analysis(
            adjusted,
            labeled_frame=labeled_frame,
            n_long=1,
            n_short=1,
            tail_fractions=(0.25,),
            costs_bps=(10, 20, 40),
            book_mode="tail_fraction",
            tail_fraction=0.25,
            default_borrow_fee_annual_bps=50.0,
            unavailable_tickers=("BBB",),
        )

        self.assertIn("score_behavior_summary", outputs)
        self.assertIn("tail_filter_comparison", outputs)
        self.assertIn("execution_sensitivity_comparison", outputs)
        self.assertIn("basket_book_comparison", outputs)
        self.assertIn("side_book_comparison", outputs)
        self.assertIn("basket_side_summary", outputs)
        self.assertIn("basket_contribution_summary", outputs)
        self.assertIn("ticker_attribution_summary", outputs)
        self.assertIn("live_scoreboard_positions", outputs)
        self.assertIn("live_scoreboard_daily", outputs)
        self.assertIn("oos_window_summary", outputs)
        self.assertIn("oos_concentration_summary", outputs)
        self.assertIn("side_isolation_summary", outputs)
        self.assertIn("borrow_overlay_summary", outputs)
        self.assertIn("borrow_overlay_positions", outputs)
        self.assertIn("tail_fraction", outputs["tail_filter_comparison"].columns)
        self.assertIn("execution_mode", outputs["execution_sensitivity_comparison"].columns)
        self.assertIn("score_plain", outputs["live_scoreboard_positions"].columns)
        self.assertIn("score_z", outputs["live_scoreboard_positions"].columns)
        self.assertIn("book_impact", outputs["live_scoreboard_positions"].columns)
        self.assertIn("best_contributor", outputs["live_scoreboard_daily"].columns)
        self.assertIn("target_gap_20d", outputs["oos_window_summary"].columns)
        self.assertIn("top2_abs_pnl_share", outputs["oos_concentration_summary"].columns)
        self.assertEqual(set(outputs["side_isolation_summary"]["side"].tolist()), {"long", "short"})
        borrow_positions = outputs["borrow_overlay_positions"]
        self.assertIn("borrow_excluded", borrow_positions.columns)
        self.assertTrue(bool(borrow_positions.loc[borrow_positions["ticker"] == "BBB", "borrow_excluded"].any()))
        self.assertGreater(int(outputs["borrow_overlay_summary"].iloc[0]["excluded_short_positions"]), 0)

    def test_generate_near_open_execution_analysis_uses_strict_day_drops_and_ohlc4_entries(self) -> None:
        book = pd.DataFrame(
            [
                {
                    "signal_session": pd.Timestamp("2026-04-01 04:00:00", tz="UTC"),
                    "prediction_session": pd.Timestamp("2026-04-02 04:00:00", tz="UTC"),
                    "ticker": "AAA",
                    "side": "long",
                    "rank": 1,
                    "weight": 0.5,
                    "ranking_score_column": "score_z",
                    "ranking_score": 1.0,
                    "book_mode": "tail_fraction",
                    "tail_fraction": 0.25,
                    "tail_size": 1,
                    "basket_id": "growth",
                },
                {
                    "signal_session": pd.Timestamp("2026-04-01 04:00:00", tz="UTC"),
                    "prediction_session": pd.Timestamp("2026-04-02 04:00:00", tz="UTC"),
                    "ticker": "BBB",
                    "side": "short",
                    "rank": 1,
                    "weight": -0.5,
                    "ranking_score_column": "score_z",
                    "ranking_score": -1.0,
                    "book_mode": "tail_fraction",
                    "tail_fraction": 0.25,
                    "tail_size": 1,
                    "basket_id": "defensive_value",
                },
                {
                    "signal_session": pd.Timestamp("2026-04-02 04:00:00", tz="UTC"),
                    "prediction_session": pd.Timestamp("2026-04-03 04:00:00", tz="UTC"),
                    "ticker": "AAA",
                    "side": "long",
                    "rank": 1,
                    "weight": 0.5,
                    "ranking_score_column": "score_z",
                    "ranking_score": 1.0,
                    "book_mode": "tail_fraction",
                    "tail_fraction": 0.25,
                    "tail_size": 1,
                    "basket_id": "growth",
                },
                {
                    "signal_session": pd.Timestamp("2026-04-02 04:00:00", tz="UTC"),
                    "prediction_session": pd.Timestamp("2026-04-03 04:00:00", tz="UTC"),
                    "ticker": "BBB",
                    "side": "short",
                    "rank": 1,
                    "weight": -0.5,
                    "ranking_score_column": "score_z",
                    "ranking_score": -1.0,
                    "book_mode": "tail_fraction",
                    "tail_fraction": 0.25,
                    "tail_size": 1,
                    "basket_id": "defensive_value",
                },
            ]
        )

        def _rows_for_day(day: str, aaa_open: float, aaa_exit: float, bbb_open: float, bbb_exit: float, spy_open: float, spy_exit: float, include_0940: bool = True):
            tz = "America/New_York"
            rows = []
            for ticker, open_price, exit_price in [("AAA", aaa_open, aaa_exit), ("BBB", bbb_open, bbb_exit), ("SPY", spy_open, spy_exit)]:
                bars = [
                    ("09:30", open_price, open_price + 0.4, open_price - 0.2, open_price + 0.2, 100.0),
                    ("09:35", open_price + 0.1, open_price + 0.3, open_price - 0.1, open_price + 0.15, 120.0),
                ]
                if include_0940:
                    bars.append(("09:40", open_price + 0.15, open_price + 0.35, open_price + 0.05, open_price + 0.25, 140.0))
                bars.append(("15:55", exit_price - 0.1, exit_price + 0.1, exit_price - 0.2, exit_price, 200.0))
                for clock_time, open_px, high_px, low_px, close_px, volume in bars:
                    rows.append(
                        {
                            "ticker": ticker,
                            "timestamp": pd.Timestamp(f"{day} {clock_time}", tz=tz),
                            "open": open_px,
                            "high": high_px,
                            "low": low_px,
                            "close": close_px,
                            "volume": volume,
                        }
                    )
            return rows

        raw_execution = pd.DataFrame(
            _rows_for_day("2026-04-02", 100.0, 105.0, 100.0, 98.0, 100.0, 101.0, include_0940=True)
            + _rows_for_day("2026-04-03", 100.0, 104.0, 100.0, 99.0, 100.0, 101.0, include_0940=False)
        )

        outputs = generate_near_open_execution_analysis(
            book_frame=book,
            raw_execution_bars=raw_execution,
            benchmark_ticker="SPY",
            costs_bps=(10, 20, 40),
            execution_mode="strict",
        )

        comparison = outputs["execution_sensitivity_comparison"]
        quality = outputs["execution_data_quality"]

        self.assertEqual(set(comparison["execution_mode"].tolist()), {"exact_open", "first_5m_avg", "first_15m_avg"})
        self.assertTrue((comparison["prediction_days"] == 1).all())
        self.assertEqual(int(comparison["dropped_days"].iloc[0]), 1)
        self.assertTrue((comparison["execution_mode_label"] == STRICT_EXECUTION_MODE_LABEL).all())
        formulas = dict(zip(comparison["execution_mode"], comparison["entry_formula"]))
        self.assertEqual(formulas["exact_open"], EXACT_OPEN_ENTRY_FORMULA)
        self.assertEqual(formulas["first_5m_avg"], FIRST_5M_ENTRY_FORMULA)
        self.assertEqual(formulas["first_15m_avg"], FIRST_15M_ENTRY_FORMULA)
        self.assertFalse(quality.empty)
        self.assertTrue((quality["execution_mode_label"] == STRICT_EXECUTION_MODE_LABEL).all())
        self.assertIn("09:40", quality.iloc[0]["reason"])

    def test_build_backtest_scorecard_combines_baseline_execution_and_attribution_views(self) -> None:
        forecasts = self._make_forecasts()
        labeled_frame = pd.DataFrame(
            [
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.02, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.015, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "AAA", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.005, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": -0.01, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "BBB", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.005, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "CCC", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": -0.02, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "DDD", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.00, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 14:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-02", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-02 15:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-03 14:30", tz="UTC")},
                {"ticker": "SPY", "session_date": pd.Timestamp("2026-04-03", tz="UTC"), "raw_slot_return": 0.01, "timestamp": pd.Timestamp("2026-04-03 15:30", tz="UTC")},
            ]
        )
        adjusted = apply_unit_beta_to_forecasts(forecasts, labeled_frame)
        signal_outputs = generate_signal_execution_analysis(
            adjusted,
            labeled_frame=labeled_frame,
            n_long=1,
            n_short=1,
            tail_fractions=(0.25,),
            costs_bps=(10, 20, 40),
            book_mode="tail_fraction",
            tail_fraction=0.25,
        )
        execution_outputs = {
            "execution_sensitivity_comparison": pd.DataFrame(
                [
                    {"execution_mode": "exact_open", "net_sharpe_10bps": 2.0, "net_sharpe_20bps": 1.5, "net_sharpe_40bps": 0.5},
                    {"execution_mode": "first_5m_avg", "net_sharpe_10bps": 1.8, "net_sharpe_20bps": 1.2, "net_sharpe_40bps": 0.2},
                    {"execution_mode": "first_15m_avg", "net_sharpe_10bps": 1.4, "net_sharpe_20bps": 0.9, "net_sharpe_40bps": -0.1},
                ]
            )
        }

        scorecard = build_backtest_scorecard(
            forecasts=adjusted,
            signal_outputs=signal_outputs,
            execution_outputs=execution_outputs,
        )

        self.assertEqual(len(scorecard), 1)
        self.assertIn("book_hit_rate", scorecard.columns)
        self.assertIn("top_contributors", scorecard.columns)
        self.assertIn("oos_target_gap_20d", scorecard.columns)
        self.assertIn("borrow_overlay_net_sharpe_10bps", scorecard.columns)
        self.assertIn("long_only_net_sharpe_10bps", scorecard.columns)
        self.assertEqual(scorecard.iloc[0]["execution_mode_label"], STRICT_EXECUTION_MODE_LABEL)
        self.assertEqual(scorecard.iloc[0]["exact_open_entry_formula"], EXACT_OPEN_ENTRY_FORMULA)


if __name__ == "__main__":
    unittest.main()
