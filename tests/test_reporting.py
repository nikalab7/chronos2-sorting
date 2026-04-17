from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from chronos_equity_prototype.reporting import export_chronos_package


class ReportingTests(unittest.TestCase):
    def test_export_chronos_package_writes_prediction_and_backtest_boards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_dir = root / "prediction"
            analysis_dir = root / "analysis"
            ledger_dir = root / "ledger"
            output_dir = root / "chronos_2026-04-17"
            prediction_dir.mkdir()
            analysis_dir.mkdir()
            ledger_dir.mkdir()

            forecasts = [
                {
                    "ticker": "AAA",
                    "basket_id": "growth",
                    "signal_session": "2026-04-16T04:00:00.000Z",
                    "prediction_session": "2026-04-17T04:00:00.000Z",
                    "stock_q10_path": [0.0] * 7,
                    "stock_q50_path": [0.01] * 7,
                    "stock_q90_path": [0.02] * 7,
                    "spy_q10_path": [0.0] * 7,
                    "spy_q50_path": [0.002] * 7,
                    "spy_q90_path": [0.004] * 7,
                    "alpha_q50_path": [0.008] * 7,
                    "beta_used": 1.0,
                    "score_plain": 0.056,
                    "score_risk": 1.0,
                    "score_z": 2.0,
                    "score_q50": 2.0,
                    "beta_mode": "unit_beta_1.0",
                },
                {
                    "ticker": "BBB",
                    "basket_id": "defensive_value",
                    "signal_session": "2026-04-15T04:00:00.000Z",
                    "prediction_session": "2026-04-16T04:00:00.000Z",
                    "stock_q10_path": [-0.02] * 7,
                    "stock_q50_path": [-0.01] * 7,
                    "stock_q90_path": [0.0] * 7,
                    "spy_q10_path": [0.0] * 7,
                    "spy_q50_path": [0.002] * 7,
                    "spy_q90_path": [0.004] * 7,
                    "alpha_q50_path": [-0.012] * 7,
                    "beta_used": 1.0,
                    "score_plain": -0.084,
                    "score_risk": -1.0,
                    "score_z": -2.0,
                    "score_q50": -2.0,
                    "beta_mode": "unit_beta_1.0",
                },
            ]
            with (prediction_dir / "forecasts.json").open("w", encoding="utf-8") as handle:
                json.dump(forecasts, handle, indent=2)
            pd.DataFrame(
                [
                    {
                        "signal_session": "2026-04-16T04:00:00.000Z",
                        "prediction_session": "2026-04-17T04:00:00.000Z",
                        "ticker": "AAA",
                        "side": "long",
                        "rank": 1,
                        "weight": 0.5,
                        "ranking_score_column": "score_z",
                        "ranking_score": 2.0,
                        "book_mode": "tail_fraction",
                        "tail_fraction": 0.1,
                        "tail_size": 1,
                    }
                ]
            ).to_csv(prediction_dir / "book.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "experiment_name": "exp",
                        "generated_at_utc": "2026-04-16T20:00:00.000Z",
                        "signal_session": "2026-04-16T04:00:00.000Z",
                        "prediction_session": "2026-04-17T04:00:00.000Z",
                        "model_id": "amazon/chronos-2",
                        "context_length": 224,
                        "basket_mode": "expanded40",
                        "beta_mode": "unit_beta_1.0",
                        "book_mode": "tail_fraction",
                        "tail_fraction": 0.1,
                        "tail_size": 1,
                        "score_column": "score_z",
                    }
                ]
            ).to_csv(prediction_dir / "prediction_summary.csv", index=False)

            pd.DataFrame([{"metric": "demo", "value": 1.0}]).to_csv(analysis_dir / "backtest_scorecard.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "signal_session": "2026-04-01T04:00:00.000Z",
                        "prediction_session": "2026-04-02T04:00:00.000Z",
                        "signal_date": "2026-04-01",
                        "prediction_date": "2026-04-02",
                        "long_basket_return": 0.01,
                        "short_basket_return": -0.01,
                        "long_short_spread": 0.02,
                        "gross_long_short_pnl": 0.01,
                        "book_hit": True,
                        "best_contributor": "AAA",
                        "best_contributor_pnl": 0.01,
                        "worst_contributor": "BBB",
                        "worst_contributor_pnl": -0.01,
                        "growth_result_gross_pnl": 0.01,
                        "defensive_result_gross_pnl": 0.0,
                    }
                ]
            ).to_csv(analysis_dir / "live_scoreboard_daily.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "signal_session": "2026-04-01T04:00:00.000Z",
                        "prediction_session": "2026-04-02T04:00:00.000Z",
                        "ticker": "AAA",
                        "basket_id": "growth",
                        "side": "long",
                        "rank": 1,
                        "score_z": 2.0,
                        "score_plain": 0.05,
                        "ranking_score_column": "score_z",
                        "ranking_score": 2.0,
                        "book_mode": "tail_fraction",
                        "tail_fraction": 0.1,
                        "tail_size": 1,
                        "beta_mode": "unit_beta_1.0",
                        "in_tail_book": True,
                        "next_day_realized_return": 0.03,
                        "next_day_realized_alpha": 0.02,
                        "weight": 0.5,
                        "gross_position_return": 0.01,
                        "book_helped": True,
                        "book_impact": "helped",
                    }
                ]
            ).to_csv(analysis_dir / "live_scoreboard_positions.csv", index=False)
            pd.DataFrame(
                [{"ticker": "AAA", "basket_id": "growth", "days_in_book": 1, "long_days": 1, "short_days": 0, "total_gross_pnl": 0.01, "mean_gross_pnl": 0.01, "mean_realized_return": 0.02, "mean_abs_score": 2.0, "mean_signed_realized_alpha": 0.02, "hit_rate": 1.0, "mean_spread_contribution": 0.01}]
            ).to_csv(analysis_dir / "ticker_attribution_summary.csv", index=False)
            pd.DataFrame(
                [
                    {"prediction_session": "2026-04-02T04:00:00.000Z", "rank_ic": 0.2, "metric": "", "value": ""},
                    {"prediction_session": "", "rank_ic": "", "metric": "mean_rank_ic", "value": 0.2},
                ]
            ).to_csv(analysis_dir / "signal_execution_baseline_rank_ic.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "signal_date": "2026-04-16",
                        "prediction_date": "2026-04-17",
                        "signal_session": "2026-04-16T04:00:00.000Z",
                        "prediction_session": "2026-04-17T04:00:00.000Z",
                        "ticker": "AAA",
                        "basket_id": "growth",
                        "side": "long",
                        "rank": 1,
                        "score_z": 2.0,
                        "score_plain": 0.056,
                        "ranking_score_column": "score_z",
                        "ranking_score": 2.0,
                        "book_mode": "tail_fraction",
                        "tail_fraction": 0.1,
                        "tail_size": 1,
                        "beta_mode": "unit_beta_1.0",
                        "in_tail_book": True,
                        "next_day_realized_return": 0.03,
                        "next_day_realized_alpha": 0.025,
                        "weight": 0.5,
                        "gross_position_return": 0.012,
                        "book_helped": True,
                        "book_impact": "helped",
                    }
                ]
            ).to_csv(ledger_dir / "live_scoreboard_positions.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "signal_date": "2026-04-16",
                        "prediction_date": "2026-04-17",
                        "signal_session": "2026-04-16T04:00:00.000Z",
                        "prediction_session": "2026-04-17T04:00:00.000Z",
                        "long_basket_return": 0.025,
                        "short_basket_return": -0.01,
                        "long_short_spread": 0.035,
                        "gross_long_short_pnl": 0.012,
                        "book_hit": True,
                        "best_contributor": "AAA",
                        "best_contributor_pnl": 0.012,
                        "worst_contributor": "AAA",
                        "worst_contributor_pnl": 0.012,
                        "growth_result_gross_pnl": 0.012,
                        "defensive_result_gross_pnl": 0.0,
                    }
                ]
            ).to_csv(ledger_dir / "live_scoreboard_daily.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "cost_bps": 10,
                        "return_type": "net",
                        "mean_daily_return": 0.01,
                        "daily_volatility": 0.02,
                        "annualized_sharpe": 1.5,
                        "win_rate": 1.0,
                    }
                ]
            ).to_csv(ledger_dir / "live_metrics_summary.csv", index=False)
            pd.DataFrame([{"execution_mode": "exact_open", "net_sharpe_10bps": 1.4}]).to_csv(
                ledger_dir / "live_execution_sensitivity_comparison.csv", index=False
            )
            pd.DataFrame([{"prediction_dir": "prediction", "ledger_dir": "ledger"}]).to_csv(
                ledger_dir / "reconciliation_summary.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "prediction_session": "2026-04-17T04:00:00.000Z",
                        "signal_session": "2026-04-16T04:00:00.000Z",
                        "generated_at_utc": "2026-04-16T20:00:00.000Z",
                    }
                ]
            ).to_csv(ledger_dir / "prediction_summary_history.csv", index=False)

            export_chronos_package(
                prediction_dir=prediction_dir,
                analysis_dir=analysis_dir,
                ledger_dir=ledger_dir,
                output_dir=output_dir,
                report_mode="live",
                report_context={"reconciliation_status": "completed"},
            )

            self.assertTrue((output_dir / "prediction_board.csv").exists())
            self.assertTrue((output_dir / "prediction_slot_board.csv").exists())
            self.assertTrue((output_dir / "prediction_summary_board.csv").exists())
            self.assertTrue((output_dir / "paper_summary.csv").exists())
            self.assertTrue((output_dir / "paper_report.html").exists())
            self.assertTrue((output_dir / "backtest_overview.csv").exists())
            self.assertTrue((output_dir / "backtest_daily_board.csv").exists())
            self.assertTrue((output_dir / "backtest_position_board.csv").exists())
            self.assertTrue((output_dir / "backtest_ticker_board.csv").exists())
            self.assertTrue((output_dir / "live_current_day_positions.csv").exists())
            self.assertTrue((output_dir / "live_current_day_daily.csv").exists())
            self.assertTrue((output_dir / "chart_manifest.csv").exists())
            self.assertTrue((output_dir / "README.txt").exists())

            prediction_board = pd.read_csv(output_dir / "prediction_board.csv")
            self.assertIn("confidence_score", prediction_board.columns)
            self.assertIn("prediction_rank_desc", prediction_board.columns)
            self.assertIn("in_book", prediction_board.columns)
            self.assertIn("selection_status", prediction_board.columns)

            paper_summary = pd.read_csv(output_dir / "paper_summary.csv")
            self.assertEqual(str(paper_summary.iloc[0]["report_mode"]), "live")


if __name__ == "__main__":
    unittest.main()
