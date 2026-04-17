from __future__ import annotations

import unittest
from zoneinfo import ZoneInfo

import pandas as pd

from chronos_equity_prototype.tasks import US_MARKET_BUSINESS_DAY
from chronos_equity_prototype.v2_basket import BasketSpec, BasketV2Builder


def _make_normalized_rows(
    *,
    session_dates: list[pd.Timestamp],
    tickers: tuple[str, ...],
    missing: set[tuple[str, pd.Timestamp, int]] | None = None,
) -> pd.DataFrame:
    missing = missing or set()
    rows: list[dict[str, object]] = []
    for session_idx, session_date in enumerate(session_dates):
        for ticker_idx, ticker in enumerate(tickers):
            for slot_id in range(1, 8):
                if (ticker, session_date, slot_id) in missing:
                    continue
                base_price = 100 + (5 * ticker_idx) + session_idx
                rows.append(
                    {
                        "ticker": ticker,
                        "session_date": session_date,
                        "slot_id": slot_id,
                        "timestamp": session_date + pd.Timedelta(hours=slot_id),
                        "open": base_price + (slot_id * 0.1),
                        "high": base_price + (slot_id * 0.1) + 0.5,
                        "low": base_price + (slot_id * 0.1) - 0.5,
                        "close": base_price + (slot_id * 0.1) + (0.2 if ticker != "SPY" else 0.05),
                        "volume": 1_000 + (slot_id * 10) + ticker_idx,
                    }
                )
    return pd.DataFrame(rows)


class BasketV2BuilderTests(unittest.TestCase):
    def test_prepare_basket_frame_imputes_single_isolated_slot_and_logs_event(self) -> None:
        tz = ZoneInfo("America/New_York")
        session_date = pd.Timestamp("2026-04-10", tz=tz)
        normalized = _make_normalized_rows(
            session_dates=[session_date],
            tickers=("AAPL", "MSFT", "SPY"),
            missing={("AAPL", session_date, 4)},
        )
        builder = BasketV2Builder(benchmark_ticker="SPY")
        basket_spec = BasketSpec(basket_id="test_growth", tickers=("AAPL", "MSFT", "SPY"))

        prepared, events = builder.prepare_basket_frame(builder.build_base_frame(normalized), basket_spec)

        self.assertEqual(len(prepared), 21)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["event_type"], "single_slot_imputation")
        self.assertEqual(events.iloc[0]["ticker"], "AAPL")
        self.assertEqual(int(events.iloc[0]["slot_id"]), 4)

    def test_prepare_basket_frame_drops_basket_when_spy_is_incomplete(self) -> None:
        tz = ZoneInfo("America/New_York")
        session_date = pd.Timestamp("2026-04-10", tz=tz)
        normalized = _make_normalized_rows(
            session_dates=[session_date],
            tickers=("AAPL", "MSFT", "SPY"),
            missing={("SPY", session_date, 3)},
        )
        builder = BasketV2Builder(benchmark_ticker="SPY")
        basket_spec = BasketSpec(basket_id="test_growth", tickers=("AAPL", "MSFT", "SPY"))

        prepared, events = builder.prepare_basket_frame(builder.build_base_frame(normalized), basket_spec)
        summary = builder._build_quality_summary(events)

        self.assertTrue(prepared.empty)
        self.assertEqual(events.iloc[0]["event_type"], "drop")
        self.assertEqual(events.iloc[0]["ticker"], "SPY")
        self.assertEqual(
            float(summary.loc[summary["metric"] == "basket_dates_dropped", "value"].iloc[0]),
            1.0,
        )

    def test_select_baskets_supports_expanded_modes(self) -> None:
        builder = BasketV2Builder(benchmark_ticker="SPY")

        expanded40 = builder.select_baskets("expanded40")
        expanded60 = builder.select_baskets("expanded60")

        self.assertEqual(len(expanded40), 2)
        self.assertEqual(len(expanded60), 2)
        self.assertEqual(len(expanded40[0].target_tickers), 20)
        self.assertEqual(len(expanded40[1].target_tickers), 20)
        self.assertEqual(len(expanded60[0].target_tickers), 30)
        self.assertEqual(len(expanded60[1].target_tickers), 30)
        self.assertIn("AAPL", builder.required_tickers("expanded60"))
        self.assertIn("DE", builder.required_tickers("expanded60"))

    def test_flatten_predictions_logs_plain_risk_and_normalized_scores(self) -> None:
        tz = ZoneInfo("America/New_York")
        session_dates = [
            pd.Timestamp("2026-04-08", tz=tz),
            pd.Timestamp("2026-04-09", tz=tz),
            pd.Timestamp("2026-04-10", tz=tz),
        ]
        normalized = _make_normalized_rows(
            session_dates=session_dates,
            tickers=("AAPL", "MSFT", "SPY"),
        )
        builder = BasketV2Builder(benchmark_ticker="SPY")
        basket_spec = BasketSpec(basket_id="test_growth", tickers=("AAPL", "MSFT", "SPY"))
        prepared, _ = builder.prepare_basket_frame(builder.build_base_frame(normalized), basket_spec)
        wide = builder._build_wide_frame(prepared, basket_spec)
        history_tail = wide[wide["session_date"] <= session_dates[1]].tail(14).reset_index(drop=True)
        future = wide[wide["session_date"] == session_dates[2]].reset_index(drop=True)

        prediction_input = builder._build_basket_prediction_input(
            basket_spec=basket_spec,
            history_tail=history_tail,
            future=future,
            realized_future=future,
            signal_session=session_dates[1],
            prediction_session=session_dates[2],
            prediction_length=7,
            use_covariates=True,
            beta_mode="unit_beta_1.0",
        )
        prediction_df = pd.DataFrame(
            [
                {
                    "item_id": prediction_input.item_id,
                    "timestamp": prediction_input.future_df["timestamp"].iloc[horizon],
                    "target_name": target_name,
                    "predictions": 0.0,
                    "0.1": q10,
                    "0.5": q50,
                    "0.9": q90,
                }
                for target_name, q10, q50, q90 in [
                    ("AAPL", 0.01, 0.03, 0.05),
                    ("MSFT", -0.01, 0.00, 0.02),
                    ("SPY", 0.00, 0.01, 0.02),
                ]
                for horizon in range(7)
            ]
        )

        flattened = builder.flatten_predictions(
            [prediction_input],
            [prediction_df],
            model_id="amazon/chronos-2",
            context_length=14,
            cross_learning=False,
            use_covariates=True,
            beta_mode="unit_beta_1.0",
        )

        aapl_row = flattened[flattened["ticker"] == "AAPL"].iloc[0]
        msft_row = flattened[flattened["ticker"] == "MSFT"].iloc[0]

        self.assertAlmostEqual(aapl_row["beta_used"], 1.0)
        self.assertAlmostEqual(aapl_row["score_plain"], 0.14)
        self.assertAlmostEqual(msft_row["score_plain"], -0.07)
        self.assertAlmostEqual(aapl_row["score_z"], 1.0)
        self.assertAlmostEqual(msft_row["score_z"], -1.0)
        self.assertAlmostEqual(aapl_row["score_q50"], aapl_row["score_z"])
        self.assertIn("realized_alpha_path", flattened.columns)
        self.assertEqual(aapl_row["beta_mode"], "unit_beta_1.0")

    def test_build_prediction_inputs_latest_only_uses_common_latest_signal_session(self) -> None:
        tz = ZoneInfo("America/New_York")
        session_dates = [
            pd.Timestamp("2026-04-08", tz=tz),
            pd.Timestamp("2026-04-09", tz=tz),
            pd.Timestamp("2026-04-10", tz=tz),
        ]
        normalized = _make_normalized_rows(
            session_dates=session_dates,
            tickers=("AAA", "AAB", "BBB", "BBC", "SPY"),
            missing={("AAA", session_dates[2], 3), ("AAA", session_dates[2], 4)},
        )
        builder = BasketV2Builder(benchmark_ticker="SPY")
        builder.select_baskets = lambda basket_mode="full": (  # type: ignore[method-assign]
            BasketSpec(basket_id="growth", tickers=("AAA", "AAB", "SPY")),
            BasketSpec(basket_id="defensive_value", tickers=("BBB", "BBC", "SPY")),
        )

        prediction_inputs, _, _ = builder.build_prediction_inputs(
            builder.build_base_frame(normalized),
            context_length=7,
            prediction_length=7,
            basket_mode="full",
            latest_only=True,
            use_covariates=True,
            beta_mode="unit_beta_1.0",
        )

        self.assertEqual(len(prediction_inputs), 2)
        signal_sessions = {prediction_input.signal_session for prediction_input in prediction_inputs}
        prediction_sessions = {prediction_input.prediction_session for prediction_input in prediction_inputs}
        self.assertEqual(signal_sessions, {session_dates[1]})
        self.assertEqual(prediction_sessions, {session_dates[1] + US_MARKET_BUSINESS_DAY})


if __name__ == "__main__":
    unittest.main()
