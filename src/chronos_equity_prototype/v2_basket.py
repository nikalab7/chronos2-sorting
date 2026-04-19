from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .tasks import US_MARKET_BUSINESS_DAY

EPSILON = 1e-8
SLOT_TO_TIME = {
    1: "09:30",
    2: "10:30",
    3: "11:30",
    4: "12:30",
    5: "13:30",
    6: "14:30",
    7: "15:30",
}


@dataclass(frozen=True)
class BasketSpec:
    basket_id: str
    tickers: tuple[str, ...]

    @property
    def target_tickers(self) -> tuple[str, ...]:
        return tuple(ticker for ticker in self.tickers if ticker != "SPY")


@dataclass(frozen=True)
class BasketPredictionInput:
    basket_id: str
    item_id: str
    signal_session: pd.Timestamp
    prediction_session: pd.Timestamp
    model_target_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    history_df: pd.DataFrame
    future_df: pd.DataFrame | None
    beta_by_ticker: dict[str, float]
    realized_future_paths: dict[str, np.ndarray]
    prediction_session_end: pd.Timestamp | None = None
    signal_day_overnight_gap_pct_by_ticker: dict[str, float] = field(default_factory=dict)


DEFAULT_GROWTH_BASKET = BasketSpec(
    basket_id="growth",
    tickers=("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "CRM", "MA", "SPY"),
)
DEFAULT_DEFENSIVE_BASKET = BasketSpec(
    basket_id="defensive_value",
    tickers=("JPM", "XOM", "LLY", "ABBV", "MRK", "UNH", "COST", "WMT", "HD", "PG", "SPY"),
)
DEFAULT_V2_BASKETS = (DEFAULT_GROWTH_BASKET, DEFAULT_DEFENSIVE_BASKET)

EXPANDED40_GROWTH_BASKET = BasketSpec(
    basket_id="growth",
    tickers=(
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "AVGO",
        "CRM",
        "MA",
        "V",
        "NFLX",
        "AMD",
        "QCOM",
        "ORCL",
        "ADBE",
        "INTU",
        "AMAT",
        "TXN",
        "NOW",
        "BKNG",
        "SPY",
    ),
)
EXPANDED40_DEFENSIVE_BASKET = BasketSpec(
    basket_id="defensive_value",
    tickers=(
        "JPM",
        "XOM",
        "LLY",
        "ABBV",
        "MRK",
        "UNH",
        "COST",
        "WMT",
        "HD",
        "PG",
        "KO",
        "PEP",
        "BAC",
        "WFC",
        "CVX",
        "JNJ",
        "MCD",
        "AMGN",
        "LIN",
        "SPY",
    ),
)
EXPANDED40_V2_BASKETS = (EXPANDED40_GROWTH_BASKET, EXPANDED40_DEFENSIVE_BASKET)

EXPANDED60_GROWTH_BASKET = BasketSpec(
    basket_id="growth",
    tickers=(
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "AVGO",
        "CRM",
        "MA",
        "V",
        "NFLX",
        "AMD",
        "QCOM",
        "ORCL",
        "ADBE",
        "INTU",
        "AMAT",
        "TXN",
        "NOW",
        "BKNG",
        "ACN",
        "AXP",
        "GS",
        "MS",
        "BLK",
        "SCHW",
        "SPGI",
        "ADP",
        "CB",
        "PGR",
        "SPY",
    ),
)
EXPANDED60_DEFENSIVE_BASKET = BasketSpec(
    basket_id="defensive_value",
    tickers=(
        "JPM",
        "XOM",
        "LLY",
        "ABBV",
        "MRK",
        "UNH",
        "COST",
        "WMT",
        "HD",
        "PG",
        "KO",
        "PEP",
        "BAC",
        "WFC",
        "CVX",
        "JNJ",
        "MCD",
        "TMO",
        "AMGN",
        "LIN",
        "HON",
        "DHR",
        "COP",
        "NEE",
        "SBUX",
        "LOW",
        "PM",
        "RTX",
        "ISRG",
        "DE",
        "SPY",
    ),
)
EXPANDED60_V2_BASKETS = (EXPANDED60_GROWTH_BASKET, EXPANDED60_DEFENSIVE_BASKET)


def _safe_numeric_zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if std <= 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    mean = float(values.mean())
    return (values - mean) / std


@dataclass
class BasketV2Builder:
    benchmark_ticker: str = "SPY"
    realized_vol_window: int = 20
    beta_lookback_slots: int = 140
    beta_min: float = 0.0
    beta_max: float = 3.0
    synthetic_start: str = "2000-01-01 00:00:00"

    asset_past_covariates: tuple[str, ...] = (
        "log_volume",
        "high_low_range",
        "rolling_slot_vol_20",
        "overnight_gap",
        "prior_session_return",
    )
    basket_past_covariates: tuple[str, ...] = (
        "basket_mean_return_ex_spy",
        "basket_dispersion",
        "basket_positive_breadth",
        "basket_realized_vol",
    )
    known_future_covariates: tuple[str, ...] = ("slot_id", "weekday", "month_end_flag")

    def select_baskets(self, basket_mode: str = "full") -> tuple[BasketSpec, ...]:
        if basket_mode == "full":
            return DEFAULT_V2_BASKETS
        if basket_mode == "growth_spike":
            return (DEFAULT_GROWTH_BASKET,)
        if basket_mode == "expanded40":
            return EXPANDED40_V2_BASKETS
        if basket_mode == "expanded60":
            return EXPANDED60_V2_BASKETS
        raise ValueError(f"Unsupported basket mode: {basket_mode}")

    def required_tickers(self, basket_mode: str = "full") -> tuple[str, ...]:
        tickers = [
            ticker
            for basket_spec in self.select_baskets(basket_mode)
            for ticker in basket_spec.target_tickers
        ]
        return tuple(dict.fromkeys(tickers))

    def build_base_frame(self, normalized_frame: pd.DataFrame) -> pd.DataFrame:
        if normalized_frame.empty:
            return normalized_frame.copy()

        base = normalized_frame.copy().sort_values(["ticker", "session_date", "slot_id"]).reset_index(drop=True)
        base["raw_slot_return"] = np.log(base["close"] / base["open"])
        base["log_volume"] = np.log(base["volume"].clip(lower=1.0))
        base["high_low_range"] = (
            (base["high"] - base["low"]) / base["open"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        base["weekday"] = base["session_date"].dt.weekday.astype(float)
        base["month_end_flag"] = base["session_date"].dt.is_month_end.astype(float)

        session_summary = (
            base.groupby(["ticker", "session_date"], as_index=False)
            .agg(session_open=("open", "first"), session_close=("close", "last"))
            .sort_values(["ticker", "session_date"])
            .reset_index(drop=True)
        )
        session_summary["prior_session_return"] = (
            np.log(session_summary["session_close"] / session_summary["session_open"])
            .groupby(session_summary["ticker"])
            .shift(1)
        )
        previous_close = session_summary.groupby("ticker")["session_close"].shift(1)
        session_summary["overnight_gap"] = np.log(session_summary["session_open"] / previous_close)
        session_summary[["prior_session_return", "overnight_gap"]] = session_summary[
            ["prior_session_return", "overnight_gap"]
        ].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        base = base.merge(
            session_summary[["ticker", "session_date", "prior_session_return", "overnight_gap"]],
            on=["ticker", "session_date"],
            how="left",
            validate="many_to_one",
        )
        base["rolling_slot_vol_20"] = (
            base.groupby("ticker")["raw_slot_return"]
            .transform(lambda series: series.rolling(self.realized_vol_window, min_periods=5).std())
            .fillna(0.0)
        )
        return base.sort_values(["ticker", "session_date", "slot_id"]).reset_index(drop=True)

    def prepare_basket_frame(self, base_frame: pd.DataFrame, basket_spec: BasketSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
        basket_frame = base_frame[base_frame["ticker"].isin(basket_spec.tickers)].copy()
        if basket_frame.empty:
            return basket_frame, pd.DataFrame(columns=self._quality_event_columns())

        sessions = sorted(basket_frame["session_date"].drop_duplicates().tolist())
        kept_frames: list[pd.DataFrame] = []
        events: list[dict[str, object]] = []

        for session_date in sessions:
            session_rows = basket_frame[basket_frame["session_date"] == session_date].copy()
            session_grid = self._build_session_grid(session_rows, basket_spec, pd.Timestamp(session_date))
            drop_reasons = self._collect_drop_reasons(session_grid, basket_spec)

            if drop_reasons:
                for ticker, reason in drop_reasons.items():
                    events.append(
                        self._quality_event(
                            basket_id=basket_spec.basket_id,
                            prediction_session=pd.Timestamp(session_date),
                            ticker=ticker,
                            slot_id=None,
                            event_type="drop",
                            reason=reason,
                            action_taken="drop_basket_date",
                        )
                    )
                continue

            imputed_grid, imputation_events = self._impute_single_slot_gaps(session_grid, basket_spec, pd.Timestamp(session_date))
            kept_frames.append(imputed_grid)
            events.extend(imputation_events)

        if not kept_frames:
            empty_columns = ["ticker", "session_date", "slot_id", "timestamp", "open", "high", "low", "close", "volume"]
            return pd.DataFrame(columns=empty_columns), pd.DataFrame.from_records(events, columns=self._quality_event_columns())

        cleaned = pd.concat(kept_frames, ignore_index=True)
        cleaned = cleaned[["ticker", "session_date", "slot_id", "timestamp", "open", "high", "low", "close", "volume"]]
        prepared = self._recompute_features(cleaned, basket_spec)
        event_frame = pd.DataFrame.from_records(events, columns=self._quality_event_columns())
        return prepared, event_frame

    def _build_session_grid(
        self,
        session_rows: pd.DataFrame,
        basket_spec: BasketSpec,
        session_date: pd.Timestamp,
    ) -> pd.DataFrame:
        expected = pd.MultiIndex.from_product(
            [basket_spec.tickers, range(1, 8)],
            names=["ticker", "slot_id"],
        ).to_frame(index=False)
        expected["session_date"] = session_date
        merged = expected.merge(
            session_rows,
            on=["ticker", "session_date", "slot_id"],
            how="left",
            suffixes=("", "_raw"),
        )
        merged["timestamp"] = merged["timestamp"].where(
            merged["timestamp"].notna(),
            merged["slot_id"].map(lambda slot_id: self._slot_timestamp(session_date, int(slot_id))),
        )
        return merged.sort_values(["ticker", "slot_id"]).reset_index(drop=True)

    def _collect_drop_reasons(self, session_grid: pd.DataFrame, basket_spec: BasketSpec) -> dict[str, str]:
        reasons: dict[str, str] = {}
        for ticker, ticker_frame in session_grid.groupby("ticker", sort=True):
            missing_slots = ticker_frame[ticker_frame["open"].isna()]["slot_id"].tolist()
            missing_count = len(missing_slots)
            if missing_count == 0:
                continue
            if ticker == self.benchmark_ticker:
                reasons[ticker] = "spy_incomplete_session"
                continue
            if missing_count == 7:
                reasons[ticker] = "stock_missing_full_session"
                continue
            if missing_count > 1:
                reasons[ticker] = "stock_missing_multiple_slots"
                continue
            slot_id = int(missing_slots[0])
            if slot_id in (1, 7):
                reasons[ticker] = "single_missing_slot_not_isolated"
                continue
            previous_row = ticker_frame[ticker_frame["slot_id"] == slot_id - 1]
            next_row = ticker_frame[ticker_frame["slot_id"] == slot_id + 1]
            if previous_row.empty or next_row.empty or previous_row["open"].isna().any() or next_row["open"].isna().any():
                reasons[ticker] = "single_missing_slot_not_isolated"
        return reasons

    def _impute_single_slot_gaps(
        self,
        session_grid: pd.DataFrame,
        basket_spec: BasketSpec,
        session_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        frame = session_grid.copy()
        events: list[dict[str, object]] = []
        numeric_columns = ["open", "high", "low", "close", "volume"]

        for ticker in basket_spec.target_tickers:
            ticker_mask = frame["ticker"] == ticker
            ticker_frame = frame.loc[ticker_mask].sort_values("slot_id")
            missing_rows = ticker_frame[ticker_frame["open"].isna()]
            if len(missing_rows) != 1:
                continue

            missing_index = missing_rows.index[0]
            slot_id = int(missing_rows.iloc[0]["slot_id"])
            for column in numeric_columns:
                previous_value = float(ticker_frame.loc[ticker_frame["slot_id"] == slot_id - 1, column].iloc[0])
                next_value = float(ticker_frame.loc[ticker_frame["slot_id"] == slot_id + 1, column].iloc[0])
                frame.at[missing_index, column] = (previous_value + next_value) / 2.0

            frame.at[missing_index, "timestamp"] = self._slot_timestamp(session_date, slot_id)
            events.append(
                self._quality_event(
                    basket_id=basket_spec.basket_id,
                    prediction_session=session_date,
                    ticker=ticker,
                    slot_id=slot_id,
                    event_type="single_slot_imputation",
                    reason="isolated_single_slot_gap",
                    action_taken="linear_midpoint_imputation",
                )
            )

        return frame.sort_values(["ticker", "slot_id"]).reset_index(drop=True), events

    def _recompute_features(self, cleaned_frame: pd.DataFrame, basket_spec: BasketSpec) -> pd.DataFrame:
        frame = self.build_base_frame(cleaned_frame)
        non_spy = frame[frame["ticker"] != self.benchmark_ticker].copy()
        basket_features = (
            non_spy.groupby(["session_date", "slot_id"], as_index=False)
            .agg(
                basket_mean_return_ex_spy=("raw_slot_return", "mean"),
                basket_dispersion=("raw_slot_return", lambda series: float(series.std(ddof=0))),
                basket_positive_breadth=("raw_slot_return", lambda series: float((series > 0).mean())),
            )
            .sort_values(["session_date", "slot_id"])
            .reset_index(drop=True)
        )
        basket_features["basket_dispersion"] = basket_features["basket_dispersion"].fillna(0.0)
        basket_features["basket_realized_vol"] = (
            basket_features["basket_mean_return_ex_spy"].rolling(self.realized_vol_window, min_periods=5).std().fillna(0.0)
        )
        basket_features["basket_id"] = basket_spec.basket_id

        frame = frame.merge(
            basket_features,
            on=["session_date", "slot_id"],
            how="left",
            validate="many_to_one",
        )
        frame["basket_id"] = basket_spec.basket_id
        return frame.sort_values(["session_date", "slot_id", "ticker"]).reset_index(drop=True)

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
    ) -> tuple[list[BasketPredictionInput], pd.DataFrame, pd.DataFrame]:
        prediction_inputs: list[BasketPredictionInput] = []
        all_events: list[pd.DataFrame] = []
        prepared_baskets: list[tuple[BasketSpec, pd.DataFrame, pd.DataFrame]] = []
        horizon_sessions = self._horizon_sessions(prediction_length)

        for basket_spec in self.select_baskets(basket_mode):
            prepared_frame, event_frame = self.prepare_basket_frame(base_frame, basket_spec)
            all_events.append(event_frame)
            if prepared_frame.empty:
                continue
            prepared_baskets.append((basket_spec, prepared_frame, event_frame))

        if latest_only:
            latest_signal_session = self._resolve_latest_common_signal_session(
                prepared_baskets,
                context_length=context_length,
            )
            if latest_signal_session is None:
                non_empty_events = [frame for frame in all_events if not frame.empty]
                events = (
                    pd.concat(non_empty_events, ignore_index=True)
                    if non_empty_events
                    else pd.DataFrame(columns=self._quality_event_columns())
                )
                summary = self._build_quality_summary(events)
                return [], events, summary

        for basket_spec, prepared_frame, _ in prepared_baskets:
            wide_frame = self._build_wide_frame(prepared_frame, basket_spec)
            session_dates = [pd.Timestamp(value) for value in wide_frame["session_date"].drop_duplicates().tolist()]
            if latest_only:
                if latest_signal_session not in set(session_dates):
                    continue
                future_sessions = self._future_sessions_from_start(
                    pd.Timestamp(latest_signal_session) + US_MARKET_BUSINESS_DAY,
                    horizon_sessions,
                )
                session_pairs = [(pd.Timestamp(latest_signal_session), future_sessions)]
            else:
                session_pairs = []
                for index, signal_session in enumerate(session_dates):
                    future_sessions = session_dates[index + 1 : index + 1 + horizon_sessions]
                    if len(future_sessions) != horizon_sessions:
                        continue
                    session_pairs.append((pd.Timestamp(signal_session), [pd.Timestamp(value) for value in future_sessions]))

            for signal_session, future_sessions in session_pairs:
                history = wide_frame[wide_frame["session_date"] <= signal_session].copy()
                prediction_session = pd.Timestamp(future_sessions[0])
                prediction_session_end = pd.Timestamp(future_sessions[-1])
                if latest_only:
                    future = self._build_future_known_covariates(future_sessions, prediction_length)
                    realized_future = None
                else:
                    future = wide_frame[wide_frame["session_date"].isin(future_sessions)].copy()
                    realized_future = future.copy()

                if len(history) < context_length or len(future) != prediction_length:
                    continue
                prediction_inputs.append(
                    self._build_basket_prediction_input(
                        basket_spec=basket_spec,
                        history_tail=history.tail(context_length).reset_index(drop=True),
                        future=future.reset_index(drop=True),
                        realized_future=realized_future.reset_index(drop=True) if realized_future is not None else None,
                        signal_session=pd.Timestamp(signal_session),
                        prediction_session=pd.Timestamp(prediction_session),
                        prediction_session_end=prediction_session_end,
                        prediction_length=prediction_length,
                        use_covariates=use_covariates,
                        beta_mode=beta_mode,
                    )
                )

        non_empty_events = [frame for frame in all_events if not frame.empty]
        events = (
            pd.concat(non_empty_events, ignore_index=True)
            if non_empty_events
            else pd.DataFrame(columns=self._quality_event_columns())
        )
        summary = self._build_quality_summary(events)
        return prediction_inputs, events, summary

    def flatten_predictions(
        self,
        prediction_inputs: list[BasketPredictionInput],
        prediction_frames: list[pd.DataFrame],
        *,
        model_id: str,
        context_length: int,
        cross_learning: bool,
        use_covariates: bool,
        beta_mode: str,
        product_mode: str = "ranker",
        confirmation_mode: str = "1d_plus_3d",
    ) -> pd.DataFrame:
        if not prediction_inputs or not prediction_frames:
            return pd.DataFrame()

        predictions = pd.concat(prediction_frames, ignore_index=True)
        input_lookup = {prediction_input.item_id: prediction_input for prediction_input in prediction_inputs}
        records: list[dict[str, object]] = []

        for item_id, input_prediction in predictions.groupby("item_id", sort=True):
            if item_id not in input_lookup:
                continue
            prediction_input = input_lookup[item_id]
            path_lookup = self._build_path_lookup(input_prediction)
            spy_paths = path_lookup[self.benchmark_ticker]

            for ticker in prediction_input.target_columns:
                stock_paths = path_lookup[ticker]
                beta_used = float(prediction_input.beta_by_ticker[ticker])
                stock_q05 = self._get_quantile_path(stock_paths, 0.05, fallback=0.1)
                stock_q10 = self._get_quantile_path(stock_paths, 0.1)
                stock_q50 = self._get_quantile_path(stock_paths, 0.5)
                stock_q90 = self._get_quantile_path(stock_paths, 0.9)
                stock_q95 = self._get_quantile_path(stock_paths, 0.95, fallback=0.9)
                spy_q05 = self._get_quantile_path(spy_paths, 0.05, fallback=0.1)
                spy_q10 = self._get_quantile_path(spy_paths, 0.1)
                spy_q50 = self._get_quantile_path(spy_paths, 0.5)
                spy_q90 = self._get_quantile_path(spy_paths, 0.9)
                spy_q95 = self._get_quantile_path(spy_paths, 0.95, fallback=0.9)

                alpha_q50_path = (stock_q50 - (beta_used * spy_q50)).tolist()
                alpha_interval_width = (stock_q90 - stock_q10) + (abs(beta_used) * (spy_q90 - spy_q10))
                alpha_uncertainty = float(np.sum(alpha_interval_width))
                score_plain = float(np.sum(alpha_q50_path))
                score_risk = float(score_plain / (EPSILON + alpha_uncertainty))

                realized_alpha_path = None
                realized_day_return = float("nan")
                realized_stock_ret_1d = float("nan")
                realized_spy_ret_1d = float("nan")
                realized_alpha_1d = float("nan")
                realized_stock_ret_3d = float("nan")
                realized_spy_ret_3d = float("nan")
                realized_alpha_3d = float("nan")
                if prediction_input.realized_future_paths:
                    stock_realized = prediction_input.realized_future_paths[ticker]
                    spy_realized = prediction_input.realized_future_paths[self.benchmark_ticker]
                    realized_alpha_path = (stock_realized - (beta_used * spy_realized)).tolist()
                    realized_stock_ret_1d = self._convert_log_path_to_return(stock_realized[:7])
                    realized_spy_ret_1d = self._convert_log_path_to_return(spy_realized[:7])
                    realized_alpha_1d = float(realized_stock_ret_1d - (beta_used * realized_spy_ret_1d))
                    realized_stock_ret_3d = self._convert_log_path_to_return(stock_realized)
                    realized_spy_ret_3d = self._convert_log_path_to_return(spy_realized)
                    realized_alpha_3d = float(realized_stock_ret_3d - (beta_used * realized_spy_ret_3d))
                    realized_day_return = realized_alpha_1d

                records.append(
                    {
                        "ticker": ticker,
                        "basket_id": prediction_input.basket_id,
                        "signal_session": prediction_input.signal_session,
                        "prediction_session": prediction_input.prediction_session,
                        "prediction_session_end": prediction_input.prediction_session_end or prediction_input.prediction_session,
                        "stock_q05_path": stock_q05.tolist(),
                        "stock_q10_path": stock_q10.tolist(),
                        "stock_q50_path": stock_q50.tolist(),
                        "stock_q90_path": stock_q90.tolist(),
                        "stock_q95_path": stock_q95.tolist(),
                        "spy_q05_path": spy_q05.tolist(),
                        "spy_q10_path": spy_q10.tolist(),
                        "spy_q50_path": spy_q50.tolist(),
                        "spy_q90_path": spy_q90.tolist(),
                        "spy_q95_path": spy_q95.tolist(),
                        "alpha_q50_path": alpha_q50_path,
                        "beta_used": beta_used,
                        "score_plain": score_plain,
                        "score_risk": score_risk,
                        "score_z": 0.0,
                        "realized_alpha_path": realized_alpha_path,
                        "realized_day_return": realized_day_return,
                        "realized_stock_ret_1d": realized_stock_ret_1d,
                        "realized_spy_ret_1d": realized_spy_ret_1d,
                        "realized_alpha_1d": realized_alpha_1d,
                        "realized_stock_ret_3d": realized_stock_ret_3d,
                        "realized_spy_ret_3d": realized_spy_ret_3d,
                        "realized_alpha_3d": realized_alpha_3d,
                        "model_id": model_id,
                        "context_length": context_length,
                        "cross_learning": cross_learning,
                        "use_covariates": use_covariates,
                        "beta_mode": beta_mode,
                        "product_mode": product_mode,
                        "signal_day_overnight_gap_pct": float(
                            prediction_input.signal_day_overnight_gap_pct_by_ticker.get(ticker, np.nan)
                        ),
                    }
                )

        flattened = pd.DataFrame.from_records(records)
        if flattened.empty:
            return flattened

        flattened["score_z"] = (
            flattened.groupby(["basket_id", "prediction_session"], group_keys=False)["score_plain"].transform(_safe_numeric_zscore)
        )
        if product_mode == "winner_picker":
            flattened = self._enrich_winner_frame(flattened, confirmation_mode=confirmation_mode)
        else:
            flattened["score_q50"] = flattened["score_z"]
        return flattened.sort_values(["signal_session", "basket_id", "ticker"]).reset_index(drop=True)

    def _build_basket_prediction_input(
        self,
        *,
        basket_spec: BasketSpec,
        history_tail: pd.DataFrame,
        future: pd.DataFrame,
        realized_future: pd.DataFrame | None,
        signal_session: pd.Timestamp,
        prediction_session: pd.Timestamp,
        prediction_session_end: pd.Timestamp | None = None,
        prediction_length: int,
        use_covariates: bool,
        beta_mode: str,
    ) -> BasketPredictionInput:
        item_id = f"{basket_spec.basket_id}|{signal_session.date()}"
        synthetic_timestamps = pd.date_range(
            start=pd.Timestamp(self.synthetic_start),
            periods=len(history_tail) + prediction_length,
            freq="h",
        )

        history_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": synthetic_timestamps[: len(history_tail)],
            }
        )
        future_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": synthetic_timestamps[len(history_tail) :],
            }
        )
        target_columns = tuple(basket_spec.tickers)

        for target_column in target_columns:
            history_df[target_column] = history_tail[target_column].to_numpy(dtype=float)

        if use_covariates:
            for ticker in basket_spec.tickers:
                for feature_name in self.asset_past_covariates:
                    column_name = f"{ticker}__{feature_name}"
                    history_df[column_name] = history_tail[column_name].to_numpy(dtype=float)

            for feature_name in self.basket_past_covariates + self.known_future_covariates:
                history_df[feature_name] = history_tail[feature_name].to_numpy(dtype=float)
            for feature_name in self.known_future_covariates:
                future_df[feature_name] = future[feature_name].to_numpy(dtype=float)
        else:
            future_df = None

        realized_future_paths = (
            {ticker: realized_future[ticker].to_numpy(dtype=float) for ticker in basket_spec.tickers}
            if realized_future is not None
            else {}
        )
        beta_by_ticker = self._compute_betas(history_tail, basket_spec, beta_mode=beta_mode)
        signal_day_overnight_gap_pct_by_ticker = self._compute_signal_day_overnight_gap_pct(
            history_tail,
            basket_spec=basket_spec,
            signal_session=signal_session,
        )

        return BasketPredictionInput(
            basket_id=basket_spec.basket_id,
            item_id=item_id,
            signal_session=signal_session,
            prediction_session=prediction_session,
            prediction_session_end=prediction_session_end,
            model_target_columns=target_columns,
            target_columns=basket_spec.target_tickers,
            history_df=history_df,
            future_df=future_df,
            beta_by_ticker=beta_by_ticker,
            realized_future_paths=realized_future_paths,
            signal_day_overnight_gap_pct_by_ticker=signal_day_overnight_gap_pct_by_ticker,
        )

    def _build_wide_frame(self, prepared_frame: pd.DataFrame, basket_spec: BasketSpec) -> pd.DataFrame:
        target_wide = (
            prepared_frame.pivot_table(
                index=["session_date", "slot_id"],
                columns="ticker",
                values="raw_slot_return",
                aggfunc="last",
            )
            .reset_index()
            .sort_values(["session_date", "slot_id"])
            .reset_index(drop=True)
        )
        if isinstance(target_wide.columns, pd.MultiIndex):
            target_wide.columns = [
                str(column[0]) if column[0] in {"session_date", "slot_id"} else str(column[-1] or column[0])
                for column in target_wide.columns.to_list()
            ]

        wide = target_wide.copy()
        for ticker in basket_spec.tickers:
            ticker_rows = prepared_frame[prepared_frame["ticker"] == ticker].sort_values(["session_date", "slot_id"])
            ticker_features = ticker_rows[["session_date", "slot_id"]].copy()
            for feature_name in self.asset_past_covariates:
                ticker_features[f"{ticker}__{feature_name}"] = ticker_rows[feature_name].to_numpy(dtype=float)
            wide = wide.merge(ticker_features, on=["session_date", "slot_id"], how="left", validate="one_to_one")

        shared_columns = list(dict.fromkeys(["session_date", "slot_id", *self.basket_past_covariates, *self.known_future_covariates]))
        shared = (
            prepared_frame[shared_columns]
            .drop_duplicates(subset=["session_date", "slot_id"])
            .sort_values(["session_date", "slot_id"])
        )
        wide = wide.merge(shared, on=["session_date", "slot_id"], how="left", validate="one_to_one")
        return wide.sort_values(["session_date", "slot_id"]).reset_index(drop=True)

    def _build_path_lookup(self, prediction_frame: pd.DataFrame) -> dict[str, dict[float, np.ndarray]]:
        lookup: dict[str, dict[float, np.ndarray]] = {}
        for target_name, target_frame in prediction_frame.groupby("target_name", sort=True):
            ordered = target_frame.sort_values("timestamp").reset_index(drop=True)
            paths: dict[float, np.ndarray] = {}
            for column in ordered.columns:
                try:
                    quantile = float(column)
                except (TypeError, ValueError):
                    continue
                paths[quantile] = ordered[column].to_numpy(dtype=float)
            if 0.5 not in paths:
                raise ValueError(f"Missing q50 path for {target_name!r} in Chronos prediction output.")
            lookup[str(target_name)] = paths
        return lookup

    def _compute_pooled_betas(self, history_tail: pd.DataFrame, basket_spec: BasketSpec) -> dict[str, float]:
        spy_history = history_tail[self.benchmark_ticker].to_numpy(dtype=float)
        betas: dict[str, float] = {}
        for ticker in basket_spec.target_tickers:
            asset_history = history_tail[ticker].to_numpy(dtype=float)
            if len(asset_history) < self.beta_lookback_slots or len(spy_history) < self.beta_lookback_slots:
                betas[ticker] = 1.0
                continue

            asset_tail = asset_history[-self.beta_lookback_slots :]
            spy_tail = spy_history[-self.beta_lookback_slots :]
            variance = float(np.var(spy_tail, ddof=0))
            if variance <= EPSILON or np.isnan(variance):
                betas[ticker] = 1.0
                continue

            covariance = float(np.cov(asset_tail, spy_tail, ddof=0)[0, 1])
            beta = covariance / variance
            betas[ticker] = float(np.clip(beta, self.beta_min, self.beta_max))
        return betas

    def _compute_betas(
        self,
        history_tail: pd.DataFrame,
        basket_spec: BasketSpec,
        *,
        beta_mode: str,
    ) -> dict[str, float]:
        if beta_mode == "unit_beta_1.0":
            return {ticker: 1.0 for ticker in basket_spec.target_tickers}
        if beta_mode == "pooled_beta_140":
            return self._compute_pooled_betas(history_tail, basket_spec)
        if beta_mode == "rolling_session_beta_20d":
            return self._compute_rolling_session_betas(history_tail, basket_spec)
        raise ValueError(f"Unsupported beta_mode: {beta_mode!r}")

    def _resolve_latest_common_signal_session(
        self,
        prepared_baskets: list[tuple[BasketSpec, pd.DataFrame, pd.DataFrame]],
        *,
        context_length: int,
    ) -> pd.Timestamp | None:
        if not prepared_baskets:
            return None

        candidate_sets: list[set[pd.Timestamp]] = []
        for basket_spec, prepared_frame, _ in prepared_baskets:
            wide_frame = self._build_wide_frame(prepared_frame, basket_spec)
            session_dates = [pd.Timestamp(value) for value in wide_frame["session_date"].drop_duplicates().tolist()]
            if not session_dates:
                return None

            valid_sessions: set[pd.Timestamp] = set()
            for signal_session in session_dates:
                history = wide_frame[wide_frame["session_date"] <= signal_session]
                if len(history) >= context_length:
                    valid_sessions.add(signal_session)
            if not valid_sessions:
                return None
            candidate_sets.append(valid_sessions)

        common_sessions = set.intersection(*candidate_sets) if candidate_sets else set()
        if not common_sessions:
            return None
        return max(common_sessions)

    @staticmethod
    def _quality_event_columns() -> list[str]:
        return [
            "basket_id",
            "prediction_session",
            "ticker",
            "slot_id",
            "event_type",
            "reason",
            "action_taken",
        ]

    def _quality_event(
        self,
        *,
        basket_id: str,
        prediction_session: pd.Timestamp,
        ticker: str,
        slot_id: int | None,
        event_type: str,
        reason: str,
        action_taken: str,
    ) -> dict[str, object]:
        return {
            "basket_id": basket_id,
            "prediction_session": prediction_session,
            "ticker": ticker,
            "slot_id": slot_id,
            "event_type": event_type,
            "reason": reason,
            "action_taken": action_taken,
        }

    def _build_quality_summary(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return pd.DataFrame(
                [
                    {"metric": "basket_dates_dropped", "value": 0.0, "ticker": None, "drop_count": np.nan},
                    {"metric": "single_slot_imputations_used", "value": 0.0, "ticker": None, "drop_count": np.nan},
                ]
            )

        dropped_dates = events[events["event_type"] == "drop"][["basket_id", "prediction_session"]].drop_duplicates()
        ticker_drop_counts = (
            events[events["event_type"] == "drop"]
            .groupby("ticker")
            .size()
            .rename("drop_count")
            .reset_index()
            .sort_values("ticker")
            .reset_index(drop=True)
        )

        summary_records = [
            {"metric": "basket_dates_dropped", "value": float(len(dropped_dates)), "ticker": None, "drop_count": np.nan},
            {
                "metric": "single_slot_imputations_used",
                "value": float((events["event_type"] == "single_slot_imputation").sum()),
                "ticker": None,
                "drop_count": np.nan,
            },
        ]
        for row in ticker_drop_counts.itertuples(index=False):
            summary_records.append(
                {
                    "metric": "ticker_drop_count",
                    "value": np.nan,
                    "ticker": row.ticker,
                    "drop_count": float(row.drop_count),
                }
            )
        return pd.DataFrame.from_records(summary_records)

    def _build_future_known_covariates(
        self,
        future_sessions: list[pd.Timestamp],
        prediction_length: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for session_date in future_sessions:
            session_ts = pd.Timestamp(session_date)
            for slot_id in range(1, 8):
                rows.append(
                    {
                        "session_date": session_ts,
                        "slot_id": slot_id,
                        "weekday": float(session_ts.weekday()),
                        "month_end_flag": float(session_ts.is_month_end),
                    }
                )
        future = pd.DataFrame.from_records(rows).sort_values(["session_date", "slot_id"]).reset_index(drop=True)
        return future.head(prediction_length)

    @staticmethod
    def _horizon_sessions(prediction_length: int) -> int:
        if int(prediction_length) <= 0 or int(prediction_length) % 7 != 0:
            raise ValueError("V2 prediction_length must be a positive multiple of 7 regular-hours slots.")
        return int(prediction_length) // 7

    def _future_sessions_from_start(
        self,
        prediction_session_start: pd.Timestamp,
        horizon_sessions: int,
    ) -> list[pd.Timestamp]:
        return [
            pd.Timestamp(prediction_session_start) + (step * US_MARKET_BUSINESS_DAY)
            for step in range(horizon_sessions)
        ]

    def _compute_signal_day_overnight_gap_pct(
        self,
        history_tail: pd.DataFrame,
        *,
        basket_spec: BasketSpec,
        signal_session: pd.Timestamp,
    ) -> dict[str, float]:
        gap_by_ticker: dict[str, float] = {}
        signal_rows = history_tail[history_tail["session_date"] == pd.Timestamp(signal_session)].copy()
        for ticker in basket_spec.target_tickers:
            column_name = f"{ticker}__overnight_gap"
            if column_name not in signal_rows.columns or signal_rows.empty:
                gap_by_ticker[ticker] = float("nan")
                continue
            slot_one = signal_rows[signal_rows["slot_id"] == 1]
            source = slot_one if not slot_one.empty else signal_rows
            gap_log = float(source.iloc[0][column_name])
            if np.isnan(gap_log):
                gap_by_ticker[ticker] = float("nan")
                continue
            gap_by_ticker[ticker] = float((np.exp(gap_log) - 1.0) * 100.0)
        return gap_by_ticker

    @staticmethod
    def _get_quantile_path(
        paths: dict[float, np.ndarray],
        level: float,
        *,
        fallback: float | None = None,
    ) -> np.ndarray:
        if level in paths:
            return paths[level]
        if fallback is not None and fallback in paths:
            return paths[fallback]
        raise ValueError(f"Missing quantile path for level {level}.")

    @staticmethod
    def _convert_log_path_to_return(values: np.ndarray) -> float:
        if len(values) == 0:
            return float("nan")
        return float(np.exp(np.sum(values)) - 1.0)

    @staticmethod
    def _rss(values: np.ndarray) -> float:
        if len(values) == 0:
            return float("nan")
        cleaned = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
        return float(np.sqrt(np.sum(np.square(cleaned))))

    def _compute_rolling_session_betas(
        self,
        history_tail: pd.DataFrame,
        basket_spec: BasketSpec,
    ) -> dict[str, float]:
        session_wide = history_tail.groupby("session_date", as_index=False)[basket_spec.tickers].sum()
        session_wide = session_wide.sort_values("session_date").reset_index(drop=True)
        spy_series = session_wide[self.benchmark_ticker].astype(float)
        betas: dict[str, float] = {}
        for ticker in basket_spec.target_tickers:
            pair = pd.DataFrame(
                {
                    "asset": session_wide[ticker].astype(float),
                    "spy": spy_series,
                }
            ).dropna()
            pair = pair.tail(20)
            if len(pair) < 10:
                betas[ticker] = 1.0
                continue
            variance = float(np.var(pair["spy"], ddof=0))
            if variance <= EPSILON or np.isnan(variance):
                betas[ticker] = 1.0
                continue
            covariance = float(np.cov(pair["asset"], pair["spy"], ddof=0)[0, 1])
            beta = covariance / variance
            betas[ticker] = float(np.clip(beta, self.beta_min, self.beta_max))
        return betas

    def _enrich_winner_frame(
        self,
        frame: pd.DataFrame,
        *,
        confirmation_mode: str,
    ) -> pd.DataFrame:
        enriched = frame.copy()
        enriched["pred_log_ret_1d_stock"] = enriched["stock_q50_path"].apply(lambda values: float(np.sum(values[:7])))
        enriched["pred_log_ret_1d_spy"] = enriched["spy_q50_path"].apply(lambda values: float(np.sum(values[:7])))
        enriched["pred_log_ret_3d_stock"] = enriched["stock_q50_path"].apply(lambda values: float(np.sum(values)))
        enriched["pred_log_ret_3d_spy"] = enriched["spy_q50_path"].apply(lambda values: float(np.sum(values)))
        enriched["pred_ret_1d"] = np.exp(enriched["pred_log_ret_1d_stock"]) - 1.0
        enriched["pred_spy_ret_1d"] = np.exp(enriched["pred_log_ret_1d_spy"]) - 1.0
        enriched["pred_ret_3d"] = np.exp(enriched["pred_log_ret_3d_stock"]) - 1.0
        enriched["pred_spy_ret_3d"] = np.exp(enriched["pred_log_ret_3d_spy"]) - 1.0
        enriched["pred_alpha_1d"] = enriched["pred_ret_1d"] - (enriched["beta_used"] * enriched["pred_spy_ret_1d"])
        enriched["pred_alpha_3d"] = enriched["pred_ret_3d"] - (enriched["beta_used"] * enriched["pred_spy_ret_3d"])
        enriched["score_plain"] = enriched["pred_alpha_1d"].astype(float)

        enriched["uncertainty_proxy_1d"] = enriched.apply(
            lambda row: self._rss(
                (np.array(row["stock_q90_path"][:7], dtype=float) - np.array(row["stock_q10_path"][:7], dtype=float))
                + (abs(float(row["beta_used"])) * (np.array(row["spy_q90_path"][:7], dtype=float) - np.array(row["spy_q10_path"][:7], dtype=float)))
            ),
            axis=1,
        )
        enriched["uncertainty_proxy_3d"] = enriched.apply(
            lambda row: self._rss(
                (np.array(row["stock_q90_path"], dtype=float) - np.array(row["stock_q10_path"], dtype=float))
                + (abs(float(row["beta_used"])) * (np.array(row["spy_q90_path"], dtype=float) - np.array(row["spy_q10_path"], dtype=float)))
            ),
            axis=1,
        )
        enriched["skew_ratio_1d"] = enriched.apply(
            lambda row: self._rss(np.array(row["stock_q90_path"][:7], dtype=float) - np.array(row["stock_q50_path"][:7], dtype=float))
            / max(
                self._rss(np.array(row["stock_q50_path"][:7], dtype=float) - np.array(row["stock_q10_path"][:7], dtype=float)),
                EPSILON,
            ),
            axis=1,
        )
        enriched["skew_ratio_3d"] = enriched.apply(
            lambda row: self._rss(np.array(row["stock_q90_path"], dtype=float) - np.array(row["stock_q50_path"], dtype=float))
            / max(
                self._rss(np.array(row["stock_q50_path"], dtype=float) - np.array(row["stock_q10_path"], dtype=float)),
                EPSILON,
            ),
            axis=1,
        )
        enriched["confidence_proxy"] = (
            0.5 * enriched["pred_ret_1d"].abs() + 0.5 * enriched["pred_alpha_1d"].abs()
        ) / (EPSILON + enriched["uncertainty_proxy_1d"])
        enriched["score_risk"] = enriched["pred_alpha_1d"] / (EPSILON + enriched["uncertainty_proxy_1d"])
        enriched["score_z"] = (
            enriched.groupby(["basket_id", "prediction_session"], group_keys=False)["score_plain"].transform(_safe_numeric_zscore)
        )

        enriched = self._add_approximate_coverage_fields(enriched)
        uncertainty_cutoff = enriched.groupby("prediction_session")["uncertainty_proxy_1d"].transform(
            lambda series: float(series.quantile(0.75)) if not series.dropna().empty else float("nan")
        )
        enriched["uncertainty_cutoff_1d"] = uncertainty_cutoff
        enriched["buy_eligible"] = (
            (enriched["pred_ret_1d"] > 0.0)
            & (enriched["pred_alpha_1d"] > 0.0)
            & (enriched["skew_ratio_1d"] > 1.0)
            & (enriched["signal_day_overnight_gap_pct"].abs() < 5.0)
            & (enriched["uncertainty_proxy_1d"] <= enriched["uncertainty_cutoff_1d"])
        )
        if confirmation_mode == "1d_only":
            enriched["confirmed_3d"] = True
        else:
            enriched["confirmed_3d"] = (enriched["pred_ret_3d"] >= 0.0) & (enriched["pred_alpha_3d"] >= 0.0)

        gap_multiplier = np.where(
            enriched["signal_day_overnight_gap_pct"].abs() < 2.0,
            1.0,
            np.where(enriched["signal_day_overnight_gap_pct"].abs() < 5.0, 0.5, 0.0),
        )
        base_edge_1d = (0.5 * enriched["pred_ret_1d"]) + (0.5 * enriched["pred_alpha_1d"])
        enriched["winner_score"] = (
            base_edge_1d * np.maximum(enriched["skew_ratio_1d"], 1.0) * gap_multiplier
        ) / (EPSILON + enriched["uncertainty_proxy_1d"])
        enriched["filter_fail_reasons"] = enriched.apply(
            lambda row: self._winner_filter_reasons(row, confirmation_mode=confirmation_mode),
            axis=1,
        )
        enriched["signal_label"] = np.where(
            enriched["buy_eligible"] & ~enriched["confirmed_3d"],
            "WATCH",
            "AVOID",
        )
        return enriched

    def _add_approximate_coverage_fields(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()

        def _approx_bounds(row: pd.Series, *, lower_quantile: str, upper_quantile: str, horizon: int) -> tuple[float, float]:
            stock_lower = np.array(row[lower_quantile][:horizon], dtype=float)
            stock_upper = np.array(row[upper_quantile][:horizon], dtype=float)
            stock_mid = np.array(row["stock_q50_path"][:horizon], dtype=float)
            mid_log = float(np.sum(stock_mid))
            upside = self._rss(stock_upper - stock_mid)
            downside = self._rss(stock_mid - stock_lower)
            lower = float(np.exp(mid_log - downside) - 1.0)
            upper = float(np.exp(mid_log + upside) - 1.0)
            return lower, upper

        for horizon, suffix in ((7, "1d"), (len(enriched.iloc[0]["stock_q50_path"]) if not enriched.empty else 21, "3d")):
            lower_80, upper_80, lower_90, upper_90 = [], [], [], []
            for row in enriched.itertuples(index=False):
                row_series = pd.Series(row._asdict())
                lo80, hi80 = _approx_bounds(row_series, lower_quantile="stock_q10_path", upper_quantile="stock_q90_path", horizon=horizon)
                lo90, hi90 = _approx_bounds(row_series, lower_quantile="stock_q05_path", upper_quantile="stock_q95_path", horizon=horizon)
                lower_80.append(lo80)
                upper_80.append(hi80)
                lower_90.append(lo90)
                upper_90.append(hi90)
            enriched[f"approx_ret_lower_80_{suffix}"] = lower_80
            enriched[f"approx_ret_upper_80_{suffix}"] = upper_80
            enriched[f"approx_ret_lower_90_{suffix}"] = lower_90
            enriched[f"approx_ret_upper_90_{suffix}"] = upper_90
        return enriched

    @staticmethod
    def _winner_filter_reasons(row: pd.Series, *, confirmation_mode: str) -> str:
        reasons: list[str] = []
        if float(row["pred_ret_1d"]) <= 0.0:
            reasons.append("pred_ret_1d_non_positive")
        if float(row["pred_alpha_1d"]) <= 0.0:
            reasons.append("pred_alpha_1d_non_positive")
        if float(row["skew_ratio_1d"]) <= 1.0:
            reasons.append("skew_ratio_1d_not_above_1")
        if abs(float(row["signal_day_overnight_gap_pct"])) >= 5.0:
            reasons.append("overnight_gap_excluded")
        if float(row["uncertainty_proxy_1d"]) > float(row["uncertainty_cutoff_1d"]):
            reasons.append("uncertainty_in_worst_quartile")
        if confirmation_mode == "1d_plus_3d" and not bool(row["confirmed_3d"]):
            reasons.append("three_day_confirmation_failed")
        return "|".join(reasons)

    def _slot_timestamp(self, session_date: pd.Timestamp, slot_id: int) -> pd.Timestamp:
        slot_time = SLOT_TO_TIME[slot_id]
        session_ts = pd.Timestamp(session_date)
        date_text = session_ts.tz_localize(None).date().isoformat() if session_ts.tzinfo else session_ts.date().isoformat()
        tz = session_ts.tzinfo
        return pd.Timestamp(f"{date_text} {slot_time}", tz=tz)
