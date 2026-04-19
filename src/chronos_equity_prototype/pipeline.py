from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import BacktestRunner
from .baselines import BaselineScorer
from .config import ForecasterConfig, ResearchConfig
from .data import SessionNormalizer, YahooIntradayProvider
from .diagnostics import (
    EXACT_OPEN_ENTRY_FORMULA,
    EXECUTION_ALPHA_FORMULA,
    EXECUTION_EXIT_FORMULA,
    FIRST_15M_ENTRY_FORMULA,
    FIRST_5M_ENTRY_FORMULA,
    STRICT_EXECUTION_MODE_LABEL,
    attach_realized_outcomes_to_forecasts,
    build_backtest_scorecard,
    generate_live_reconciliation_outputs,
    generate_near_open_execution_analysis,
    generate_signal_execution_analysis,
    summarize_execution_daily_returns,
    write_diagnostic_outputs,
)
from .forecasting import ZeroShotForecaster
from .labels import LabelBuilder
from .portfolio import PortfolioBuilder
from .tasks import ChronosTaskBuilder
from .types import ExperimentResult, ForecastBatch, LivePredictionResult, StockTask
from .v2_basket import BasketPredictionInput, BasketV2Builder


def _model_slug(model_id: str) -> str:
    slug = model_id.strip().lower().replace("/", "_").replace("-", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in slug)


@dataclass
class ZeroShotResearchPipeline:
    provider: YahooIntradayProvider | None = None
    normalizer: SessionNormalizer | None = None
    label_builder: LabelBuilder | None = None
    task_builder: ChronosTaskBuilder | None = None
    forecaster: ZeroShotForecaster | None = None
    portfolio_builder: PortfolioBuilder | None = None
    backtest_runner: BacktestRunner | None = None
    baseline_scorer: BaselineScorer | None = None
    v2_builder: BasketV2Builder | None = None
    research_config: ResearchConfig = ResearchConfig()

    def __post_init__(self) -> None:
        self.provider = self.provider or YahooIntradayProvider()
        self.normalizer = self.normalizer or SessionNormalizer(market_timezone=self.research_config.market_timezone)
        self.label_builder = self.label_builder or LabelBuilder(realized_vol_window=self.research_config.realized_vol_window)
        self.task_builder = self.task_builder or ChronosTaskBuilder(benchmark_ticker=self.research_config.benchmark_ticker)
        self.forecaster = self.forecaster or ZeroShotForecaster()
        self.portfolio_builder = self.portfolio_builder or PortfolioBuilder()
        self.backtest_runner = self.backtest_runner or BacktestRunner()
        self.baseline_scorer = self.baseline_scorer or BaselineScorer(benchmark_ticker=self.research_config.benchmark_ticker)
        self.v2_builder = self.v2_builder or BasketV2Builder(
            benchmark_ticker=self.research_config.benchmark_ticker,
            realized_vol_window=self.research_config.realized_vol_window,
        )

    def prepare_labeled_frame(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        interval: str = "60m",
    ) -> pd.DataFrame:
        symbols = list(dict.fromkeys([*tickers, self.research_config.benchmark_ticker]))
        raw = self.provider.fetch(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            regular_only=True,
            auto_adjust=True,
        )
        normalized = self.normalizer.normalize(raw)
        return self.label_builder.build(normalized, benchmark=self.research_config.benchmark_ticker)

    def prepare_market_frames(
        self,
        tickers: Iterable[str],
        start: str,
        end: str,
        interval: str = "60m",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        symbols = list(dict.fromkeys([*tickers, self.research_config.benchmark_ticker]))
        raw = self.provider.fetch(
            tickers=symbols,
            start=start,
            end=end,
            interval=interval,
            regular_only=True,
            auto_adjust=True,
        )
        normalized_complete = self.normalizer.normalize(raw, drop_incomplete_sessions=True)
        labeled_frame = self.label_builder.build(normalized_complete, benchmark=self.research_config.benchmark_ticker)
        normalized_incomplete = self.normalizer.normalize(raw, drop_incomplete_sessions=False)
        v2_base_frame = self.v2_builder.build_base_frame(normalized_incomplete)
        return labeled_frame, v2_base_frame

    def _predict_by_signal_session(self, tasks: list[StockTask], config: ForecasterConfig) -> ForecastBatch:
        if not tasks:
            return ForecastBatch(frame=pd.DataFrame())

        frames: list[pd.DataFrame] = []
        grouped = self.task_builder.group_by_signal_session(tasks)
        for signal_session in sorted(grouped.keys()):
            day_tasks = sorted(grouped[signal_session], key=lambda task: task.ticker)
            for batch_start in range(0, len(day_tasks), config.batch_size):
                batch = day_tasks[batch_start : batch_start + config.batch_size]
                forecasts = self.forecaster.predict(batch, model_id=config.model_id, config=config)
                frames.append(forecasts.frame)

        if not frames:
            return ForecastBatch(frame=pd.DataFrame())
        combined = pd.concat(frames, ignore_index=True).sort_values(["signal_session", "ticker"]).reset_index(drop=True)
        return ForecastBatch(frame=combined)

    def _predict_v2_baskets(
        self,
        prediction_inputs: list[BasketPredictionInput],
        config: ForecasterConfig,
        *,
        product_mode: str = "ranker",
        confirmation_mode: str = "1d_plus_3d",
    ) -> ForecastBatch:
        if not prediction_inputs:
            return ForecastBatch(frame=pd.DataFrame())

        prediction_frames: list[pd.DataFrame] = []
        grouped: dict[str, list[BasketPredictionInput]] = {}
        for prediction_input in prediction_inputs:
            grouped.setdefault(prediction_input.basket_id, []).append(prediction_input)

        for basket_id in sorted(grouped.keys()):
            basket_inputs = grouped[basket_id]
            history_df = pd.concat([prediction_input.history_df for prediction_input in basket_inputs], ignore_index=True)
            future_frames = [prediction_input.future_df for prediction_input in basket_inputs if prediction_input.future_df is not None]
            future_df = pd.concat(future_frames, ignore_index=True) if future_frames else None
            prediction_frames.append(
                self.forecaster.predict_dataframe(
                    history_df,
                    model_id=config.model_id,
                    config=config,
                    target_columns=basket_inputs[0].model_target_columns,
                    future_df=future_df,
                )
            )

        flattened = self.v2_builder.flatten_predictions(
            prediction_inputs,
            prediction_frames,
            model_id=config.model_id,
            context_length=config.context_length,
            cross_learning=config.cross_learning,
            use_covariates=config.use_covariates,
            beta_mode=config.beta_mode,
            product_mode=product_mode,
            confirmation_mode=confirmation_mode,
        )
        return ForecastBatch(frame=flattened)

    @staticmethod
    def _build_v2_live_freshness_metadata(
        v2_base_frame: pd.DataFrame,
        signal_session: pd.Timestamp | None,
    ) -> dict[str, object]:
        if v2_base_frame.empty or signal_session is None or "session_date" not in v2_base_frame.columns:
            return {
                "observed_latest_session": pd.NaT,
                "has_newer_observed_session_than_signal": False,
                "prediction_freshness": "unknown",
                "prediction_freshness_note": "",
            }

        observed_sessions = pd.to_datetime(v2_base_frame["session_date"], utc=True, errors="coerce").dropna()
        if observed_sessions.empty:
            return {
                "observed_latest_session": pd.NaT,
                "has_newer_observed_session_than_signal": False,
                "prediction_freshness": "unknown",
                "prediction_freshness_note": "",
            }

        latest_observed_session = observed_sessions.max()
        normalized_signal = pd.Timestamp(signal_session)
        if normalized_signal.tzinfo is None:
            normalized_signal = normalized_signal.tz_localize("UTC")
        else:
            normalized_signal = normalized_signal.tz_convert("UTC")
        has_newer_session = bool(latest_observed_session > normalized_signal)
        if has_newer_session:
            freshness = "newer_incomplete_session_present"
            note = (
                "Fetched market data already contained a later observed session than the signal session. "
                "That newer session was excluded from history because it was incomplete, so this run targets "
                "that later session rather than a fully unopened future session."
            )
        else:
            freshness = "clean_next_session"
            note = "No later observed session was present in the fetched market window."

        return {
            "observed_latest_session": latest_observed_session,
            "has_newer_observed_session_than_signal": has_newer_session,
            "prediction_freshness": freshness,
            "prediction_freshness_note": note,
        }

    @staticmethod
    def _validate_v2_ranker_request(*, prediction_length: int) -> None:
        if int(prediction_length) != 7:
            raise ValueError(
                "The V2 ranker path is defined as one full next regular session = 7 slots. "
                f"Received prediction_length={prediction_length}. Keep prediction_length=7 for mathematically correct ranker usage."
            )

    @staticmethod
    def _validate_v2_winner_request(*, prediction_length: int) -> None:
        if int(prediction_length) != 21:
            raise ValueError(
                "The V2 winner_picker path is defined as three regular sessions = 21 slots. "
                f"Received prediction_length={prediction_length}. Keep prediction_length=21 for mathematically correct winner usage."
            )

    @staticmethod
    def _assert_clean_future_prediction_window(
        freshness: dict[str, object],
        *,
        signal_session: pd.Timestamp | None,
    ) -> None:
        if not bool(freshness.get("has_newer_observed_session_than_signal", False)):
            return

        signal_text = (
            str(pd.Timestamp(signal_session).date())
            if signal_session is not None
            else "unknown"
        )
        observed_latest = freshness.get("observed_latest_session", pd.NaT)
        observed_text = (
            str(pd.Timestamp(observed_latest).date())
            if pd.notna(observed_latest)
            else "unknown"
        )
        raise ValueError(
            "predict-next refused to run because the fetched market window already contains a newer observed session "
            f"({observed_text}) than the latest complete signal session ({signal_text}). "
            "That means the target session has already started, so this would not be a clean future next-session forecast. "
            "Run after the regular market close, or cap the fetch window so only fully completed sessions are included."
        )

    @staticmethod
    def _assert_prediction_session_after_current_market_day(
        prediction_session: pd.Timestamp | None,
        *,
        market_timezone: str,
        now: pd.Timestamp | None = None,
    ) -> None:
        if prediction_session is None:
            return

        prediction_ts = pd.Timestamp(prediction_session)
        if prediction_ts.tzinfo is None:
            prediction_market_date = prediction_ts.tz_localize("UTC").tz_convert(market_timezone).normalize()
        else:
            prediction_market_date = prediction_ts.tz_convert(market_timezone).normalize()

        current_ts = now if now is not None else pd.Timestamp.now(tz=market_timezone)
        if current_ts.tzinfo is None:
            current_market_date = current_ts.tz_localize(market_timezone).normalize()
        else:
            current_market_date = current_ts.tz_convert(market_timezone).normalize()

        if prediction_market_date <= current_market_date:
            raise ValueError(
                "predict-next refused to run because the target prediction_session is not strictly after the current "
                f"US market day. Current market date is {current_market_date.date()}, while prediction_session is "
                f"{prediction_market_date.date()}. This project is defined as after-close forecasting for the following "
                "session, so live mode will not emit today's or a past session."
            )

    @staticmethod
    def _normalize_market_timestamp(
        timestamp: pd.Timestamp | str | None,
        *,
        market_timezone: str,
    ) -> pd.Timestamp | None:
        if timestamp is None:
            return None
        normalized = pd.Timestamp(timestamp)
        if normalized.tzinfo is None:
            return normalized.tz_localize(market_timezone)
        return normalized.tz_convert(market_timezone)

    @classmethod
    def reconciliation_readiness(
        cls,
        prediction_sessions: Iterable[pd.Timestamp | str],
        *,
        market_timezone: str,
        now: pd.Timestamp | None = None,
    ) -> tuple[bool, str]:
        normalized_sessions = pd.to_datetime(list(prediction_sessions), utc=True, errors="coerce")
        normalized_sessions = [session for session in normalized_sessions.tolist() if pd.notna(session)]
        if not normalized_sessions:
            return True, ""

        current_market_now = cls._normalize_market_timestamp(
            now if now is not None else pd.Timestamp.now(tz=market_timezone),
            market_timezone=market_timezone,
        )
        if current_market_now is None:
            return True, ""
        current_market_date = current_market_now.normalize()
        regular_close = current_market_date + pd.Timedelta(hours=16)

        blocked_future_sessions: list[pd.Timestamp] = []
        blocked_same_day_sessions: list[pd.Timestamp] = []
        for prediction_session in normalized_sessions:
            prediction_market_date = cls._normalize_market_timestamp(
                prediction_session,
                market_timezone=market_timezone,
            )
            if prediction_market_date is None:
                continue
            prediction_market_date = prediction_market_date.normalize()
            if prediction_market_date > current_market_date:
                blocked_future_sessions.append(prediction_market_date)
            elif prediction_market_date == current_market_date and current_market_now < regular_close:
                blocked_same_day_sessions.append(prediction_market_date)

        if blocked_future_sessions:
            next_blocked = min(blocked_future_sessions)
            return (
                False,
                "Reconciliation deferred because prediction_session "
                f"{next_blocked.date()} is still in the future relative to the current New York market date "
                f"{current_market_date.date()}. Realized session and 5-minute execution data must only be fetched "
                "after the predicted session exists and finishes.",
            )

        if blocked_same_day_sessions:
            next_blocked = min(blocked_same_day_sessions)
            return (
                False,
                "Reconciliation deferred because prediction_session "
                f"{next_blocked.date()} is today in New York and the regular session is not fully complete yet. "
                f"Current New York time is {current_market_now.strftime('%Y-%m-%d %H:%M:%S %Z')}. "
                "Run reconcile-live after the regular close so realized session and 5-minute execution data are final.",
            )

        return True, ""

    def compute_rank_ic(
        self,
        forecasts: pd.DataFrame,
        realized_returns: pd.DataFrame,
        *,
        score_column: str = "score_q50",
    ) -> pd.DataFrame:
        def _safe_stat(series: pd.Series, stat: str) -> float:
            clean = series.dropna()
            if clean.empty:
                return float("nan")
            if stat == "mean":
                return float(clean.mean())
            if stat == "median":
                return float(clean.median())
            raise ValueError(f"Unsupported stat: {stat}")

        if "realized_day_return" in forecasts.columns:
            merged = forecasts.copy()
        else:
            merged = forecasts.merge(
                realized_returns[["ticker", "prediction_session", "realized_day_return"]],
                on=["ticker", "prediction_session"],
                how="left",
                validate="many_to_one",
            )
        if merged.empty:
            return pd.DataFrame(columns=["prediction_session", "rank_ic"])
        if merged["realized_day_return"].isna().any():
            raise ValueError("Missing realized_day_return values while computing Rank IC.")
        if score_column not in merged.columns:
            raise ValueError(f"Missing score column {score_column!r} while computing Rank IC.")

        daily_ic_records: list[dict[str, object]] = []
        for prediction_session, day_frame in merged.groupby("prediction_session", sort=True):
            ranked_scores = day_frame[score_column].rank(method="average")
            ranked_returns = day_frame["realized_day_return"].rank(method="average")
            if ranked_scores.nunique(dropna=True) <= 1 or ranked_returns.nunique(dropna=True) <= 1:
                rank_ic = float("nan")
            else:
                rank_ic = float(ranked_scores.corr(ranked_returns))
            daily_ic_records.append({"prediction_session": prediction_session, "rank_ic": rank_ic})

        daily_ic = pd.DataFrame.from_records(daily_ic_records)
        summary = pd.DataFrame(
            [
                {"metric": "mean_rank_ic", "value": _safe_stat(daily_ic["rank_ic"], "mean")},
                {"metric": "median_rank_ic", "value": _safe_stat(daily_ic["rank_ic"], "median")},
                {"metric": "positive_rank_ic_rate", "value": float((daily_ic["rank_ic"] > 0).mean())},
            ]
        )
        return pd.concat([daily_ic, summary], ignore_index=True, sort=False)

    def _build_winner_outputs(
        self,
        forecasts: pd.DataFrame,
        book_frame: pd.DataFrame,
        *,
        include_realized: bool,
        promotable_threshold: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        board = forecasts.copy()
        if board.empty:
            empty_board = pd.DataFrame()
            empty_summary = pd.DataFrame()
            return {
                "winner_board": empty_board,
                "winner_top_picks": empty_board,
                "winner_daily_positions": empty_board,
                "winner_daily_summary": empty_summary,
                "winner_backtest_summary": empty_summary,
            }

        selected_columns = [
            column
            for column in [
                "signal_session",
                "prediction_session",
                "ticker",
                "rank",
                "weight",
                "ranking_score_column",
                "ranking_score",
                "book_mode",
                "tail_fraction",
                "tail_size",
                "book_active",
            ]
            if column in book_frame.columns
        ]
        selected = book_frame[selected_columns].drop_duplicates(subset=["prediction_session", "ticker"]) if not book_frame.empty else pd.DataFrame(columns=selected_columns)
        board = board.merge(
            selected,
            on=["signal_session", "prediction_session", "ticker"],
            how="left",
            validate="one_to_one",
        )
        board["selected"] = board["rank"].notna()

        def _label_row(row: pd.Series) -> str:
            if bool(row["selected"]):
                if float(row["skew_ratio_1d"]) > 1.3 and abs(float(row["signal_day_overnight_gap_pct"])) < 2.0:
                    return "BUY_STRONG"
                return "BUY"
            if bool(row["buy_eligible"]) and not bool(row["confirmed_3d"]):
                return "WATCH"
            if bool(row["buy_eligible"]) and bool(row["confirmed_3d"]):
                return "WATCH"
            return "AVOID"

        board["signal_label"] = board.apply(_label_row, axis=1)
        board["filter_fail_reasons"] = board.apply(
            lambda row: (
                f"{row['filter_fail_reasons']}|not_top_winner_score"
                if bool(row["buy_eligible"]) and bool(row["confirmed_3d"]) and not bool(row["selected"]) and str(row["filter_fail_reasons"])
                else ("not_top_winner_score" if bool(row["buy_eligible"]) and bool(row["confirmed_3d"]) and not bool(row["selected"]) else str(row["filter_fail_reasons"]))
            ).strip("|"),
            axis=1,
        )

        winner_top_picks = board[board["selected"]].copy().sort_values(["prediction_session", "rank", "ticker"]).reset_index(drop=True)
        winner_daily_positions = winner_top_picks.copy()
        if include_realized:
            winner_daily_positions["gross_position_return"] = winner_daily_positions["weight"] * winner_daily_positions["realized_day_return"]
            winner_daily_positions["beats_spy_1d"] = winner_daily_positions["realized_stock_ret_1d"] > winner_daily_positions["realized_spy_ret_1d"]
            winner_daily_positions["book_helped"] = winner_daily_positions["gross_position_return"] > 0
            winner_daily_positions["book_impact"] = np.where(
                winner_daily_positions["book_helped"],
                "helped",
                np.where(winner_daily_positions["gross_position_return"] < 0, "hurt", "flat"),
            )
        else:
            winner_daily_positions["gross_position_return"] = np.nan
            winner_daily_positions["beats_spy_1d"] = np.nan
            winner_daily_positions["book_helped"] = np.nan
            winner_daily_positions["book_impact"] = ""

        prediction_sessions = sorted(pd.to_datetime(board["prediction_session"], utc=True).dropna().unique().tolist())
        daily_records: list[dict[str, object]] = []
        for prediction_session in prediction_sessions:
            session_board = board[board["prediction_session"] == prediction_session].copy()
            session_positions = winner_daily_positions[winner_daily_positions["prediction_session"] == prediction_session].copy()
            n_selected = int(len(session_positions))
            gross_return = float(session_positions["gross_position_return"].sum()) if include_realized and n_selected > 0 else 0.0
            gross_exposure = float(session_positions["weight"].abs().sum()) if n_selected > 0 else 0.0
            record = {
                "signal_session": session_board["signal_session"].iloc[0],
                "prediction_session": prediction_session,
                "prediction_session_end": session_board["prediction_session_end"].iloc[0] if "prediction_session_end" in session_board.columns else prediction_session,
                "n_candidates": int(len(session_board)),
                "n_buy_eligible": int(session_board["buy_eligible"].sum()),
                "n_confirmed_3d": int(session_board["confirmed_3d"].sum()),
                "n_selected": n_selected,
                "book_active": bool(n_selected > 0),
                "selected_tickers": "|".join(session_positions["ticker"].tolist()),
                "gross_return": gross_return,
                "gross_exposure": gross_exposure,
                "mean_selected_return": float(session_positions["realized_stock_ret_1d"].mean()) if include_realized and n_selected > 0 else float("nan"),
                "mean_selected_alpha": float(session_positions["realized_alpha_1d"].mean()) if include_realized and n_selected > 0 else float("nan"),
                "pct_selected_beating_spy": float(session_positions["beats_spy_1d"].mean()) if include_realized and n_selected > 0 else float("nan"),
                "best_contributor": (
                    session_positions.sort_values(["gross_position_return", "ticker"], ascending=[False, True]).iloc[0]["ticker"]
                    if include_realized and n_selected > 0
                    else ""
                ),
                "worst_contributor": (
                    session_positions.sort_values(["gross_position_return", "ticker"], ascending=[True, True]).iloc[0]["ticker"]
                    if include_realized and n_selected > 0
                    else ""
                ),
            }
            if include_realized:
                for cost_bps in self.research_config.costs_bps:
                    record[f"net_return_{cost_bps}bps"] = gross_return - (gross_exposure * (cost_bps / 10000.0))
            daily_records.append(record)
        winner_daily_summary = pd.DataFrame.from_records(daily_records).sort_values("prediction_session").reset_index(drop=True)

        if not include_realized:
            return {
                "winner_board": board.sort_values(["prediction_session", "winner_score", "ticker"], ascending=[True, False, True]).reset_index(drop=True),
                "winner_top_picks": winner_top_picks,
                "winner_daily_positions": winner_daily_positions,
                "winner_daily_summary": winner_daily_summary,
                "winner_backtest_summary": pd.DataFrame(),
            }

        daily_gross = winner_daily_summary["gross_return"] if not winner_daily_summary.empty else pd.Series(dtype=float)
        daily_summary_records: dict[str, object] = {
            "valid_prediction_days": int(len(winner_daily_summary)),
            "valid_pick_days": int(winner_daily_summary["book_active"].sum()) if not winner_daily_summary.empty else 0,
            "avg_picks_per_day": float(winner_daily_summary["n_selected"].mean()) if not winner_daily_summary.empty else 0.0,
            "selected_long_win_rate": float((winner_daily_positions["realized_alpha_1d"] > 0).mean()) if not winner_daily_positions.empty else float("nan"),
            "avg_selected_long_return": float(winner_daily_positions["realized_stock_ret_1d"].mean()) if not winner_daily_positions.empty else float("nan"),
            "avg_selected_long_alpha": float(winner_daily_positions["realized_alpha_1d"].mean()) if not winner_daily_positions.empty else float("nan"),
            "pct_selected_longs_beating_spy": float(winner_daily_positions["beats_spy_1d"].mean()) if not winner_daily_positions.empty else float("nan"),
            "gross_sharpe": float(np.sqrt(252) * daily_gross.mean() / daily_gross.std(ddof=1)) if len(daily_gross.dropna()) > 1 and float(daily_gross.std(ddof=1)) > 0 else float("nan"),
            "approx_pi80_coverage_1d": float(
                (
                    (winner_daily_positions["realized_stock_ret_1d"] >= winner_daily_positions["approx_ret_lower_80_1d"])
                    & (winner_daily_positions["realized_stock_ret_1d"] <= winner_daily_positions["approx_ret_upper_80_1d"])
                ).mean()
            ) if not winner_daily_positions.empty else float("nan"),
            "approx_pi90_coverage_1d": float(
                (
                    (winner_daily_positions["realized_stock_ret_1d"] >= winner_daily_positions["approx_ret_lower_90_1d"])
                    & (winner_daily_positions["realized_stock_ret_1d"] <= winner_daily_positions["approx_ret_upper_90_1d"])
                ).mean()
            ) if not winner_daily_positions.empty else float("nan"),
            "approx_pi80_coverage_3d": float(
                (
                    (winner_daily_positions["realized_stock_ret_3d"] >= winner_daily_positions["approx_ret_lower_80_3d"])
                    & (winner_daily_positions["realized_stock_ret_3d"] <= winner_daily_positions["approx_ret_upper_80_3d"])
                ).mean()
            ) if not winner_daily_positions.empty else float("nan"),
            "approx_pi90_coverage_3d": float(
                (
                    (winner_daily_positions["realized_stock_ret_3d"] >= winner_daily_positions["approx_ret_lower_90_3d"])
                    & (winner_daily_positions["realized_stock_ret_3d"] <= winner_daily_positions["approx_ret_upper_90_3d"])
                ).mean()
            ) if not winner_daily_positions.empty else float("nan"),
            "promotable": bool(len(winner_daily_summary) >= int(promotable_threshold or self.research_config.winner_min_promotable_days)),
        }
        for cost_bps in self.research_config.costs_bps:
            net_series = winner_daily_summary[f"net_return_{cost_bps}bps"] if not winner_daily_summary.empty else pd.Series(dtype=float)
            daily_summary_records[f"net_sharpe_{cost_bps}bps"] = float(
                np.sqrt(252) * net_series.mean() / net_series.std(ddof=1)
            ) if len(net_series.dropna()) > 1 and float(net_series.std(ddof=1)) > 0 else float("nan")

        if not winner_daily_positions.empty:
            ticker_pnl = winner_daily_positions.groupby("ticker")["gross_position_return"].sum().abs()
            total_abs_pnl = float(ticker_pnl.sum())
            if total_abs_pnl > 0:
                shares = ticker_pnl / total_abs_pnl
                daily_summary_records["contribution_top1_share"] = float(shares.max())
                daily_summary_records["contribution_hhi"] = float(np.square(shares).sum())
            else:
                daily_summary_records["contribution_top1_share"] = 0.0
                daily_summary_records["contribution_hhi"] = 0.0
            basket_counts = winner_daily_positions["basket_id"].value_counts(normalize=True)
            daily_summary_records["basket_selection_concentration_proxy"] = float(basket_counts.max()) if not basket_counts.empty else float("nan")
        else:
            daily_summary_records["contribution_top1_share"] = float("nan")
            daily_summary_records["contribution_hhi"] = float("nan")
            daily_summary_records["basket_selection_concentration_proxy"] = float("nan")

        return {
            "winner_board": board.sort_values(["prediction_session", "winner_score", "ticker"], ascending=[True, False, True]).reset_index(drop=True),
            "winner_top_picks": winner_top_picks,
            "winner_daily_positions": winner_daily_positions,
            "winner_daily_summary": winner_daily_summary,
            "winner_backtest_summary": pd.DataFrame([daily_summary_records]),
        }

    @staticmethod
    def _validate_ranker_only_request(*, product_mode: str, architecture_label: str) -> None:
        if product_mode != "ranker":
            raise ValueError(
                f"{architecture_label} only supports product_mode='ranker'. "
                f"Received product_mode={product_mode!r}."
            )

    @staticmethod
    def _detect_product_mode(
        *,
        prediction_summary: pd.DataFrame | None = None,
        forecasts: pd.DataFrame | None = None,
    ) -> str:
        for frame in [prediction_summary, forecasts]:
            if frame is None or frame.empty or "product_mode" not in frame.columns:
                continue
            values = frame["product_mode"].dropna()
            if not values.empty:
                return str(values.iloc[0])
        return "ranker"

    @staticmethod
    def _prediction_sessions_from_payload(
        *,
        prediction_summary: pd.DataFrame | None = None,
        book: pd.DataFrame | None = None,
        forecasts: pd.DataFrame | None = None,
    ) -> list[pd.Timestamp]:
        for frame in [book, prediction_summary, forecasts]:
            if frame is None or frame.empty or "prediction_session" not in frame.columns:
                continue
            sessions = pd.to_datetime(frame["prediction_session"], utc=True, errors="coerce").dropna().unique().tolist()
            if sessions:
                return sorted(sessions)
        return []

    def _winner_max_picks_fraction(self) -> float:
        return float(self.research_config.default_winner_max_picks_fraction)

    def _build_winner_prediction_summary(
        self,
        *,
        experiment_name: str,
        forecasts: pd.DataFrame,
        book_frame: pd.DataFrame,
        forecaster_config: ForecasterConfig,
        basket_mode: str,
        freshness: dict[str, object],
        winner_outputs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        columns = [
            "experiment_name",
            "generated_at_utc",
            "signal_session",
            "prediction_session",
            "prediction_session_end",
            "n_forecasts",
            "n_candidates",
            "n_buy_eligible",
            "n_confirmed_3d",
            "n_selected",
            "n_longs",
            "n_shorts",
            "book_active",
            "model_id",
            "context_length",
            "prediction_length",
            "cross_learning",
            "use_covariates",
            "basket_mode",
            "beta_mode",
            "product_mode",
            "confirmation_mode",
            "book_mode",
            "tail_fraction",
            "tail_size",
            "score_column",
            "observed_latest_session",
            "has_newer_observed_session_than_signal",
            "prediction_freshness",
            "prediction_freshness_note",
        ]
        if forecasts.empty:
            return pd.DataFrame(columns=columns)

        daily_summary = winner_outputs.get("winner_daily_summary", pd.DataFrame())
        daily_row = daily_summary.iloc[0].to_dict() if not daily_summary.empty else {}
        selected_count = int(len(book_frame))
        prediction_session_end = (
            forecasts["prediction_session_end"].iloc[0]
            if "prediction_session_end" in forecasts.columns and not forecasts.empty
            else forecasts["prediction_session"].iloc[0]
        )
        tail_size = int(book_frame["tail_size"].dropna().iloc[0]) if "tail_size" in book_frame.columns and not book_frame["tail_size"].dropna().empty else 0
        return pd.DataFrame(
            [
                {
                    "experiment_name": experiment_name,
                    "generated_at_utc": pd.Timestamp.utcnow(),
                    "signal_session": forecasts["signal_session"].iloc[0],
                    "prediction_session": forecasts["prediction_session"].iloc[0],
                    "prediction_session_end": prediction_session_end,
                    "n_forecasts": int(len(forecasts)),
                    "n_candidates": int(daily_row.get("n_candidates", len(forecasts))),
                    "n_buy_eligible": int(daily_row.get("n_buy_eligible", 0)),
                    "n_confirmed_3d": int(daily_row.get("n_confirmed_3d", 0)),
                    "n_selected": selected_count,
                    "n_longs": selected_count,
                    "n_shorts": 0,
                    "book_active": bool(selected_count > 0),
                    "model_id": forecaster_config.model_id,
                    "context_length": forecaster_config.context_length,
                    "prediction_length": forecaster_config.prediction_length,
                    "cross_learning": forecaster_config.cross_learning,
                    "use_covariates": forecaster_config.use_covariates,
                    "basket_mode": basket_mode,
                    "beta_mode": forecaster_config.beta_mode,
                    "product_mode": forecaster_config.product_mode,
                    "confirmation_mode": forecaster_config.confirmation_mode,
                    "book_mode": "winner_top_fraction",
                    "tail_fraction": self._winner_max_picks_fraction(),
                    "tail_size": tail_size,
                    "score_column": "winner_score",
                    **freshness,
                }
            ]
        )

    def _realize_winner_forecasts(
        self,
        *,
        forecasts: pd.DataFrame,
        labeled_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        realized = attach_realized_outcomes_to_forecasts(
            forecasts,
            labeled_frame,
            benchmark_ticker=self.research_config.benchmark_ticker,
        )
        if realized.empty:
            return realized

        def _convert_log_path_to_return(values: object, horizon: int | None = None) -> float:
            if not isinstance(values, list):
                return float("nan")
            subset = values[:horizon] if horizon is not None else values
            if len(subset) == 0:
                return float("nan")
            return float(np.exp(np.sum(np.asarray(subset, dtype=float))) - 1.0)

        realized["realized_stock_ret_1d"] = realized["realized_slot_path"].apply(lambda values: _convert_log_path_to_return(values, 7))
        realized["realized_spy_ret_1d"] = realized["spy_realized_slot_path"].apply(lambda values: _convert_log_path_to_return(values, 7))
        realized["realized_stock_ret_3d"] = realized["realized_slot_path"].apply(_convert_log_path_to_return)
        realized["realized_spy_ret_3d"] = realized["spy_realized_slot_path"].apply(_convert_log_path_to_return)
        realized["realized_alpha_1d"] = realized["realized_stock_ret_1d"] - (realized["beta_used"] * realized["realized_spy_ret_1d"])
        realized["realized_alpha_3d"] = realized["realized_stock_ret_3d"] - (realized["beta_used"] * realized["realized_spy_ret_3d"])
        realized["realized_day_return"] = realized["realized_alpha_1d"]
        return realized

    def _summarize_cumulative_winner_ledger(
        self,
        *,
        winner_positions: pd.DataFrame,
        winner_daily_summary: pd.DataFrame,
        promotable_threshold: int | None = None,
    ) -> pd.DataFrame:
        if winner_daily_summary.empty:
            return pd.DataFrame(
                columns=[
                    "valid_prediction_days",
                    "valid_pick_days",
                    "avg_picks_per_day",
                    "selected_long_win_rate",
                    "avg_selected_long_return",
                    "avg_selected_long_alpha",
                    "pct_selected_longs_beating_spy",
                    "gross_sharpe",
                    "approx_pi80_coverage_1d",
                    "approx_pi90_coverage_1d",
                    "approx_pi80_coverage_3d",
                    "approx_pi90_coverage_3d",
                    "promotable",
                ]
            )

        positions = winner_positions.copy()
        summary_record: dict[str, object] = {
            "valid_prediction_days": int(winner_daily_summary["prediction_session"].nunique()),
            "valid_pick_days": int(winner_daily_summary["book_active"].fillna(False).astype(bool).sum()),
            "avg_picks_per_day": float(winner_daily_summary["n_selected"].mean()),
            "gross_sharpe": float(np.sqrt(252) * winner_daily_summary["gross_return"].mean() / winner_daily_summary["gross_return"].std(ddof=1))
            if len(winner_daily_summary["gross_return"].dropna()) > 1 and float(winner_daily_summary["gross_return"].std(ddof=1)) > 0
            else float("nan"),
            "promotable": bool(
                winner_daily_summary["prediction_session"].nunique()
                >= int(promotable_threshold or self.research_config.winner_min_promotable_days)
            ),
        }
        for cost_bps in self.research_config.costs_bps:
            column = f"net_return_{cost_bps}bps"
            if column in winner_daily_summary.columns:
                series = winner_daily_summary[column]
                summary_record[f"net_sharpe_{cost_bps}bps"] = float(np.sqrt(252) * series.mean() / series.std(ddof=1)) if len(series.dropna()) > 1 and float(series.std(ddof=1)) > 0 else float("nan")

        if positions.empty:
            summary_record.update(
                {
                    "selected_long_win_rate": float("nan"),
                    "avg_selected_long_return": float("nan"),
                    "avg_selected_long_alpha": float("nan"),
                    "pct_selected_longs_beating_spy": float("nan"),
                    "approx_pi80_coverage_1d": float("nan"),
                    "approx_pi90_coverage_1d": float("nan"),
                    "approx_pi80_coverage_3d": float("nan"),
                    "approx_pi90_coverage_3d": float("nan"),
                    "contribution_top1_share": float("nan"),
                    "contribution_hhi": float("nan"),
                    "basket_selection_concentration_proxy": float("nan"),
                }
            )
            return pd.DataFrame([summary_record])

        realized_return_column = "realized_stock_ret_1d" if "realized_stock_ret_1d" in positions.columns else "next_day_realized_return"
        realized_alpha_column = "realized_alpha_1d" if "realized_alpha_1d" in positions.columns else "next_day_realized_alpha"
        summary_record["selected_long_win_rate"] = float((positions[realized_alpha_column] > 0).mean())
        summary_record["avg_selected_long_return"] = float(positions[realized_return_column].mean())
        summary_record["avg_selected_long_alpha"] = float(positions[realized_alpha_column].mean())
        if "realized_spy_ret_1d" in positions.columns and "realized_stock_ret_1d" in positions.columns:
            summary_record["pct_selected_longs_beating_spy"] = float((positions["realized_stock_ret_1d"] > positions["realized_spy_ret_1d"]).mean())
        else:
            summary_record["pct_selected_longs_beating_spy"] = float("nan")

        for lower, upper, label in [
            ("approx_ret_lower_80_1d", "approx_ret_upper_80_1d", "approx_pi80_coverage_1d"),
            ("approx_ret_lower_90_1d", "approx_ret_upper_90_1d", "approx_pi90_coverage_1d"),
            ("approx_ret_lower_80_3d", "approx_ret_upper_80_3d", "approx_pi80_coverage_3d"),
            ("approx_ret_lower_90_3d", "approx_ret_upper_90_3d", "approx_pi90_coverage_3d"),
        ]:
            if lower in positions.columns and upper in positions.columns:
                realized_column = "realized_stock_ret_1d" if label.endswith("1d") else "realized_stock_ret_3d"
                if realized_column in positions.columns:
                    summary_record[label] = float(
                        ((positions[realized_column] >= positions[lower]) & (positions[realized_column] <= positions[upper])).mean()
                    )
                else:
                    summary_record[label] = float("nan")

        ticker_pnl = positions.groupby("ticker")["gross_position_return"].sum().abs()
        total_abs_pnl = float(ticker_pnl.sum())
        if total_abs_pnl > 0:
            shares = ticker_pnl / total_abs_pnl
            summary_record["contribution_top1_share"] = float(shares.max())
            summary_record["contribution_hhi"] = float(np.square(shares).sum())
        else:
            summary_record["contribution_top1_share"] = 0.0
            summary_record["contribution_hhi"] = 0.0
        basket_counts = positions["basket_id"].value_counts(normalize=True) if "basket_id" in positions.columns else pd.Series(dtype=float)
        summary_record["basket_selection_concentration_proxy"] = float(basket_counts.max()) if not basket_counts.empty else float("nan")
        return pd.DataFrame([summary_record])

    @staticmethod
    def _sanitize_generated_output(frame: pd.DataFrame) -> pd.DataFrame:
        sanitized = frame.copy()
        is_v2_output = "basket_id" in sanitized.columns or (
            "product_mode" in sanitized.columns
            and sanitized["product_mode"].astype(str).isin(["ranker", "winner_picker"]).any()
        )
        if is_v2_output and "score_q50" in sanitized.columns:
            sanitized = sanitized.drop(columns=["score_q50"])
        return sanitized

    def run_experiment(
        self,
        labeled_frame: pd.DataFrame,
        forecaster_config: ForecasterConfig,
        experiment_name: str,
    ) -> ExperimentResult:
        self._validate_ranker_only_request(
            product_mode=forecaster_config.product_mode,
            architecture_label="v1_residual",
        )
        tasks = self.task_builder.build_tasks(
            labeled_frame,
            context_length=forecaster_config.context_length,
            prediction_length=forecaster_config.prediction_length,
            use_covariates=forecaster_config.use_covariates,
        )
        forecasts = self._predict_by_signal_session(tasks, forecaster_config)
        realized = self.baseline_scorer.build_realized_daily_returns(labeled_frame)
        rank_ic_summary = self.compute_rank_ic(forecasts.frame, realized)
        book = self.portfolio_builder.rank_and_size(
            forecasts,
            n_long=self.research_config.n_long,
            n_short=self.research_config.n_short,
        )
        metrics = self.backtest_runner.run(book, realized, self.research_config.costs_bps)
        return ExperimentResult(
            experiment_name=experiment_name,
            forecasts=forecasts,
            book=book,
            metrics=metrics,
            rank_ic_summary=rank_ic_summary,
        )

    def run_v2_experiment(
        self,
        v2_base_frame: pd.DataFrame,
        forecaster_config: ForecasterConfig,
        experiment_name: str,
        *,
        basket_mode: str = "full",
        book_mode: str = "tail_fraction",
        tail_fraction: float = 0.10,
        score_column: str = "score_z",
    ) -> ExperimentResult:
        product_mode = forecaster_config.product_mode
        if product_mode == "winner_picker":
            self._validate_v2_winner_request(prediction_length=forecaster_config.prediction_length)
        else:
            self._validate_v2_ranker_request(prediction_length=forecaster_config.prediction_length)
        prediction_inputs, data_quality_events, data_quality_summary = self.v2_builder.build_prediction_inputs(
            v2_base_frame,
            context_length=forecaster_config.context_length,
            prediction_length=forecaster_config.prediction_length,
            basket_mode=basket_mode,
            latest_only=False,
            use_covariates=forecaster_config.use_covariates,
            beta_mode=forecaster_config.beta_mode,
        )
        forecasts = self._predict_v2_baskets(
            prediction_inputs,
            forecaster_config,
            product_mode=product_mode,
            confirmation_mode=forecaster_config.confirmation_mode,
        )
        realized = self._realized_from_forecasts(forecasts.frame)
        extra_outputs: dict[str, pd.DataFrame] = {}
        if product_mode == "winner_picker":
            rank_ic_summary = self.compute_rank_ic(forecasts.frame, realized, score_column="winner_score")
            book = self.portfolio_builder.select_winners(
                forecasts,
                score_column="winner_score",
                max_fraction=self._winner_max_picks_fraction(),
            )
            extra_outputs = self._build_winner_outputs(
                forecasts.frame,
                book.frame,
                include_realized=True,
                promotable_threshold=self.research_config.winner_min_promotable_days,
            )
        else:
            rank_ic_summary = self.compute_rank_ic(forecasts.frame, realized, score_column=score_column)
            book = self.portfolio_builder.rank_and_size(
                forecasts,
                n_long=self.research_config.n_long,
                n_short=self.research_config.n_short,
                score_column=score_column,
                book_mode=book_mode,
                tail_fraction=tail_fraction,
            )
        metrics = self.backtest_runner.run(book, realized, self.research_config.costs_bps)
        return ExperimentResult(
            experiment_name=experiment_name,
            forecasts=forecasts,
            book=book,
            metrics=metrics,
            rank_ic_summary=rank_ic_summary,
            data_quality_events=data_quality_events,
            data_quality_summary=data_quality_summary,
            extra_outputs=extra_outputs,
        )

    def run_next_session_prediction(
        self,
        labeled_frame: pd.DataFrame,
        forecaster_config: ForecasterConfig,
        experiment_name: str,
    ) -> LivePredictionResult:
        self._validate_ranker_only_request(
            product_mode=forecaster_config.product_mode,
            architecture_label="v1_residual",
        )
        tasks = self.task_builder.build_latest_tasks(
            labeled_frame,
            context_length=forecaster_config.context_length,
            prediction_length=forecaster_config.prediction_length,
            use_covariates=forecaster_config.use_covariates,
        )
        self._assert_prediction_session_after_current_market_day(
            tasks[0].prediction_session if tasks else None,
            market_timezone=self.research_config.market_timezone,
        )
        forecasts = self.forecaster.predict(tasks, forecaster_config.model_id, forecaster_config)
        book = self.portfolio_builder.rank_and_size(
            forecasts,
            n_long=self.research_config.n_long,
            n_short=self.research_config.n_short,
        )

        if forecasts.frame.empty:
            prediction_summary = pd.DataFrame(
                columns=[
                    "experiment_name",
                    "generated_at_utc",
                    "signal_session",
                    "prediction_session",
                    "n_forecasts",
                    "n_longs",
                    "n_shorts",
                    "model_id",
                    "context_length",
                    "prediction_length",
                    "cross_learning",
                    "use_covariates",
                ]
            )
        else:
            prediction_summary = pd.DataFrame(
                [
                    {
                        "experiment_name": experiment_name,
                        "generated_at_utc": pd.Timestamp.utcnow(),
                        "signal_session": forecasts.frame["signal_session"].iloc[0],
                        "prediction_session": forecasts.frame["prediction_session"].iloc[0],
                        "n_forecasts": int(len(forecasts.frame)),
                        "n_longs": self.research_config.n_long,
                        "n_shorts": self.research_config.n_short,
                        "model_id": forecaster_config.model_id,
                        "context_length": forecaster_config.context_length,
                        "prediction_length": forecaster_config.prediction_length,
                        "cross_learning": forecaster_config.cross_learning,
                        "use_covariates": forecaster_config.use_covariates,
                    }
                ]
            )

        return LivePredictionResult(
            experiment_name=experiment_name,
            forecasts=forecasts,
            book=book,
            prediction_summary=prediction_summary,
        )

    def run_v2_next_session_prediction(
        self,
        v2_base_frame: pd.DataFrame,
        forecaster_config: ForecasterConfig,
        experiment_name: str,
        *,
        basket_mode: str = "full",
        book_mode: str = "tail_fraction",
        tail_fraction: float = 0.10,
        score_column: str = "score_z",
    ) -> LivePredictionResult:
        product_mode = forecaster_config.product_mode
        if product_mode == "winner_picker":
            self._validate_v2_winner_request(prediction_length=forecaster_config.prediction_length)
        else:
            self._validate_v2_ranker_request(prediction_length=forecaster_config.prediction_length)
        prediction_inputs, data_quality_events, data_quality_summary = self.v2_builder.build_prediction_inputs(
            v2_base_frame,
            context_length=forecaster_config.context_length,
            prediction_length=forecaster_config.prediction_length,
            basket_mode=basket_mode,
            latest_only=True,
            use_covariates=forecaster_config.use_covariates,
            beta_mode=forecaster_config.beta_mode,
        )
        freshness = self._build_v2_live_freshness_metadata(
            v2_base_frame,
            signal_session=prediction_inputs[0].signal_session if prediction_inputs else None,
        )
        self._assert_clean_future_prediction_window(
            freshness,
            signal_session=prediction_inputs[0].signal_session if prediction_inputs else None,
        )
        self._assert_prediction_session_after_current_market_day(
            prediction_inputs[0].prediction_session if prediction_inputs else None,
            market_timezone=self.research_config.market_timezone,
        )
        forecasts = self._predict_v2_baskets(
            prediction_inputs,
            forecaster_config,
            product_mode=product_mode,
            confirmation_mode=forecaster_config.confirmation_mode,
        )
        extra_outputs: dict[str, pd.DataFrame] = {}
        if product_mode == "winner_picker":
            book = self.portfolio_builder.select_winners(
                forecasts,
                score_column="winner_score",
                max_fraction=self._winner_max_picks_fraction(),
            )
            extra_outputs = self._build_winner_outputs(
                forecasts.frame,
                book.frame,
                include_realized=False,
                promotable_threshold=self.research_config.winner_min_promotable_days,
            )
        else:
            book = self.portfolio_builder.rank_and_size(
                forecasts,
                n_long=self.research_config.n_long,
                n_short=self.research_config.n_short,
                score_column=score_column,
                book_mode=book_mode,
                tail_fraction=tail_fraction,
            )

        if forecasts.frame.empty:
            base_columns = [
                "experiment_name",
                "generated_at_utc",
                "signal_session",
                "prediction_session",
                "prediction_session_end",
                "n_forecasts",
                "n_longs",
                "n_shorts",
                "model_id",
                "context_length",
                "prediction_length",
                "cross_learning",
                "use_covariates",
                "basket_mode",
                "beta_mode",
                "book_mode",
                "tail_fraction",
                "tail_size",
                "score_column",
                "product_mode",
                "confirmation_mode",
                "observed_latest_session",
                "has_newer_observed_session_than_signal",
                "prediction_freshness",
                "prediction_freshness_note",
            ]
            if product_mode == "winner_picker":
                base_columns.extend(["n_candidates", "n_buy_eligible", "n_confirmed_3d", "n_selected", "book_active"])
            prediction_summary = pd.DataFrame(columns=base_columns)
        else:
            if product_mode == "winner_picker":
                prediction_summary = self._build_winner_prediction_summary(
                    experiment_name=experiment_name,
                    forecasts=forecasts.frame,
                    book_frame=book.frame,
                    forecaster_config=forecaster_config,
                    basket_mode=basket_mode,
                    freshness=freshness,
                    winner_outputs=extra_outputs,
                )
            else:
                tail_size = int(book.frame["tail_size"].dropna().iloc[0]) if "tail_size" in book.frame.columns and not book.frame["tail_size"].dropna().empty else 0
                n_longs = int((book.frame["side"] == "long").sum())
                n_shorts = int((book.frame["side"] == "short").sum())
                prediction_summary = pd.DataFrame(
                    [
                        {
                            "experiment_name": experiment_name,
                            "generated_at_utc": pd.Timestamp.utcnow(),
                            "signal_session": forecasts.frame["signal_session"].iloc[0],
                            "prediction_session": forecasts.frame["prediction_session"].iloc[0],
                            "prediction_session_end": forecasts.frame["prediction_session_end"].iloc[0]
                            if "prediction_session_end" in forecasts.frame.columns
                            else forecasts.frame["prediction_session"].iloc[0],
                            "n_forecasts": int(len(forecasts.frame)),
                            "n_longs": n_longs,
                            "n_shorts": n_shorts,
                            "model_id": forecaster_config.model_id,
                            "context_length": forecaster_config.context_length,
                            "prediction_length": forecaster_config.prediction_length,
                            "cross_learning": forecaster_config.cross_learning,
                            "use_covariates": forecaster_config.use_covariates,
                            "basket_mode": basket_mode,
                            "beta_mode": forecaster_config.beta_mode,
                            "book_mode": book_mode,
                            "tail_fraction": tail_fraction,
                            "tail_size": tail_size,
                            "score_column": score_column,
                            "product_mode": product_mode,
                            "confirmation_mode": forecaster_config.confirmation_mode,
                            **freshness,
                        }
                    ]
                )

        return LivePredictionResult(
            experiment_name=experiment_name,
            forecasts=forecasts,
            book=book,
            prediction_summary=prediction_summary,
            data_quality_events=data_quality_events,
            data_quality_summary=data_quality_summary,
            extra_outputs=extra_outputs,
        )

    def run_baselines(self, labeled_frame: pd.DataFrame) -> dict[str, ExperimentResult]:
        realized = self.baseline_scorer.build_realized_daily_returns(labeled_frame)
        baseline_scores = self.baseline_scorer.score(labeled_frame)
        results: dict[str, ExperimentResult] = {}
        for baseline_name, baseline_frame in baseline_scores.groupby("baseline_name", sort=True):
            forecast_batch = ForecastBatch(frame=baseline_frame.reset_index(drop=True))
            rank_ic_summary = self.compute_rank_ic(forecast_batch.frame, realized)
            book = self.portfolio_builder.rank_and_size(
                forecast_batch,
                n_long=self.research_config.n_long,
                n_short=self.research_config.n_short,
            )
            metrics = self.backtest_runner.run(book, realized, self.research_config.costs_bps)
            results[baseline_name] = ExperimentResult(
                experiment_name=baseline_name,
                forecasts=forecast_batch,
                book=book,
                metrics=metrics,
                rank_ic_summary=rank_ic_summary,
            )
        return results

    def run_matrix(self, labeled_frame: pd.DataFrame) -> dict[str, ExperimentResult]:
        results: dict[str, ExperimentResult] = {}
        model_slug = _model_slug(self.research_config.model_id)
        for context_length, cross_learning, use_covariates in product(
            self.research_config.candidate_context_lengths,
            self.research_config.candidate_cross_learning,
            self.research_config.candidate_use_covariates,
        ):
            config = ForecasterConfig(
                model_id=self.research_config.model_id,
                prediction_length=self.research_config.prediction_length,
                context_length=context_length,
                quantiles=self.research_config.quantiles,
                cross_learning=cross_learning,
                use_covariates=use_covariates,
                batch_size=self.research_config.batch_size,
            )
            experiment_name = f"{model_slug}_ctx{context_length}_cross{int(cross_learning)}_cov{int(use_covariates)}"
            results[experiment_name] = self.run_experiment(labeled_frame, config, experiment_name)
        return results

    def run_v2_matrix(
        self,
        v2_base_frame: pd.DataFrame,
        *,
        basket_mode: str = "full",
        beta_mode: str = "unit_beta_1.0",
        book_mode: str = "tail_fraction",
        tail_fraction: float = 0.10,
        score_column: str = "score_z",
    ) -> dict[str, ExperimentResult]:
        results: dict[str, ExperimentResult] = {}
        model_slug = _model_slug(self.research_config.model_id)
        for context_length in self.research_config.candidate_context_lengths:
            config = ForecasterConfig(
                model_id=self.research_config.model_id,
                prediction_length=self.research_config.prediction_length,
                context_length=context_length,
                quantiles=self.research_config.quantiles,
                cross_learning=False,
                use_covariates=True,
                batch_size=self.research_config.batch_size,
                beta_mode=beta_mode,
            )
            experiment_name = (
                f"v2_flagship_basket_raw_{basket_mode}_{model_slug}_ctx{context_length}_cross0_cov1"
            )
            results[experiment_name] = self.run_v2_experiment(
                v2_base_frame,
                config,
                experiment_name,
                basket_mode=basket_mode,
                book_mode=book_mode,
                tail_fraction=tail_fraction,
                score_column=score_column,
            )
        return results

    def run_v2_winner_matrix(
        self,
        v2_base_frame: pd.DataFrame,
        *,
        basket_mode: str = "expanded40",
    ) -> dict[str, ExperimentResult]:
        results: dict[str, ExperimentResult] = {}
        model_slug = _model_slug(self.research_config.model_id)
        for context_length, confirmation_mode, cross_learning, beta_mode in product(
            self.research_config.candidate_winner_context_lengths,
            self.research_config.candidate_winner_confirmation_modes,
            self.research_config.candidate_cross_learning,
            ("unit_beta_1.0", "rolling_session_beta_20d"),
        ):
            config = ForecasterConfig(
                model_id=self.research_config.model_id,
                prediction_length=self.research_config.winner_prediction_length,
                context_length=context_length,
                quantiles=self.research_config.winner_quantiles,
                cross_learning=cross_learning,
                use_covariates=True,
                batch_size=self.research_config.batch_size,
                beta_mode=beta_mode,
                product_mode="winner_picker",
                confirmation_mode=confirmation_mode,
            )
            experiment_name = (
                f"winner_picker_{basket_mode}_{model_slug}_ctx{context_length}_"
                f"confirm{confirmation_mode}_cross{int(cross_learning)}_beta{beta_mode}"
            )
            results[experiment_name] = self.run_v2_experiment(
                v2_base_frame,
                config,
                experiment_name,
                basket_mode=basket_mode,
            )
        return results

    def analyze_signal_run(
        self,
        *,
        run_dir: Path,
        experiment: str,
        output_dir: Path,
        score_column: str = "score_z",
        tail_fractions: tuple[float, ...] = (0.10, 0.15, 0.20),
        execution_interval: str = "5m",
        execution_mode: str = "strict",
        default_borrow_fee_annual_bps: float | None = None,
        borrow_profile_path: Path | None = None,
        unavailable_tickers: tuple[str, ...] = (),
    ) -> None:
        labeled_frame = pd.read_csv(run_dir / "labeled_frame.csv")
        with (run_dir / experiment / "forecasts.json").open("r", encoding="utf-8") as handle:
            forecasts = pd.DataFrame(json.load(handle))
        borrow_profile = pd.read_csv(borrow_profile_path) if borrow_profile_path is not None else None
        borrow_fee_annual_bps = (
            float(default_borrow_fee_annual_bps)
            if default_borrow_fee_annual_bps is not None
            else float(self.research_config.default_short_borrow_fee_annual_bps)
        )

        signal_outputs = generate_signal_execution_analysis(
            forecasts,
            labeled_frame=labeled_frame,
            execution_score_column=score_column,
            score_behavior_columns=("score_plain", "score_z"),
            benchmark_ticker=self.research_config.benchmark_ticker,
            n_long=self.research_config.n_long,
            n_short=self.research_config.n_short,
            costs_bps=self.research_config.costs_bps,
            tail_fractions=tail_fractions,
            book_mode=self.research_config.default_v2_book_mode,
            tail_fraction=self.research_config.default_v2_tail_fraction,
            default_borrow_fee_annual_bps=borrow_fee_annual_bps,
            borrow_profile=borrow_profile,
            unavailable_tickers=unavailable_tickers,
        )

        baseline_book = signal_outputs["signal_execution_baseline_book"]
        if baseline_book.empty:
            execution_outputs = generate_near_open_execution_analysis(
                book_frame=baseline_book,
                raw_execution_bars=pd.DataFrame(),
                benchmark_ticker=self.research_config.benchmark_ticker,
                costs_bps=self.research_config.costs_bps,
                execution_mode=execution_mode,
            )
        else:
            prediction_sessions = sorted(pd.to_datetime(baseline_book["prediction_session"], utc=True).unique().tolist())
            start = pd.Timestamp(min(prediction_sessions)).date().isoformat()
            end = (pd.Timestamp(max(prediction_sessions)) + pd.Timedelta(days=1)).date().isoformat()
            execution_tickers = sorted(set(baseline_book["ticker"].tolist()) | {self.research_config.benchmark_ticker})
            raw_execution_bars = self.provider.fetch(
                tickers=execution_tickers,
                start=start,
                end=end,
                interval=execution_interval,
                regular_only=True,
                auto_adjust=True,
            )
            execution_outputs = generate_near_open_execution_analysis(
                book_frame=baseline_book,
                raw_execution_bars=raw_execution_bars,
                benchmark_ticker=self.research_config.benchmark_ticker,
                costs_bps=self.research_config.costs_bps,
                execution_mode=execution_mode,
            )
        outputs = {
            "analysis_config": pd.DataFrame(
                [
                    {
                        "run_dir": str(run_dir),
                        "experiment": experiment,
                        "score_column": score_column,
                        "book_mode": self.research_config.default_v2_book_mode,
                        "tail_fraction": self.research_config.default_v2_tail_fraction,
                        "beta_mode": str(forecasts["beta_mode"].iloc[0]) if "beta_mode" in forecasts.columns and not forecasts.empty else "unknown",
                        "tail_fractions": ",".join(str(value) for value in tail_fractions),
                        "execution_interval": execution_interval,
                        "execution_mode": execution_mode,
                        "default_borrow_fee_annual_bps": borrow_fee_annual_bps,
                        "borrow_profile_path": str(borrow_profile_path) if borrow_profile_path is not None else "",
                        "borrow_unavailable_tickers": ",".join(sorted(set(unavailable_tickers))),
                        "execution_mode_label": STRICT_EXECUTION_MODE_LABEL,
                        "exact_open_entry_formula": EXACT_OPEN_ENTRY_FORMULA,
                        "first_5m_avg_entry_formula": FIRST_5M_ENTRY_FORMULA,
                        "first_15m_avg_entry_formula": FIRST_15M_ENTRY_FORMULA,
                        "execution_exit_formula": EXECUTION_EXIT_FORMULA,
                        "execution_alpha_formula": EXECUTION_ALPHA_FORMULA,
                        "benchmark_ticker": self.research_config.benchmark_ticker,
                    }
                ]
            ),
            **signal_outputs,
            **execution_outputs,
        }
        outputs["backtest_scorecard"] = build_backtest_scorecard(
            forecasts=forecasts,
            signal_outputs=signal_outputs,
            execution_outputs=execution_outputs,
        )
        write_diagnostic_outputs(output_dir, outputs)

    def reconcile_live_prediction(
        self,
        *,
        prediction_dir: Path,
        ledger_dir: Path,
        execution_interval: str = "5m",
        execution_mode: str = "strict",
        now: pd.Timestamp | None = None,
    ) -> None:
        with (prediction_dir / "forecasts.json").open("r", encoding="utf-8") as handle:
            forecasts = pd.DataFrame(json.load(handle))
        book = pd.read_csv(prediction_dir / "book.csv")
        prediction_summary = pd.read_csv(prediction_dir / "prediction_summary.csv")

        product_mode = self._detect_product_mode(prediction_summary=prediction_summary, forecasts=forecasts)
        if forecasts.empty or prediction_summary.empty or (product_mode != "winner_picker" and book.empty):
            raise ValueError("Prediction directory is missing a usable forecasts/book/prediction summary payload.")

        prediction_sessions = self._prediction_sessions_from_payload(
            prediction_summary=prediction_summary,
            book=book,
            forecasts=forecasts,
        )
        is_ready, readiness_message = self.reconciliation_readiness(
            prediction_sessions,
            market_timezone=self.research_config.market_timezone,
            now=now,
        )
        if not is_ready:
            raise ValueError(readiness_message)
        start = pd.Timestamp(min(prediction_sessions)).date().isoformat()
        end = (pd.Timestamp(max(prediction_sessions)) + pd.Timedelta(days=1)).date().isoformat()
        forecast_tickers = sorted(set(forecasts["ticker"].tolist()) | {self.research_config.benchmark_ticker})
        execution_tickers = sorted(set(book["ticker"].tolist()) | {self.research_config.benchmark_ticker}) if not book.empty else []

        labeled_frame = self.prepare_labeled_frame(
            tickers=forecast_tickers,
            start=start,
            end=end,
            interval="60m",
        )
        if product_mode == "winner_picker":
            realized_forecasts = self._realize_winner_forecasts(
                forecasts=forecasts,
                labeled_frame=labeled_frame,
            )
            winner_outputs = self._build_winner_outputs(
                realized_forecasts,
                book,
                include_realized=True,
                promotable_threshold=self.research_config.winner_min_promotable_days,
            )

            ledger_dir.mkdir(parents=True, exist_ok=True)
            prediction_summary_history = prediction_summary.copy()
            if "product_mode" not in prediction_summary_history.columns:
                prediction_summary_history["product_mode"] = product_mode
            prediction_summary_history["prediction_dir"] = str(prediction_dir)
            prediction_summary_history["execution_interval"] = execution_interval
            prediction_summary_history["execution_mode"] = execution_mode

            cumulative_prediction_summary = self._upsert_csv(
                ledger_dir / "prediction_summary_history.csv",
                prediction_summary_history,
                dedupe_keys=["prediction_session", "product_mode"],
                time_columns=["signal_session", "prediction_session", "generated_at_utc", "prediction_session_end"],
                sort_columns=["prediction_session"],
            )
            cumulative_forecasts = self._upsert_csv(
                ledger_dir / "live_reconciled_forecasts.csv",
                realized_forecasts,
                dedupe_keys=["ticker", "prediction_session"],
                time_columns=["signal_session", "prediction_session", "prediction_session_end"],
                sort_columns=["prediction_session", "ticker"],
            )
            cumulative_winner_board = self._upsert_csv(
                ledger_dir / "live_winner_board.csv",
                winner_outputs["winner_board"],
                dedupe_keys=["ticker", "prediction_session"],
                time_columns=["signal_session", "prediction_session", "prediction_session_end"],
                sort_columns=["prediction_session", "ticker"],
            )
            cumulative_winner_positions = self._upsert_csv(
                ledger_dir / "live_winner_positions.csv",
                winner_outputs["winner_daily_positions"],
                dedupe_keys=["prediction_session", "ticker"],
                time_columns=["signal_session", "prediction_session", "prediction_session_end"],
                sort_columns=["prediction_session", "rank", "ticker"],
            )
            cumulative_winner_daily = self._upsert_csv(
                ledger_dir / "live_winner_daily_summary.csv",
                winner_outputs["winner_daily_summary"],
                dedupe_keys=["prediction_session"],
                time_columns=["signal_session", "prediction_session", "prediction_session_end"],
                sort_columns=["prediction_session"],
            )
            live_winner_summary = self._summarize_cumulative_winner_ledger(
                winner_positions=cumulative_winner_positions,
                winner_daily_summary=cumulative_winner_daily,
                promotable_threshold=self.research_config.winner_min_promotable_days,
            )
            live_winner_summary.to_csv(ledger_dir / "live_winner_backtest_summary.csv", index=False)

            live_winner_rank_ic = self.compute_rank_ic(
                cumulative_forecasts,
                self._realized_from_forecasts(cumulative_forecasts),
                score_column="winner_score",
            )
            live_winner_rank_ic.to_csv(ledger_dir / "live_winner_rank_ic.csv", index=False)

            if execution_tickers:
                raw_execution_bars = self.provider.fetch(
                    tickers=execution_tickers,
                    start=start,
                    end=end,
                    interval=execution_interval,
                    regular_only=True,
                    auto_adjust=True,
                )
                execution_outputs = generate_near_open_execution_analysis(
                    book_frame=book,
                    raw_execution_bars=raw_execution_bars,
                    benchmark_ticker=self.research_config.benchmark_ticker,
                    costs_bps=self.research_config.costs_bps,
                    execution_mode=execution_mode,
                )
                self._upsert_csv(
                    ledger_dir / "live_winner_execution_position_returns.csv",
                    execution_outputs["execution_position_returns"],
                    dedupe_keys=["execution_mode", "prediction_session", "ticker", "side"],
                    time_columns=["signal_session", "prediction_session"],
                    sort_columns=["execution_mode", "prediction_session", "rank", "ticker"],
                )
                self._upsert_csv(
                    ledger_dir / "live_winner_execution_daily_returns.csv",
                    execution_outputs["execution_daily_returns"],
                    dedupe_keys=["execution_mode", "prediction_session"],
                    time_columns=["prediction_session"],
                    sort_columns=["execution_mode", "prediction_session"],
                )
                self._upsert_csv(
                    ledger_dir / "live_winner_execution_data_quality.csv",
                    execution_outputs["execution_data_quality"],
                    dedupe_keys=["prediction_session", "ticker", "execution_mode", "event_type", "reason"],
                    time_columns=["prediction_session"],
                    sort_columns=["prediction_session", "ticker"],
                )

            reconciliation_summary = pd.DataFrame(
                [
                    {
                        "prediction_dir": str(prediction_dir),
                        "ledger_dir": str(ledger_dir),
                        "product_mode": product_mode,
                        "prediction_days_reconciled_total": int(cumulative_prediction_summary["prediction_session"].nunique()),
                        "forecast_rows_total": int(len(cumulative_forecasts)),
                        "book_positions_total": int(len(cumulative_winner_positions)),
                        "book_days_total": int(cumulative_winner_daily["prediction_session"].nunique()) if not cumulative_winner_daily.empty else 0,
                        "execution_days_total": 0,
                        "execution_interval": execution_interval,
                        "execution_mode": execution_mode,
                        "benchmark_ticker": self.research_config.benchmark_ticker,
                    }
                ]
            )
            reconciliation_summary.to_csv(ledger_dir / "reconciliation_summary.csv", index=False)
            return
        raw_execution_bars = self.provider.fetch(
            tickers=execution_tickers,
            start=start,
            end=end,
            interval=execution_interval,
            regular_only=True,
            auto_adjust=True,
        )
        live_outputs = generate_live_reconciliation_outputs(
            forecasts=forecasts,
            book_frame=book,
            labeled_frame=labeled_frame,
            raw_execution_bars=raw_execution_bars,
            benchmark_ticker=self.research_config.benchmark_ticker,
            costs_bps=self.research_config.costs_bps,
            execution_mode=execution_mode,
        )

        ledger_dir.mkdir(parents=True, exist_ok=True)
        prediction_summary_history = prediction_summary.copy()
        if "product_mode" not in prediction_summary_history.columns:
            prediction_summary_history["product_mode"] = product_mode
        prediction_summary_history["prediction_dir"] = str(prediction_dir)
        prediction_summary_history["execution_interval"] = execution_interval
        prediction_summary_history["execution_mode"] = execution_mode

        cumulative_prediction_summary = self._upsert_csv(
            ledger_dir / "prediction_summary_history.csv",
            prediction_summary_history,
            dedupe_keys=["prediction_session", "product_mode"],
            time_columns=["signal_session", "prediction_session", "generated_at_utc"],
            sort_columns=["prediction_session"],
        )
        cumulative_forecasts = self._upsert_csv(
            ledger_dir / "live_reconciled_forecasts.csv",
            live_outputs["reconciled_forecasts"],
            dedupe_keys=["ticker", "prediction_session"],
            time_columns=["signal_session", "prediction_session"],
            sort_columns=["prediction_session", "ticker"],
        )
        cumulative_position_returns = self._upsert_csv(
            ledger_dir / "live_position_returns.csv",
            live_outputs["live_position_returns"],
            dedupe_keys=["prediction_session", "ticker", "side"],
            time_columns=["signal_session", "prediction_session"],
            sort_columns=["prediction_session", "side", "rank", "ticker"],
        )
        cumulative_scoreboard_positions = self._upsert_csv(
            ledger_dir / "live_scoreboard_positions.csv",
            live_outputs["live_scoreboard_positions"],
            dedupe_keys=["prediction_session", "ticker", "side"],
            time_columns=["signal_session", "prediction_session"],
            sort_columns=["prediction_session", "side", "rank", "ticker"],
        )
        cumulative_scoreboard_daily = self._upsert_csv(
            ledger_dir / "live_scoreboard_daily.csv",
            live_outputs["live_scoreboard_daily"],
            dedupe_keys=["prediction_session"],
            time_columns=["signal_session", "prediction_session"],
            sort_columns=["prediction_session"],
        )
        cumulative_execution_positions = self._upsert_csv(
            ledger_dir / "live_execution_position_returns.csv",
            live_outputs["execution_position_returns"],
            dedupe_keys=["execution_mode", "prediction_session", "ticker", "side"],
            time_columns=["signal_session", "prediction_session"],
            sort_columns=["execution_mode", "prediction_session", "side", "rank", "ticker"],
        )
        cumulative_execution_daily = self._upsert_csv(
            ledger_dir / "live_execution_daily_returns.csv",
            live_outputs["execution_daily_returns"],
            dedupe_keys=["execution_mode", "prediction_session"],
            time_columns=["prediction_session"],
            sort_columns=["execution_mode", "prediction_session"],
        )
        cumulative_execution_quality = self._upsert_csv(
            ledger_dir / "live_execution_data_quality.csv",
            live_outputs["execution_data_quality"],
            dedupe_keys=["prediction_session", "ticker", "execution_mode", "event_type", "reason"],
            time_columns=["prediction_session"],
            sort_columns=["prediction_session", "ticker"],
        )

        live_rank_ic = self.compute_rank_ic(
            cumulative_forecasts,
            self._realized_from_forecasts(cumulative_forecasts),
        )
        live_rank_ic.to_csv(ledger_dir / "live_rank_ic.csv", index=False)

        live_book = self._book_from_position_returns(cumulative_position_returns)
        live_realized = self._realized_from_position_returns(cumulative_position_returns)
        live_metrics = self.backtest_runner.run(live_book, live_realized, self.research_config.costs_bps)
        live_metrics.summary.to_csv(ledger_dir / "live_metrics_summary.csv", index=False)
        live_metrics.daily_returns.to_csv(ledger_dir / "live_daily_returns.csv", index=False)

        live_execution_summary = summarize_execution_daily_returns(
            cumulative_execution_daily,
            costs_bps=self.research_config.costs_bps,
        )
        live_execution_summary.to_csv(ledger_dir / "live_execution_sensitivity_comparison.csv", index=False)

        reconciliation_summary = pd.DataFrame(
            [
                {
                    "prediction_dir": str(prediction_dir),
                    "ledger_dir": str(ledger_dir),
                    "prediction_days_reconciled_total": int(cumulative_prediction_summary["prediction_session"].nunique()),
                    "forecast_rows_total": int(len(cumulative_forecasts)),
                    "book_positions_total": int(len(cumulative_scoreboard_positions)),
                    "book_days_total": int(cumulative_scoreboard_daily["prediction_session"].nunique()) if not cumulative_scoreboard_daily.empty else 0,
                    "execution_days_total": int(cumulative_execution_daily["prediction_session"].nunique()) if not cumulative_execution_daily.empty else 0,
                    "execution_interval": execution_interval,
                    "execution_mode": execution_mode,
                    "benchmark_ticker": self.research_config.benchmark_ticker,
                }
            ]
        )
        reconciliation_summary.to_csv(ledger_dir / "reconciliation_summary.csv", index=False)

    @staticmethod
    def _realized_from_forecasts(forecast_frame: pd.DataFrame) -> pd.DataFrame:
        if forecast_frame.empty:
            return pd.DataFrame(columns=["ticker", "prediction_session", "realized_day_return"])
        return (
            forecast_frame[["ticker", "prediction_session", "realized_day_return"]]
            .drop_duplicates(subset=["ticker", "prediction_session"])
            .reset_index(drop=True)
        )

    @staticmethod
    def _normalize_time_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        normalized = frame.copy()
        for column in columns:
            if column in normalized.columns:
                normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="coerce")
        return normalized

    @classmethod
    def _upsert_csv(
        cls,
        path: Path,
        new_frame: pd.DataFrame,
        *,
        dedupe_keys: list[str],
        time_columns: list[str],
        sort_columns: list[str],
    ) -> pd.DataFrame:
        incoming = cls._normalize_time_columns(new_frame.copy(), time_columns)
        if path.exists():
            existing = pd.read_csv(path)
            existing = cls._normalize_time_columns(existing, time_columns)
            combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
        else:
            combined = incoming
        if combined.empty:
            combined.to_csv(path, index=False)
            return combined
        combined = cls._normalize_time_columns(combined, time_columns)
        combined = combined.drop_duplicates(subset=dedupe_keys, keep="last")
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
        combined.to_csv(path, index=False)
        return combined

    @staticmethod
    def _book_from_position_returns(position_returns: pd.DataFrame) -> pd.DataFrame:
        if position_returns.empty:
            return pd.DataFrame()
        book_columns = [
            column
            for column in [
                "signal_session",
                "prediction_session",
                "ticker",
                "side",
                "rank",
                "weight",
                "ranking_score_column",
                "ranking_score",
                "book_mode",
                "tail_fraction",
                "tail_size",
                "basket_id",
                "score_plain",
                "score_risk",
                "score_z",
                "score_q50",
                "beta_mode",
            ]
            if column in position_returns.columns
        ]
        return position_returns[book_columns].drop_duplicates(subset=["prediction_session", "ticker", "side"]).reset_index(drop=True)

    @staticmethod
    def _realized_from_position_returns(position_returns: pd.DataFrame) -> pd.DataFrame:
        if position_returns.empty:
            return pd.DataFrame(columns=["ticker", "prediction_session", "realized_day_return"])
        realized_column = "next_day_realized_alpha" if "next_day_realized_alpha" in position_returns.columns else "realized_day_return"
        return (
            position_returns[["ticker", "prediction_session", realized_column]]
            .drop_duplicates(subset=["ticker", "prediction_session"])
            .rename(columns={realized_column: "realized_day_return"})
            .reset_index(drop=True)
        )

    @staticmethod
    def write_result(output_dir: Path, result: ExperimentResult) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ZeroShotResearchPipeline._sanitize_generated_output(result.forecasts.frame).to_json(
            output_dir / "forecasts.json",
            orient="records",
            date_format="iso",
            indent=2,
        )
        ZeroShotResearchPipeline._sanitize_generated_output(result.book.frame).to_csv(output_dir / "book.csv", index=False)
        result.metrics.summary.to_csv(output_dir / "metrics_summary.csv", index=False)
        result.metrics.daily_returns.to_csv(output_dir / "daily_returns.csv", index=False)
        result.rank_ic_summary.to_csv(output_dir / "rank_ic.csv", index=False)
        if len(result.data_quality_events.columns) > 0:
            result.data_quality_events.to_csv(output_dir / "data_quality_events.csv", index=False)
        if len(result.data_quality_summary.columns) > 0:
            result.data_quality_summary.to_csv(output_dir / "data_quality_summary.csv", index=False)
        for name, frame in result.extra_outputs.items():
            ZeroShotResearchPipeline._sanitize_generated_output(frame).to_csv(output_dir / f"{name}.csv", index=False)

    @staticmethod
    def write_live_prediction(output_dir: Path, result: LivePredictionResult) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ZeroShotResearchPipeline._sanitize_generated_output(result.forecasts.frame).to_json(
            output_dir / "forecasts.json",
            orient="records",
            date_format="iso",
            indent=2,
        )
        ZeroShotResearchPipeline._sanitize_generated_output(result.book.frame).to_csv(output_dir / "book.csv", index=False)
        result.prediction_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
        if len(result.data_quality_events.columns) > 0:
            result.data_quality_events.to_csv(output_dir / "data_quality_events.csv", index=False)
        if len(result.data_quality_summary.columns) > 0:
            result.data_quality_summary.to_csv(output_dir / "data_quality_summary.csv", index=False)
        for name, frame in result.extra_outputs.items():
            ZeroShotResearchPipeline._sanitize_generated_output(frame).to_csv(output_dir / f"{name}.csv", index=False)
