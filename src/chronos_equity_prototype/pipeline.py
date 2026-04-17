from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Iterable

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

    def compute_rank_ic(self, forecasts: pd.DataFrame, realized_returns: pd.DataFrame) -> pd.DataFrame:
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

        daily_ic_records: list[dict[str, object]] = []
        for prediction_session, day_frame in merged.groupby("prediction_session", sort=True):
            ranked_scores = day_frame["score_q50"].rank(method="average")
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

    def run_experiment(
        self,
        labeled_frame: pd.DataFrame,
        forecaster_config: ForecasterConfig,
        experiment_name: str,
    ) -> ExperimentResult:
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
        prediction_inputs, data_quality_events, data_quality_summary = self.v2_builder.build_prediction_inputs(
            v2_base_frame,
            context_length=forecaster_config.context_length,
            prediction_length=forecaster_config.prediction_length,
            basket_mode=basket_mode,
            latest_only=False,
            use_covariates=forecaster_config.use_covariates,
            beta_mode=forecaster_config.beta_mode,
        )
        forecasts = self._predict_v2_baskets(prediction_inputs, forecaster_config)
        realized = self._realized_from_forecasts(forecasts.frame)
        rank_ic_summary = self.compute_rank_ic(forecasts.frame, realized)
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
        )

    def run_next_session_prediction(
        self,
        labeled_frame: pd.DataFrame,
        forecaster_config: ForecasterConfig,
        experiment_name: str,
    ) -> LivePredictionResult:
        tasks = self.task_builder.build_latest_tasks(
            labeled_frame,
            context_length=forecaster_config.context_length,
            prediction_length=forecaster_config.prediction_length,
            use_covariates=forecaster_config.use_covariates,
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
        prediction_inputs, data_quality_events, data_quality_summary = self.v2_builder.build_prediction_inputs(
            v2_base_frame,
            context_length=forecaster_config.context_length,
            prediction_length=forecaster_config.prediction_length,
            basket_mode=basket_mode,
            latest_only=True,
            use_covariates=forecaster_config.use_covariates,
            beta_mode=forecaster_config.beta_mode,
        )
        forecasts = self._predict_v2_baskets(prediction_inputs, forecaster_config)
        book = self.portfolio_builder.rank_and_size(
            forecasts,
            n_long=self.research_config.n_long,
            n_short=self.research_config.n_short,
            score_column=score_column,
            book_mode=book_mode,
            tail_fraction=tail_fraction,
        )
        freshness = self._build_v2_live_freshness_metadata(
            v2_base_frame,
            signal_session=forecasts.frame["signal_session"].iloc[0] if not forecasts.frame.empty else None,
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
                    "basket_mode",
                    "beta_mode",
                    "book_mode",
                    "tail_fraction",
                    "tail_size",
                    "score_column",
                    "observed_latest_session",
                    "has_newer_observed_session_than_signal",
                    "prediction_freshness",
                    "prediction_freshness_note",
                ]
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
    ) -> None:
        with (prediction_dir / "forecasts.json").open("r", encoding="utf-8") as handle:
            forecasts = pd.DataFrame(json.load(handle))
        book = pd.read_csv(prediction_dir / "book.csv")
        prediction_summary = pd.read_csv(prediction_dir / "prediction_summary.csv")

        if forecasts.empty or book.empty or prediction_summary.empty:
            raise ValueError("Prediction directory is missing a usable forecasts/book/prediction summary payload.")

        prediction_sessions = sorted(pd.to_datetime(book["prediction_session"], utc=True).unique().tolist())
        start = pd.Timestamp(min(prediction_sessions)).date().isoformat()
        end = (pd.Timestamp(max(prediction_sessions)) + pd.Timedelta(days=1)).date().isoformat()
        forecast_tickers = sorted(set(forecasts["ticker"].tolist()) | {self.research_config.benchmark_ticker})
        execution_tickers = sorted(set(book["ticker"].tolist()) | {self.research_config.benchmark_ticker})

        labeled_frame = self.prepare_labeled_frame(
            tickers=forecast_tickers,
            start=start,
            end=end,
            interval="60m",
        )
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
        prediction_summary_history["prediction_dir"] = str(prediction_dir)
        prediction_summary_history["execution_interval"] = execution_interval
        prediction_summary_history["execution_mode"] = execution_mode

        cumulative_prediction_summary = self._upsert_csv(
            ledger_dir / "prediction_summary_history.csv",
            prediction_summary_history,
            dedupe_keys=["prediction_session"],
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
        result.forecasts.frame.to_json(output_dir / "forecasts.json", orient="records", date_format="iso", indent=2)
        result.book.frame.to_csv(output_dir / "book.csv", index=False)
        result.metrics.summary.to_csv(output_dir / "metrics_summary.csv", index=False)
        result.metrics.daily_returns.to_csv(output_dir / "daily_returns.csv", index=False)
        result.rank_ic_summary.to_csv(output_dir / "rank_ic.csv", index=False)
        if len(result.data_quality_events.columns) > 0:
            result.data_quality_events.to_csv(output_dir / "data_quality_events.csv", index=False)
        if len(result.data_quality_summary.columns) > 0:
            result.data_quality_summary.to_csv(output_dir / "data_quality_summary.csv", index=False)

    @staticmethod
    def write_live_prediction(output_dir: Path, result: LivePredictionResult) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.forecasts.frame.to_json(output_dir / "forecasts.json", orient="records", date_format="iso", indent=2)
        result.book.frame.to_csv(output_dir / "book.csv", index=False)
        result.prediction_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
        if len(result.data_quality_events.columns) > 0:
            result.data_quality_events.to_csv(output_dir / "data_quality_events.csv", index=False)
        if len(result.data_quality_summary.columns) > 0:
            result.data_quality_summary.to_csv(output_dir / "data_quality_summary.csv", index=False)
