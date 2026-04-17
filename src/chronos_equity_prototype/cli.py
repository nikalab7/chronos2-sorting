from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_LIQUID_SP500,
    DEFAULT_V2_BETA_MODE,
    DEFAULT_V2_BOOK_MODE,
    DEFAULT_V2_TAIL_FRACTION,
    ForecasterConfig,
    ResearchConfig,
)
from .pipeline import ZeroShotResearchPipeline
from .reporting import default_chronos_output_dir, export_chronos_package
from .v2_basket import BasketV2Builder


def _model_slug(model_id: str) -> str:
    slug = model_id.strip().lower().replace("/", "_").replace("-", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in slug)


def _parse_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if args.tickers_file:
        content = Path(args.tickers_file).read_text(encoding="utf-8")
        return [line.strip().upper() for line in content.splitlines() if line.strip()]
    return list(DEFAULT_LIQUID_SP500)


def _resolve_v2_tickers(base_tickers: list[str], args: argparse.Namespace) -> list[str]:
    if getattr(args, "architecture", "v1_residual") != "v2_flagship_basket_raw":
        return base_tickers
    basket_tickers = BasketV2Builder().required_tickers(args.basket_mode)
    return list(dict.fromkeys([*base_tickers, *basket_tickers]))


def _default_start_end() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=59)
    return start.isoformat(), end.isoformat()


def _default_prediction_window() -> tuple[str, str]:
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=59)
    return start.isoformat(), end.isoformat()


def _resolve_context_length(args: argparse.Namespace, research_config: ResearchConfig) -> int:
    if args.context_length is not None:
        return args.context_length
    if getattr(args, "architecture", "v1_residual") == "v2_flagship_basket_raw":
        return research_config.default_v2_context_length
    return research_config.default_context_length


def _parse_tail_fractions(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _build_experiment_name(args: argparse.Namespace) -> str:
    model_slug = _model_slug(args.model_id)
    if args.architecture == "v2_flagship_basket_raw":
        return (
            f"{args.architecture}_{args.basket_mode}_{model_slug}_ctx{args.context_length}_"
            f"cross{int(args.cross_learning)}_cov{int(not args.no_covariates)}"
        )
    return f"{model_slug}_ctx{args.context_length}_cross{int(args.cross_learning)}_cov{int(not args.no_covariates)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the zero-shot Chronos-2 equity prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the default experiment or the ablation matrix.")
    run_parser.add_argument("--tickers", help="Comma-separated tickers. Defaults to a liquid S&P 500 sample.")
    run_parser.add_argument("--tickers-file", help="Path to a newline-delimited ticker file.")
    run_parser.add_argument("--start", help="Inclusive start date in YYYY-MM-DD format.")
    run_parser.add_argument("--end", help="Exclusive end date in YYYY-MM-DD format.")
    run_parser.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")
    run_parser.add_argument("--model-id", default="amazon/chronos-2")
    run_parser.add_argument("--prediction-length", type=int, default=7)
    run_parser.add_argument("--context-length", type=int)
    run_parser.add_argument("--batch-size", type=int, default=8)
    run_parser.add_argument("--n-long", type=int, default=5)
    run_parser.add_argument("--n-short", type=int, default=5)
    run_parser.add_argument("--benchmark", default="SPY")
    run_parser.add_argument("--cross-learning", action="store_true", help="Enable Chronos-2 cross-learning.")
    run_parser.add_argument("--no-covariates", action="store_true", help="Run the pure-univariate variant.")
    run_parser.add_argument("--matrix", action="store_true", help="Run the plan ablation matrix.")
    run_parser.add_argument(
        "--architecture",
        default="v1_residual",
        choices=("v1_residual", "v2_flagship_basket_raw"),
        help="Forecast architecture to run.",
    )
    run_parser.add_argument(
        "--basket-mode",
        default="full",
        choices=("full", "growth_spike", "expanded40", "expanded60"),
        help="V2 basket rollout mode.",
    )
    run_parser.add_argument(
        "--beta-mode",
        default=DEFAULT_V2_BETA_MODE,
        choices=("unit_beta_1.0", "pooled_beta_140"),
        help="V2 beta handling mode.",
    )
    run_parser.add_argument(
        "--book-mode",
        default=DEFAULT_V2_BOOK_MODE,
        choices=("tail_fraction", "fixed_count"),
        help="Book construction mode for V2.",
    )
    run_parser.add_argument(
        "--tail-fraction",
        type=float,
        default=DEFAULT_V2_TAIL_FRACTION,
        help="Tail fraction for V2 tail-only books.",
    )

    predict_parser = subparsers.add_parser(
        "predict-next",
        help="Forecast the next regular session from the latest complete market session.",
    )
    predict_parser.add_argument("--tickers", help="Comma-separated tickers. Defaults to a liquid S&P 500 sample.")
    predict_parser.add_argument("--tickers-file", help="Path to a newline-delimited ticker file.")
    predict_parser.add_argument("--start", help="Inclusive start date in YYYY-MM-DD format.")
    predict_parser.add_argument("--end", help="Exclusive end date in YYYY-MM-DD format.")
    predict_parser.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")
    predict_parser.add_argument("--model-id", default="amazon/chronos-2")
    predict_parser.add_argument("--prediction-length", type=int, default=7)
    predict_parser.add_argument("--context-length", type=int)
    predict_parser.add_argument("--batch-size", type=int, default=8)
    predict_parser.add_argument("--n-long", type=int, default=5)
    predict_parser.add_argument("--n-short", type=int, default=5)
    predict_parser.add_argument("--benchmark", default="SPY")
    predict_parser.add_argument("--cross-learning", action="store_true", help="Enable Chronos-2 cross-learning.")
    predict_parser.add_argument("--no-covariates", action="store_true", help="Run the pure-univariate variant.")
    predict_parser.add_argument(
        "--architecture",
        default="v1_residual",
        choices=("v1_residual", "v2_flagship_basket_raw"),
        help="Forecast architecture to run.",
    )
    predict_parser.add_argument(
        "--basket-mode",
        default="full",
        choices=("full", "growth_spike", "expanded40", "expanded60"),
        help="V2 basket rollout mode.",
    )

    predict_parser.add_argument(
        "--beta-mode",
        default=DEFAULT_V2_BETA_MODE,
        choices=("unit_beta_1.0", "pooled_beta_140"),
        help="V2 beta handling mode.",
    )
    predict_parser.add_argument(
        "--book-mode",
        default=DEFAULT_V2_BOOK_MODE,
        choices=("tail_fraction", "fixed_count"),
        help="Book construction mode for V2.",
    )
    predict_parser.add_argument(
        "--tail-fraction",
        type=float,
        default=DEFAULT_V2_TAIL_FRACTION,
        help="Tail fraction for V2 tail-only books.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze-signal",
        help="Run tail/execution/basket diagnostics on a frozen V2 experiment without rerunning Chronos-2.",
    )
    analyze_parser.add_argument("--run-dir", required=True, help="Root output directory containing labeled_frame.csv.")
    analyze_parser.add_argument("--experiment", required=True, help="Experiment subdirectory to analyze.")
    analyze_parser.add_argument("--output-dir", required=True, help="Directory where analysis outputs will be written.")
    analyze_parser.add_argument("--score-column", default="score_z")
    analyze_parser.add_argument("--tail-fractions", default="0.10,0.15,0.20")
    analyze_parser.add_argument("--execution-interval", default="5m")
    analyze_parser.add_argument("--execution-mode", default="strict", choices=("strict",))

    reconcile_parser = subparsers.add_parser(
        "reconcile-live",
        help="Reconcile a saved predict-next output with realized next-session returns and append a live paper-trading ledger.",
    )
    reconcile_parser.add_argument(
        "--prediction-dir",
        required=True,
        help="Experiment directory produced by predict-next containing forecasts.json, book.csv, and prediction_summary.csv.",
    )
    reconcile_parser.add_argument(
        "--ledger-dir",
        required=True,
        help="Directory where cumulative live reconciliation outputs will be written.",
    )
    reconcile_parser.add_argument("--execution-interval", default="5m")
    reconcile_parser.add_argument("--execution-mode", default="strict", choices=("strict",))

    chronos_parser = subparsers.add_parser(
        "chronos-report",
        help="Build a dated, human-readable Chronos package with prediction boards and optional backtest boards.",
    )
    chronos_parser.add_argument(
        "--prediction-dir",
        required=True,
        help="Experiment directory containing forecasts.json, book.csv, and prediction_summary.csv.",
    )
    chronos_parser.add_argument(
        "--analysis-dir",
        help="Optional analysis directory containing backtest_scorecard.csv and related analysis outputs.",
    )
    chronos_parser.add_argument(
        "--ledger-dir",
        help="Optional live scoreboard ledger directory to include current live reconciliation boards in the package.",
    )
    chronos_parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to outputs/chronos_<today>.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "chronos-report":
        output_dir = Path(args.output_dir) if args.output_dir else default_chronos_output_dir()
        export_chronos_package(
            prediction_dir=Path(args.prediction_dir),
            analysis_dir=Path(args.analysis_dir) if args.analysis_dir else None,
            ledger_dir=Path(args.ledger_dir) if args.ledger_dir else None,
            output_dir=output_dir,
            report_mode="report_only",
            report_context={
                "prediction_dir": args.prediction_dir,
                "analysis_dir": args.analysis_dir or "",
                "ledger_dir": args.ledger_dir or "",
                "reconciliation_status": "not_run",
                "reconciliation_message": "",
            },
        )
        print(f"Wrote Chronos report package to {output_dir}")
        return

    if args.command == "reconcile-live":
        pipeline = ZeroShotResearchPipeline()
        pipeline.reconcile_live_prediction(
            prediction_dir=Path(args.prediction_dir),
            ledger_dir=Path(args.ledger_dir),
            execution_interval=args.execution_interval,
            execution_mode=args.execution_mode,
        )
        print(f"Wrote live reconciliation outputs to {Path(args.ledger_dir)}")
        return

    if args.command == "analyze-signal":
        pipeline = ZeroShotResearchPipeline()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline.analyze_signal_run(
            run_dir=Path(args.run_dir),
            experiment=args.experiment,
            output_dir=output_dir,
            score_column=args.score_column,
            tail_fractions=_parse_tail_fractions(args.tail_fractions),
            execution_interval=args.execution_interval,
            execution_mode=args.execution_mode,
        )
        print(f"Wrote signal analysis outputs to {output_dir}")
        return

    start, end = _default_start_end() if args.command == "run" else _default_prediction_window()
    if args.start:
        start = args.start
    if args.end:
        end = args.end

    tickers = _resolve_v2_tickers(_parse_tickers(args), args)
    default_contexts = ResearchConfig()
    resolved_context_length = _resolve_context_length(args, default_contexts)
    research_config = ResearchConfig(
        model_id=args.model_id,
        prediction_length=args.prediction_length,
        default_context_length=default_contexts.default_context_length if args.context_length is None else args.context_length,
        benchmark_ticker=args.benchmark.upper(),
        batch_size=args.batch_size,
        n_long=args.n_long,
        n_short=args.n_short,
        liquid_universe=tuple(tickers),
    )
    args.context_length = resolved_context_length
    pipeline = ZeroShotResearchPipeline(research_config=research_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.architecture == "v2_flagship_basket_raw":
        labeled_frame, v2_base_frame = pipeline.prepare_market_frames(tickers=tickers, start=start, end=end)
        labeled_frame.to_csv(output_dir / "labeled_frame.csv", index=False)
        v2_base_frame.to_csv(output_dir / "v2_base_frame.csv", index=False)
    else:
        labeled_frame = pipeline.prepare_labeled_frame(tickers=tickers, start=start, end=end)
        labeled_frame.to_csv(output_dir / "labeled_frame.csv", index=False)
        v2_base_frame = pd.DataFrame()

    if args.command == "run" and args.matrix and args.architecture == "v2_flagship_basket_raw":
        matrix_results = pipeline.run_v2_matrix(
            v2_base_frame,
            basket_mode=args.basket_mode,
            beta_mode=args.beta_mode,
            book_mode=args.book_mode,
            tail_fraction=args.tail_fraction,
        )
        for experiment_name, result in matrix_results.items():
            ZeroShotResearchPipeline.write_result(output_dir / experiment_name, result)
        v1_config = research_config.default_forecaster_config(use_covariates=True, cross_learning=False)
        v1_experiment_name = f"{_model_slug(args.model_id)}_ctx{v1_config.context_length}_cross0_cov1"
        v1_result = pipeline.run_experiment(labeled_frame, v1_config, v1_experiment_name)
        ZeroShotResearchPipeline.write_result(output_dir / v1_experiment_name, v1_result)
        baseline_results = pipeline.run_baselines(labeled_frame)
        for baseline_name, result in baseline_results.items():
            ZeroShotResearchPipeline.write_result(output_dir / baseline_name, result)
        print(f"Wrote matrix outputs to {output_dir}")
        return

    if args.command == "run" and args.matrix:
        matrix_results = pipeline.run_matrix(labeled_frame)
        for experiment_name, result in matrix_results.items():
            ZeroShotResearchPipeline.write_result(output_dir / experiment_name, result)
        baseline_results = pipeline.run_baselines(labeled_frame)
        for baseline_name, result in baseline_results.items():
            ZeroShotResearchPipeline.write_result(output_dir / baseline_name, result)
        print(f"Wrote matrix outputs to {output_dir}")
        return

    forecaster_config = ForecasterConfig(
        model_id=args.model_id,
        prediction_length=args.prediction_length,
        context_length=resolved_context_length,
        quantiles=(0.1, 0.5, 0.9),
        cross_learning=args.cross_learning,
        use_covariates=not args.no_covariates,
        batch_size=args.batch_size,
        beta_mode=args.beta_mode,
    )
    experiment_name = _build_experiment_name(args)
    if args.command == "predict-next":
        if args.architecture == "v2_flagship_basket_raw":
            result = pipeline.run_v2_next_session_prediction(
                v2_base_frame,
                forecaster_config,
                experiment_name,
                basket_mode=args.basket_mode,
                book_mode=args.book_mode,
                tail_fraction=args.tail_fraction,
            )
        else:
            result = pipeline.run_next_session_prediction(labeled_frame, forecaster_config, experiment_name)
        ZeroShotResearchPipeline.write_live_prediction(output_dir / experiment_name, result)
        if result.book.frame.empty:
            print("No next-session prediction was produced. Check history length and universe coverage.")
            return

        summary = result.prediction_summary.iloc[0]
        preview_columns = [column for column in ["side", "rank", "ticker", "basket_id", "score_plain", "score_z", "score_q50"] if column in result.book.frame.columns]
        preview = result.book.frame[preview_columns].sort_values(["side", "rank", "ticker"])
        print(
            "Predicted next session book for "
            f"{pd.Timestamp(summary['prediction_session']).date()} using signal session "
            f"{pd.Timestamp(summary['signal_session']).date()}:"
        )
        if bool(summary.get("has_newer_observed_session_than_signal", False)):
            observed_latest = summary.get("observed_latest_session", "")
            if observed_latest:
                observed_text = str(pd.Timestamp(observed_latest).date())
            else:
                observed_text = "unknown"
            print(
                "WARNING: fetched market data contained a newer observed session "
                f"({observed_text}) than the latest complete signal session. "
                "This prediction does not leak future labels, but it may target an already-started session."
            )
        print(preview.to_string(index=False))
        print(f"Wrote next-session outputs to {output_dir / experiment_name}")
        return

    if args.architecture == "v2_flagship_basket_raw":
        result = pipeline.run_v2_experiment(
            v2_base_frame,
            forecaster_config,
            experiment_name,
            basket_mode=args.basket_mode,
            book_mode=args.book_mode,
            tail_fraction=args.tail_fraction,
        )
    else:
        result = pipeline.run_experiment(labeled_frame, forecaster_config, experiment_name)
    ZeroShotResearchPipeline.write_result(output_dir / experiment_name, result)

    baseline_results = pipeline.run_baselines(labeled_frame)
    for baseline_name, baseline_result in baseline_results.items():
        ZeroShotResearchPipeline.write_result(output_dir / baseline_name, baseline_result)

    print(f"Wrote experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
