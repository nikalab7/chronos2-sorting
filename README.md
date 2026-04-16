# Chronos Equity Prototype

This repository contains a zero-shot Chronos-2 research prototype for US equities using Yahoo Finance intraday data as a pipeline-validation source.

## What It Does

- Downloads regular-hours intraday bars from Yahoo Finance
- Normalizes each session into 7 slots:
  - 09:30
  - 10:30
  - 11:30
  - 12:30
  - 13:30
  - 14:30
  - 15:30
- Builds slot-level residual return labels versus `SPY`
- Creates official Chronos-2 list-of-dicts forecasting tasks
- Runs zero-shot forecasts with the flagship `amazon/chronos-2` model on CPU
- Builds an equal-weight dollar-neutral long-short book
- Reports gross and net results across cost stress tests
- Benchmarks against simple non-ML baselines

## Install

```bash
pip install -e .
```

If you only want to run the unit tests for the non-model math, the codebase works without `chronos-forecasting` being installed. The Chronos import is lazy and only required for the forecasting step.

## Quick Start

```bash
python -m chronos_equity_prototype.cli run ^
  --start 2026-02-15 ^
  --end 2026-04-16 ^
  --output-dir outputs/default_run
```

The project now defaults to the 120M flagship `amazon/chronos-2` model with a CPU-safer default batch size of `8`.

## Predict The Next Session

```bash
python -m chronos_equity_prototype.cli predict-next ^
  --output-dir outputs/next_session
```

This command uses the latest complete regular-hours session in your Yahoo window as the signal date, forecasts the next US trading session's 7 slots, writes `forecasts.json`, `book.csv`, and `prediction_summary.csv`, and prints the long/short book in the terminal.

## Test

```bash
python -m unittest discover -s tests -v
```
