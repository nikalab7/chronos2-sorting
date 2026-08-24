import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from chronos import BaseChronosPipeline

MODEL_NAME = "amazon/chronos-bolt-base"
MODEL_REVISION = "5d9f166d69f47aef3401367a7b842e78fe97b121"
CONTEXT = 512
HORIZON = 48
Q_LEVELS = [0.5]
EWMA_LAMBDA = 0.94
EWMA_WARMUP = 52
N_ORIGINS = 10
OUT_DIR = Path("wdc_representation_results")
OUT_DIR.mkdir(exist_ok=True)


def download_aligned():
    raw = yf.download(
        ["WDC", "SOXX"],
        interval="15m",
        period="60d",
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("Yahoo returned no data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        raise RuntimeError(f"Unexpected yfinance columns: {raw.columns}")

    close = close[["WDC", "SOXX"]].dropna(how="any")
    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    close.index = close.index.tz_convert("America/New_York")

    # yfinance timestamps are bar starts. Keep regular 15-minute bars only.
    close = close.between_time("09:30", "15:45")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if (close <= 0).any().any():
        raise RuntimeError("Non-positive adjusted close found")
    return close


def p50_path(pipeline, x):
    ctx = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        try:
            q, _ = pipeline.predict_quantiles(
                ctx,
                prediction_length=HORIZON,
                quantile_levels=Q_LEVELS,
                limit_prediction_length=False,
            )
        except TypeError:
            q, _ = pipeline.predict_quantiles(
                ctx,
                prediction_length=HORIZON,
                quantile_levels=Q_LEVELS,
            )
    a = q.detach().cpu().float().numpy()
    a = np.squeeze(a)
    if a.ndim == 1:
        out = a
    elif a.shape == (HORIZON, 1):
        out = a[:, 0]
    elif a.shape == (1, HORIZON):
        out = a[0]
    else:
        raise RuntimeError(f"Unexpected p50 shape {a.shape}")
    if out.shape != (HORIZON,) or not np.isfinite(out).all():
        raise RuntimeError(f"Bad forecast path shape/values: {out.shape}")
    return out.astype(np.float64)


def ewma_z_context(log_returns):
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) < CONTEXT + EWMA_WARMUP:
        raise ValueError("Insufficient history for causal EWMA standardization")

    v = float(np.var(r[:EWMA_WARMUP], ddof=1))
    v = max(v, 1e-12)
    z = np.full(len(r), np.nan, dtype=np.float64)

    for i in range(EWMA_WARMUP, len(r)):
        sigma = math.sqrt(max(v, 1e-12))
        z[i] = r[i] / sigma
        v = EWMA_LAMBDA * v + (1.0 - EWMA_LAMBDA) * (r[i] ** 2)

    usable = z[np.isfinite(z)]
    if len(usable) < CONTEXT:
        raise ValueError("Not enough finite EWMA-z values")
    return usable[-CONTEXT:], v


def invert_ewma_z(pred_z, variance_after_origin):
    v = max(float(variance_after_origin), 1e-12)
    raw = []
    for z in np.asarray(pred_z, dtype=np.float64):
        sigma = math.sqrt(max(v, 1e-12))
        r = float(z) * sigma
        raw.append(r)
        v = EWMA_LAMBDA * v + (1.0 - EWMA_LAMBDA) * (r ** 2)
    return np.asarray(raw, dtype=np.float64)


def endpoint_return_from_log_path(path):
    return float(np.exp(np.sum(path)) - 1.0)


def select_origins(close):
    candidates = []
    for i, ts in enumerate(close.index):
        if ts.strftime("%H:%M") != "15:45":
            continue
        # Need enough transformed history and exactly 48 future aligned bars.
        if i < CONTEXT + EWMA_WARMUP + 2:
            continue
        if i + HORIZON >= len(close):
            continue
        candidates.append(i)

    if len(candidates) < N_ORIGINS:
        raise RuntimeError(f"Only {len(candidates)} valid 15:45 origins")
    return candidates[-N_ORIGINS:]


def main():
    close = download_aligned()
    origins = select_origins(close)
    print(f"Aligned bars: {len(close)}")
    print("Origins:")
    for i in origins:
        print(" ", close.index[i].isoformat())

    print("Loading Chronos-Bolt Base at pinned revision...")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    print("Model loaded")

    rows = []

    for origin_i in origins:
        origin_ts = close.index[origin_i]
        hist = close.iloc[: origin_i + 1]
        future = close.iloc[origin_i + 1 : origin_i + 1 + HORIZON]
        if len(future) != HORIZON:
            raise RuntimeError("Future horizon length mismatch")

        last_close = float(hist["WDC"].iloc[-1])
        actual_ret = float(future["WDC"].iloc[-1] / last_close - 1.0)

        # A: raw adjusted close control.
        close_ctx = hist["WDC"].to_numpy(dtype=np.float64)[-CONTEXT:]
        pred_close = p50_path(pipeline, close_ctx)
        pred_close_ret = float(pred_close[-1] / last_close - 1.0)

        # Paired WDC/SOXX log returns up through the origin.
        logs = np.log(hist[["WDC", "SOXX"]]).diff().dropna()
        wdc_r = logs["WDC"].to_numpy(dtype=np.float64)
        soxx_r = logs["SOXX"].to_numpy(dtype=np.float64)

        # B: ordinary log return.
        log_ctx = wdc_r[-CONTEXT:]
        pred_log = p50_path(pipeline, log_ctx)
        pred_log_ret = endpoint_return_from_log_path(pred_log)

        # C: causal EWMA-volatility-standardized log return.
        z_ctx, variance_after_origin = ewma_z_context(wdc_r)
        pred_z = p50_path(pipeline, z_ctx)
        pred_z_raw = invert_ewma_z(pred_z, variance_after_origin)
        pred_z_ret = endpoint_return_from_log_path(pred_z_raw)

        # D: sector residual decomposition.
        # Estimate beta on the same last-512 return window, no future data.
        w = wdc_r[-CONTEXT:]
        s = soxx_r[-CONTEXT:]
        denom = float(np.dot(s, s))
        beta = float(np.dot(w, s) / denom) if denom > 1e-16 else 0.0
        residual = w - beta * s

        pred_residual = p50_path(pipeline, residual)
        pred_sector = p50_path(pipeline, s)
        pred_sector_residual_raw = beta * pred_sector + pred_residual
        pred_sector_residual_ret = endpoint_return_from_log_path(pred_sector_residual_raw)

        preds = {
            "CLOSE_CONTROL": pred_close_ret,
            "LOG_RETURN": pred_log_ret,
            "EWMA_Z_RETURN": pred_z_ret,
            "SECTOR_RESIDUAL": pred_sector_residual_ret,
        }

        for branch, pred_ret in preds.items():
            rows.append(
                {
                    "origin": origin_ts.isoformat(),
                    "branch": branch,
                    "last_close": last_close,
                    "predicted_return": pred_ret,
                    "actual_return": actual_ret,
                    "predicted_direction": int(np.sign(pred_ret)),
                    "actual_direction": int(np.sign(actual_ret)),
                    "direction_correct": int(np.sign(pred_ret) == np.sign(actual_ret)),
                    "absolute_error": abs(pred_ret - actual_ret),
                    "beta_soxx": beta if branch == "SECTOR_RESIDUAL" else np.nan,
                }
            )

        print(
            f"{origin_ts.date()} actual={actual_ret:+.3%} | "
            + " | ".join(f"{k}={v:+.3%}" for k, v in preds.items())
        )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("branch")
        .agg(
            forecasts=("origin", "count"),
            direction_accuracy=("direction_correct", "mean"),
            mae=("absolute_error", "mean"),
            median_abs_error=("absolute_error", "median"),
            mean_predicted_return=("predicted_return", "mean"),
            mean_actual_return=("actual_return", "mean"),
        )
        .reset_index()
    )

    # Which branch had the lowest absolute error at each origin?
    winner_counts = (
        df.loc[df.groupby("origin")["absolute_error"].idxmin(), "branch"]
        .value_counts()
        .to_dict()
    )

    # Direct head-to-head vs LOG_RETURN on each origin.
    pivot = df.pivot(index="origin", columns="branch", values="absolute_error")
    vs_log = {}
    for branch in ["CLOSE_CONTROL", "EWMA_Z_RETURN", "SECTOR_RESIDUAL"]:
        d = pivot[branch] - pivot["LOG_RETURN"]
        vs_log[branch] = {
            "beats_log_on_abs_error_origins": int((d < 0).sum()),
            "loses_to_log_on_abs_error_origins": int((d > 0).sum()),
            "ties": int((d == 0).sum()),
            "mean_abs_error_delta_vs_log": float(d.mean()),
            "median_abs_error_delta_vs_log": float(d.median()),
        }

    df.to_csv(OUT_DIR / "wdc_bolt_representation_predictions.csv", index=False)
    summary.to_csv(OUT_DIR / "wdc_bolt_representation_summary.csv", index=False)

    result = {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "ticker": "WDC",
        "sector_proxy": "SOXX",
        "context": CONTEXT,
        "horizon": HORIZON,
        "origins": [close.index[i].isoformat() for i in origins],
        "summary": summary.to_dict(orient="records"),
        "lowest_abs_error_winner_counts": winner_counts,
        "head_to_head_vs_log_return": vs_log,
        "notes": {
            "CLOSE_CONTROL": "Last 512 adjusted closes; direct p50 endpoint return.",
            "LOG_RETURN": "Last 512 causal log returns; p50 path summed and exponentiated.",
            "EWMA_Z_RETURN": "Each historical log return divided by causal pre-return EWMA sigma (lambda=0.94); forecast inverted recursively using predicted returns only.",
            "SECTOR_RESIDUAL": "WDC return = beta*SOXX return + residual. Beta fit on last 512 past paired returns. Bolt forecasts SOXX and residual separately, then paths are recombined. No future sector values are used.",
        },
    }
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== LOWEST ABS ERROR WIN COUNTS ===")
    print(json.dumps(winner_counts, indent=2))
    print("\n=== HEAD TO HEAD VS LOG_RETURN ===")
    print(json.dumps(vs_log, indent=2))
    print("\nRESULT_JSON=" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
