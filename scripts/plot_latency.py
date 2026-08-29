"""
plot_latency.py — reads all CSVs from data/ and produces:
  1. TTFT p99 vs lambda (one line per config)
  2. Throughput (req/s) vs lambda (one line per config)
  3. CDF of E2E latency at a fixed lambda

Usage:
    python scripts/plot_latency.py
    python scripts/plot_latency.py --datadir=data --fixed-lambda=5
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#ccc",
    "xtick.color":      "#999",
    "ytick.color":      "#999",
    "text.color":       "#ccc",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#444",
})

CONFIG_COLORS = {
    "continuous_fcfs":        "#4fc3f7",
    "continuous_shortprompt": "#81c995",
    "continuous_slo":         "#ba68c8",
    "static_fcfs":            "#ff7043",
    "static_shortprompt":     "#ffd54f",
}

def load_data(datadir):
    frames = []
    for path in glob.glob(os.path.join(datadir, "*_lam*.csv")):
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        raise FileNotFoundError(f"No sweep CSVs found in '{datadir}/'. Run sweep.ps1 first.")
    return pd.concat(frames, ignore_index=True)

def compute_metrics(df):
    df = df.copy()
    df["ttft_ms"]   = df["first_token_ms"] - df["arrival_ms"]
    df["e2e_ms"]    = df["end_ms"]          - df["arrival_ms"]
    df["wait_ms"]   = df["start_ms"]        - df["arrival_ms"]
    df["config"]    = df["mode"] + "_" + df["policy"]

    def calc_agg(g):
        return pd.Series({
            "ttft_p50":     g["ttft_ms"].quantile(0.50),
            "ttft_p99":     g["ttft_ms"].quantile(0.99),
            "e2e_p99":      g["e2e_ms"].quantile(0.99),
            "throughput":   len(g) / (g["end_ms"].max() - g["arrival_ms"].min()) * 1000 if (g["end_ms"].max() > g["arrival_ms"].min()) else 0.0,
        })

    agg = df.groupby(["config", "lambda"], as_index=False).apply(calc_agg, include_groups=False)
    return df, agg

def plot_ttft_vs_lambda(agg, outpath):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("TTFT p99  vs  Arrival Rate", fontsize=14, pad=12)
    ax.set_xlabel("λ  (requests / sec)")
    ax.set_ylabel("TTFT p99  (ms)")
    ax.grid(True)

    for cfg, grp in agg.groupby("config"):
        grp = grp.sort_values("lambda")
        color = CONFIG_COLORS.get(cfg, "#aaa")
        ax.plot(grp["lambda"], grp["ttft_p99"], marker="o", label=cfg,
                color=color, linewidth=2, markersize=5)

    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"saved {outpath}")
    plt.close(fig)

def plot_throughput_vs_lambda(agg, outpath):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Throughput  vs  Arrival Rate", fontsize=14, pad=12)
    ax.set_xlabel("λ  (requests / sec)")
    ax.set_ylabel("Throughput  (req / sec)")
    ax.grid(True)

    for cfg, grp in agg.groupby("config"):
        grp = grp.sort_values("lambda")
        color = CONFIG_COLORS.get(cfg, "#aaa")
        ax.plot(grp["lambda"], grp["throughput"], marker="s", label=cfg,
                color=color, linewidth=2, markersize=5)

    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"saved {outpath}")
    plt.close(fig)

def plot_e2e_cdf(df, fixed_lambda, outpath):
    sub = df[df["lambda"] == fixed_lambda]
    if sub.empty:
        print(f"no data for lambda={fixed_lambda}, skipping CDF")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"CDF of E2E Latency  (λ = {fixed_lambda} req/s)", fontsize=14, pad=12)
    ax.set_xlabel("E2E latency  (ms)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.grid(True)

    for cfg, grp in sub.groupby("config"):
        vals = np.sort(grp["e2e_ms"].values)
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        color = CONFIG_COLORS.get(cfg, "#aaa")
        ax.plot(vals, cdf, label=cfg, color=color, linewidth=2)

    ax.legend(framealpha=0.6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"saved {outpath}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",      default="data")
    parser.add_argument("--outdir",       default="data")
    parser.add_argument("--fixed-lambda", type=float, default=5.0,
                        help="lambda value to use for the CDF plot")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_raw = load_data(args.datadir)
    df, agg = compute_metrics(df_raw)

    plot_ttft_vs_lambda(agg, os.path.join(args.outdir, "ttft_p99.png"))
    plot_throughput_vs_lambda(agg, os.path.join(args.outdir, "throughput.png"))
    plot_e2e_cdf(df, args.fixed_lambda, os.path.join(args.outdir, "e2e_cdf.png"))

if __name__ == "__main__":
    main()
