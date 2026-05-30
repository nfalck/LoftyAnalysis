import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

EXCLUDE_PROPERTIES = [
    "two 3-unit properties",
    "two sfrs and one duplex",
]

COMMERCIAL_SET = {"commercial"}
MIXED_USE_SET = {"mixed use"}
MULTIFAMILY_SET = {"duplex", "triplex", "fourplex", "multi family"}

# Order used for the boxplot scatter overlay
WANTED_ORDER = [
    "Duplex",
    "Commercial",
    "Vacation Rental",
    "Single Family",
    "Multi-family residential",
    "Commercial + Mixed Use",
]

VALUE_COL = "underlying_asset_price"

# Helper scripts
def price_fmt(x, _):
    """Format price axis as $Xk or $X.XM"""
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    return f"${int(x/1e3)}k"


def prep_df(df):
    """Correct dtypes, strip whitespace,
    drop missing/infinite values, and apply exclusions."""
    df = df.copy()
    df["property_type"] = df["property_type"].astype(str).str.strip()
    df["ptlc"] = df["property_type"].str.lower()
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    df = df.dropna(subset=["ptlc", VALUE_COL])
    df = df[np.isfinite(df[VALUE_COL]) & (df[VALUE_COL] > 0)]
    mask = df["ptlc"].apply(lambda s: any(k in s for k in EXCLUDE_PROPERTIES))
    return df[~mask].copy()


def build_series(df):
    """Build grouped and standalone series from the dataframe"""
    series = {}
    series["Commercial + Mixed Use"] = df[df["ptlc"].isin(COMMERCIAL_SET | MIXED_USE_SET)][VALUE_COL].values
    series["Multi-family residential"] = df[df["ptlc"].isin(MULTIFAMILY_SET)][VALUE_COL].values
    series["Commercial"] = df[df["ptlc"].isin(COMMERCIAL_SET)][VALUE_COL].values
    series["Duplex"] = df[df["ptlc"].eq("duplex")][VALUE_COL].values
    excluded = COMMERCIAL_SET | MIXED_USE_SET | MULTIFAMILY_SET
    for t in sorted(set(df["ptlc"].unique()) - excluded):
        series[t.title()] = df[df["ptlc"].eq(t)][VALUE_COL].values
    return {k: v[(np.isfinite(v)) & (v > 0)] for k, v in series.items()
            if len(v[(np.isfinite(v)) & (v > 0)]) > 0}


def safe_filename(title, suffix):
    return title.replace(" ", "_").replace("—", "-").replace("/", "_") + f"_{suffix}.png"


def plot_hist_by_series(df, title, output_dir="."):
    """Histogram on a log x-scale, one panel per property type."""
    series = build_series(df)
    if not series:
        print(f"No data for: {title}")
        return

    # Sort panels by data size descending
    order = sorted(series.keys(), key=lambda k: len(series[k]), reverse=True)

    # Shared log bins across all panels
    all_x = np.concatenate([series[k] for k in order])
    log_bins = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), 15)

    ncols = 3
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for i, label in enumerate(order):
        ax = axes[i]
        x = series[label]
        ax.hist(x, bins=log_bins,
                color=colors[i % len(colors)] if colors else None,
                alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_title(f"{label} (n={len(x)})", fontsize=10)
        ax.set_xlabel("Underlying Asset Price ($, log scale)")
        ax.set_ylabel("Number of properties")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(price_fmt))
        ax.tick_params(labelbottom=True)

    for j in range(len(order), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fname = f"{output_dir}/{safe_filename(title, 'hist')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_box_by_series(df, title, output_dir="."):
    """Single pooled boxplot (log scale) with colored jittered scatter overlay per group."""
    series = build_series(df)
    if not series:
        print(f"No data for: {title}")
        return

    groups = [g for g in WANTED_ORDER if g in series and len(series[g]) > 0]
    all_x = np.concatenate([series[g] for g in groups]).astype(float)
    all_x = all_x[np.isfinite(all_x) & (all_x > 0)]

    fig, ax = plt.subplots(figsize=(10, 3.2))

    ax.boxplot(
        [all_x],
        vert=False, showfliers=False, widths=0.35, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="lightgray", alpha=0.35),
    )

    # Colored jittered scatter overlay by group
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, g in enumerate(groups):
        x = np.asarray(series[g], dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if len(x) == 0:
            continue
        y = np.ones(len(x)) + (np.random.rand(len(x)) - 0.5) * 0.18
        ax.scatter(x, y, s=16, alpha=0.9,
                   color=colors[i % len(colors)] if colors else None,
                   edgecolors="none", label=f"{g} (n={len(x)})")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(price_fmt))
    ax.set_yticks([])
    ax.set_xlabel("Underlying Asset Price ($, log scale)")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=9, frameon=True, title="Group")
    plt.tight_layout()
    fname = f"{output_dir}/{safe_filename(title, 'box')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

df = pd.read_csv("properties_v7_with_ratios.csv")
df = prep_df(df)

plot_hist_by_series(df, "Distribution of Underlying Asset Price by Property Type", output_dir=".")
plot_box_by_series(df,  "Distribution of Underlying Asset Price by Property Type", output_dir=".")