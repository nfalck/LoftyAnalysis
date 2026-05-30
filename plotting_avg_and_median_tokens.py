import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

EXCLUDE_PROPERTIES = [
    "two 3-unit properties",
    "two sfrs and one duplex",
]

COMMERCIAL_SET  = {"commercial"}
MIXED_USE_SET   = {"mixed use"}
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

# Helper scripts
def prep_df(df, value_col):
    """Correct dtypes, strip whitespace,
    drop missing/infinite values, and apply exclusions."""
    df = df.copy()
    df["property_type"] = df["property_type"].astype(str).str.strip()
    df["ptlc"] = df["property_type"].str.lower()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["ptlc", value_col])
    df = df[np.isfinite(df[value_col])]
    mask = df["ptlc"].apply(lambda s: any(k in s for k in EXCLUDE_PROPERTIES))
    return df[~mask].copy()


def build_series(df, value_col):
    """Build grouped and standalone series from the dataframe."""
    series = {}
    series["Commercial + Mixed Use"] = df[df["ptlc"].isin(COMMERCIAL_SET | MIXED_USE_SET)][value_col].values
    series["Multi-family residential"] = df[df["ptlc"].isin(MULTIFAMILY_SET)][value_col].values
    series["Commercial"] = df[df["ptlc"].isin(COMMERCIAL_SET)][value_col].values
    series["Duplex"] = df[df["ptlc"].eq("duplex")][value_col].values
    excluded = COMMERCIAL_SET | MIXED_USE_SET | MULTIFAMILY_SET
    for t in sorted(set(df["ptlc"].unique()) - excluded):
        series[t.title()] = df[df["ptlc"].eq(t)][value_col].values
    return {k: v[np.isfinite(v)] for k, v in series.items() if len(v[np.isfinite(v)]) > 0}


def safe_filename(title, suffix):
    return title.replace(" ", "_").replace("—", "-").replace("/", "_") + f"_{suffix}.png"

def plot_hist_by_series(df, value_col, title, x_label, cap_pct=95, ncols=3, output_dir="."):
    """Plot histograms, one panel per property type, shared x-axis,
    x-axis capped at cap_pct percentile to handle extreme outliers"""
    df = prep_df(df, value_col)
    series = build_series(df, value_col)
    if not series:
        print(f"No data for: {title}")
        return

    order = sorted(series.keys(), key=lambda k: len(series[k]), reverse=True)
    all_x = np.concatenate([series[k] for k in order])
    xmax = np.percentile(all_x, cap_pct)
    xmin = float(all_x.min())
    bins = np.linspace(xmin, xmax, 20)

    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for i, label in enumerate(order):
        ax = axes[i]
        x_vis = series[label][series[label] <= xmax]
        ax.hist(x_vis, bins=bins,
                color=colors[i % len(colors)] if colors else None,
                alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{label} (n={len(series[label])})", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Number of properties")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelbottom=True)

    for j in range(len(order), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fname = f"{output_dir}/{safe_filename(title, 'hist')}"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_box_by_series(df, value_col, title, x_label, cap_pct=95, output_dir="."):
    """Single pooled boxplot with colored jittered scatter overlay per group,
    x-axis capped at cap_pct percentile to handle extreme outliers."""
    df = prep_df(df, value_col)
    series = build_series(df, value_col)
    if not series:
        print(f"No data for: {title}")
        return

    groups = [g for g in WANTED_ORDER if g in series and len(series[g]) > 0]
    all_x = np.concatenate([series[g] for g in groups]).astype(float)
    all_x = all_x[np.isfinite(all_x)]
    xmax = np.percentile(all_x, cap_pct)

    fig, ax = plt.subplots(figsize=(10, 3.2))

    ax.boxplot(
        [all_x[all_x <= xmax]],
        vert=False, showfliers=False, widths=0.35, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="lightgray", alpha=0.35),
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, g in enumerate(groups):
        x = np.asarray(series[g], dtype=float)
        x = x[np.isfinite(x) & (x <= xmax)]
        if len(x) == 0:
            continue
        y = np.ones(len(x)) + (np.random.rand(len(x)) - 0.5) * 0.18
        ax.scatter(x, y, s=16, alpha=0.9,
                   color=colors[i % len(colors)] if colors else None,
                   edgecolors="none", label=f"{g} (n={len(series[g])})")

    ax.set_yticks([])
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=9, frameon=True, title="Group")
    plt.tight_layout()
    fname = f"{output_dir}/{safe_filename(title, 'box')}"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

all_df = pd.read_csv("per_property_tokens_all_rounded_v7.csv")
inv_df = pd.read_csv("per_property_tokens_investors_rounded_v7.csv")

for df, dataset_label in [(all_df, "ALL wallets"), (inv_df, "INVESTORS only")]:
    for value_col, value_name in [
        ("mean_tokens_per_holder",   "Average"),
        ("median_tokens_per_holder", "Median"),
    ]:
        t  = f"{value_name} tokens held per holder - {dataset_label}"
        xl = f"{value_name} tokens held per holder"
        plot_hist_by_series(df, value_col, t, xl, output_dir=".")
        plot_box_by_series(df, value_col, t, xl, output_dir=".")

# ALL wallets - MEAN
plot_hist_by_series(
    all_df,
    value_col="mean_tokens_per_holder",
    fig_title="Average tokens held per holder — ALL wallets",
    x_label="Average tokens held per holder",
)
plot_box_by_series(
    all_df,
    value_col="mean_tokens_per_holder",
    fig_title="Average tokens held per holder — ALL wallets",
    x_label="Average tokens held per holder",
)

# ALL wallets - MEDIAN
plot_hist_by_series(
    all_df,
    value_col="median_tokens_per_holder",
    fig_title="Median tokens held per holder — ALL wallets",
    x_label="Median tokens held per holder",
)
plot_box_by_series(
    all_df,
    value_col="median_tokens_per_holder",
    fig_title="Median tokens held per holder — ALL wallets",
    x_label="Median tokens held per holder",
)

# INVESTORS only - MEAN
plot_hist_by_series(
    inv_df,
    value_col="mean_tokens_per_holder",
    fig_title="Average tokens held per holder — INVESTORS only",
    x_label="Average tokens held per holder",
)
plot_box_by_series(
    inv_df,
    value_col="mean_tokens_per_holder",
    fig_title="Average tokens held per holder — INVESTORS only",
    x_label="Average tokens held per holder",
)

# INVESTORS only - MEDIAN
plot_hist_by_series(
    inv_df,
    value_col="median_tokens_per_holder",
    fig_title="Median tokens held per holder — INVESTORS only",
    x_label="Median tokens held per holder",
)
plot_box_by_series(
    inv_df,
    value_col="median_tokens_per_holder",
    fig_title="Median tokens held per holder — INVESTORS only",
    x_label="Median tokens held per holder",
)