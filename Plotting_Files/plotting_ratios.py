import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

EXCLUDE_PROPERTIES = [
    "two 3-unit properties",
    "two sfrs and one duplex",
]

COMMERCIAL_SET  = {"commercial"}
MIXED_USE_SET   = {"mixed use"}
MULTIFAMILY_SET = {"duplex", "triplex", "fourplex", "multi family"}

CSV_PATH = "properties_v7_with_ratios.csv"

def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["property_type"]    = df["property_type"].astype(str).str.strip()
    df["property_type_lc"] = df["property_type"].str.lower()
    df["token_ratio"]      = pd.to_numeric(df["token_ratio"], errors="coerce")
    df = df.dropna(subset=["property_type", "token_ratio"])
    df = df[np.isfinite(df["token_ratio"])]
    return df


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    exclusion = df["property_type_lc"].apply(
        lambda s: any(k in s for k in EXCLUDE_PROPERTIES)
    )
    return df[~exclusion].copy()


def build_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    series = {}

    # Grouped series
    series["Commercial + Mixed Use"] = df[
        df["property_type_lc"].isin(COMMERCIAL_SET | MIXED_USE_SET)
    ]["token_ratio"].to_numpy()

    series["Multi-family residential"] = df[
        df["property_type_lc"].isin(MULTIFAMILY_SET)
    ]["token_ratio"].to_numpy()

    # Standalone series
    series["Commercial"] = df[
        df["property_type_lc"].isin(COMMERCIAL_SET)
    ]["token_ratio"].to_numpy()

    series["Duplex"] = df[
        df["property_type_lc"].eq("duplex")
    ]["token_ratio"].to_numpy()

    excluded_from_other = COMMERCIAL_SET | MIXED_USE_SET | MULTIFAMILY_SET
    other_types = sorted(set(df["property_type_lc"].unique()) - excluded_from_other)
    for t in other_types:
        series[t.title()] = df[
            df["property_type_lc"].eq(t)
        ]["token_ratio"].to_numpy()

    # Drop empty series
    series = {k: v for k, v in series.items() if len(v) > 0}
    return series


def fd_bin_edges(x: np.ndarray, min_bins: int = 8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([0.0, 1.0])
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        bins = max(int(np.ceil(np.sqrt(len(x)))), min_bins)
        return np.linspace(x.min(), x.max(), bins + 1)
    bw = 2 * iqr * (len(x) ** (-1 / 3))
    if bw <= 0:
        bins = max(int(np.ceil(np.sqrt(len(x)))), min_bins)
        return np.linspace(x.min(), x.max(), bins + 1)
    n_bins = max(int(np.ceil((x.max() - x.min()) / bw)), min_bins)
    return np.linspace(x.min(), x.max(), n_bins + 1)


def plot_boxplot(series: dict[str, np.ndarray], title: str):
    """Single horizontal boxplot with per-series colored jitter overlay."""
    all_x = np.concatenate(list(series.values())).astype(float)
    all_x = all_x[np.isfinite(all_x)]
    if len(all_x) == 0:
        print("No finite values for boxplot.")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    fig, ax = plt.subplots(figsize=(10, 3.2))

    ax.boxplot(
        [all_x],
        vert=False,
        showfliers=False,
        widths=0.35,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="lightgray", alpha=0.35),
    )

    for i, (label, x) in enumerate(series.items()):
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        c = colors[i % len(colors)] if colors else None
        y = np.ones(len(x)) + (np.random.rand(len(x)) - 0.5) * 0.18
        ax.scatter(x, y, s=16, alpha=0.9, color=c, edgecolors="none",
                   label=f"{label} (n={len(x)})")

    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Ratio (0–1)")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=9, frameon=True, title="Group")
    plt.tight_layout()
    plt.show()


def plot_histograms(series: dict[str, np.ndarray], title: str, ncols: int = 3):
    """One histogram per series, ncols per row, shared x-axis."""
    order  = sorted(series.keys(), key=lambda k: len(series[k]), reverse=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    all_x  = np.concatenate([series[k] for k in order])
    edges  = fd_bin_edges(all_x)
    xmin, xmax = float(all_x.min()), float(all_x.max())

    n_panels = len(order)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        sharex=True,
    )
    axes = np.array(axes).reshape(-1)

    for i, label in enumerate(order):
        ax = axes[i]
        x  = series[label]
        c  = colors[i % len(colors)] if colors else None

        ax.hist(x, bins=edges, color=c, alpha=0.85,
                edgecolor="black", linewidth=0.5)

        ax.set_title(f"{label} (n={len(x)})")
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Ratio (0–1)")
        ax.set_ylabel("Number of properties")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_ecdf(series: dict[str, np.ndarray], title: str, ncols: int = 3):
    """One ECDF per series — same layout & style as histograms."""
    order  = sorted(series.keys(), key=lambda k: len(series[k]), reverse=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    all_x  = np.concatenate([series[k] for k in order])
    xmin, xmax = float(all_x.min()), float(all_x.max())

    n_panels = len(order)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        sharex=True, sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for i, label in enumerate(order):
        ax = axes[i]
        x  = np.sort(series[label])
        y  = np.arange(1, len(x) + 1) / len(x)
        c  = colors[i % len(colors)] if colors else None

        ax.step(x, y, where="post", color=c, linewidth=1.5)

        med = float(np.median(x))
        ax.axvline(med, color="red", linestyle="--", linewidth=1, alpha=0.5,
                   label=f"Median = {med:.3f}")

        ax.set_title(f"{label} (n={len(x)})")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Ratio (0–1)")
        ax.set_ylabel("ECDF")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, frameon=False)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    df = prep_df(df)
    df = apply_exclusions(df)
    series = build_series(df)

    plot_boxplot(series,    "Token Ratio - All property types")
    plot_histograms(series, "Token Ratio - All property types")
    plot_ecdf(series,       "Token Ratio - All property types")