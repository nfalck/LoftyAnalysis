import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Properties to exclude from the dataset because they only include 1 property each and are very different from the rest
EXCLUDE_PROPERTIES = [
    "two 3-unit properties",
    "two sfrs and one duplex",
]

# Any type that matches these will be part of the grouped series
COMMERCIAL_SET = {"commercial"}
MIXED_USE_SET = {"mixed use"}
MULTIFAMILY_SET = {"duplex", "triplex", "fourplex", "multi family"}


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the data for the plotting by making sure the property types
    and number of holders are in correct data types and formatting,
    as well as remove missing values (if there are)"""
    df = df.copy()
    df["property_type"] = df["property_type"].astype(str).str.strip()
    df["property_type_lc"] = df["property_type"].str.lower()
    df["n_holders"] = pd.to_numeric(df["n_holders"])
    df = df.dropna(subset=["property_type", "n_holders"])
    df = df[np.isfinite(df["n_holders"])]  # To make sure no values are NaN, Inf, -Inf
    return df


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude the chosen property types in EXCLUDE_PROPERTIES from the dataset"""
    df = df.copy()
    exclusion = df["property_type_lc"].apply(
        lambda s: any(k in s for k in EXCLUDE_PROPERTIES)
    )

    # Return a copy of the dataframe that is now filtered
    return df[~exclusion].copy()


def fd_bin_edges(x: np.ndarray, min_bins: int = 8) -> np.ndarray:
    """Create bins for the histograms that follow the Freedman-Diaconis rule
    with minimum number of bins if needed"""

    # Convert to float array and ensure there is no NaN, Inf, -inf
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    # If no data, set dummy range
    if len(x) == 0:
        return np.array([0, 1])

    # Calculate the interquartile range
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25

    # Use the square-root rule if iqr is 0 or invalid (fallback)
    if iqr <= 0:
        bins = max(int(np.ceil(np.sqrt(len(x)))), min_bins)
        print("Fallback to square-root rule for the bins")
        return np.linspace(x.min(), x.max(), bins + 1)

    # Set the bin width (bw) according to the Freedman-Diaconis formula
    bw = 2 * iqr * (len(x) ** (-1/3))

    # Check if good width, otherwise fallback to square-root rule
    if bw <= 0:
        bins = max(int(np.ceil(np.sqrt(len(x)))), min_bins)
        print("Fallback to square-root rule for the bins")
        return np.linspace(x.min(), x.max(), bins + 1)

    # Calculate the number of bins and set a minimum number
    n_bins = int(np.ceil((x.max() - x.min()) / bw))
    n_bins = max(n_bins, min_bins)

    # Return the bins for histogram plotting
    return np.linspace(x.min(), x.max(), n_bins + 1)


def build_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Create standalone series of each property type (except the excluded ones) and grouped series of
    Commercial + Mixed use and Multi-family residential (duplex, triplex and fourplex)
    """
    series = {}

    # Grouped series
    series["Commercial + Mixed Use"] = df[df["property_type_lc"].isin(COMMERCIAL_SET | MIXED_USE_SET)]["n_holders"].to_numpy()
    series["Multi-family residential"] = df[df["property_type_lc"].isin(MULTIFAMILY_SET)]["n_holders"].to_numpy()

    # Standalone series
    series["Commercial"] = df[df["property_type_lc"].isin(COMMERCIAL_SET)]["n_holders"].to_numpy()
    series["Duplex"] = df[df["property_type_lc"].eq("duplex")]["n_holders"].to_numpy()
    excluded_from_other = (COMMERCIAL_SET | MIXED_USE_SET | MULTIFAMILY_SET)
    other_types = sorted(set(df["property_type_lc"].unique()) - excluded_from_other)

    # Make the property types in title case
    for t in other_types:
        label = t.title()
        series[label] = df[df["property_type_lc"].eq(t)]["n_holders"].to_numpy()

    return series


def plot_hist_by_series(df: pd.DataFrame, title: str, ncols: int = 3):
    """Plot a histogram for each property type with a different color in the same figure"""
    # Prepare the data and create the needed series
    df = prep_df(df)
    df = apply_exclusions(df)
    series = build_series(df)

    # Sort by data size
    order = sorted(series.keys(), key=lambda k: len(series[k]), reverse=True)

    # Set the x-axis range and calculate the bins needed
    all_x = np.concatenate([series[k] for k in order])
    edges = fd_bin_edges(all_x)
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))

    # Determine the subplot layout, 3 plots per row
    n_panels = len(order)
    nrows = int(np.ceil(n_panels / ncols))

    # Create figure and axes
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),  # Adjust figure size based on how many histograms to be made
        sharex=True
    )
    axes = np.array(axes).reshape(-1)

    # Retrieve a list of matplotlib colors for consistent coloring
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    # Plot each histogram
    for i, label in enumerate(order):
        ax = axes[i]
        x = series[label]

        color = colors[i % len(colors)] if colors else None  # To cycle the colors

        ax.hist(
            x,
            bins=edges,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5
        )

        # Axis labelling and formatting
        ax.set_title(f"{label} (n={len(x)})")
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Number of holders per property")
        ax.set_ylabel("Number of properties")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_box_by_series(df: pd.DataFrame, title: str):
    df = prep_df(df)
    df = apply_exclusions(df)
    series = build_series(df)

    wanted_order = [
        "Duplex",
        "Commercial",
        "Vacation Rental",
        "Single Family",
        "Multi-family residential",
        "Commercial + Mixed Use",
    ]

    groups = [g for g in wanted_order if g in series and len(series[g]) > 0]

    # pool all values into one array for the single boxplot
    all_x = np.concatenate([series[g] for g in groups]).astype(float)
    all_x = all_x[np.isfinite(all_x)]
    if len(all_x) == 0:
        print(f"[WARN] No finite values for {title}")
        return

    # plot
    plt.figure(figsize=(10, 3.2))
    ax = plt.gca()

    ax.boxplot(
        [all_x],
        vert=False,
        showfliers=False,
        widths=0.35,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="lightgray", alpha=0.35),
    )

    # colored strip overlay by group (jittered around y=1)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, g in enumerate(groups):
        x = np.asarray(series[g], dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue

        c = colors[i % len(colors)] if colors else None

        y = np.ones(len(x)) + (np.random.rand(len(x)) - 0.5) * 0.18  # jitter
        ax.scatter(
            x, y,
            s=16,
            alpha=0.9,
            color=c,
            edgecolors="none",
            label=f"{g} (n={len(x)})"
        )

    ax.set_yticks([])
    ax.set_xlabel("Number of holders per property")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=9, frameon=True, title="Group")
    plt.tight_layout()
    plt.show()



# Read the CSVs
all_df = pd.read_csv("holders_per_property_all_v7.csv")
inv_df = pd.read_csv("holders_per_property_investors_only_v7.csv")

# Plot the histograms and boxplots
plot_hist_by_series(all_df, "Holders per property - ALL wallets")
plot_box_by_series(all_df,  "Holders per property - ALL wallets")
plot_hist_by_series(inv_df, "Holders per property - INVESTORS only")
plot_box_by_series(inv_df,  "Holders per property - INVESTORS only")