import pandas as pd
import numpy as np

PANEL_FILE = "transaction_panel_v5_clean.xlsx"
VNQ_FILE = "VNQ.xlsx"
MORTGAGE_FILE = "Res_mortgage_ret.xlsx"
ETH_FILE = "eth-usd-max.csv"
OUTPUT_FILE = "transaction_panel_v6_clean.xlsx"

Y = "log_net_flow"

L = 7  # distributed-lag window in days

ENTITY = "asa_id"
DATE = "date"

# Load Panel v5
panel = pd.read_excel(PANEL_FILE)
panel[DATE] = pd.to_datetime(panel[DATE])

# VNQ returns
vnq = pd.read_excel(VNQ_FILE)
vnq = vnq[["Date", "Returns"]].copy()
vnq.columns = ["date", "vnq_return"]
vnq["date"] = pd.to_datetime(vnq["date"])
vnq["vnq_return"] = pd.to_numeric(vnq["vnq_return"], errors="coerce")
vnq = vnq.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

full_idx = pd.date_range(vnq["date"].min(), panel[DATE].max(), freq="D")
vnq = (
    vnq.set_index("date")
    .reindex(full_idx)
    .ffill()
    .reset_index()
    .rename(columns={"index": "date"})
)

# Mortgage returns
mort = pd.read_excel(MORTGAGE_FILE)
mort = mort[["observation_date", "NASDAQNQUSB30203020T_PCH"]].copy()
mort.columns = ["date", "mortgage_ret"]
mort["date"] = pd.to_datetime(mort["date"])
mort["mortgage_ret"] = pd.to_numeric(mort["mortgage_ret"], errors="coerce")
mort = mort.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# Forward-fill over weekends and holidays
full_idx_m = pd.date_range(mort["date"].min(), panel[DATE].max(), freq="D")
mort = (
    mort.set_index("date")
    .reindex(full_idx_m)
    .ffill()
    .reset_index()
    .rename(columns={"index": "date"})
)

# Merge new variables into panel
panel = panel.merge(vnq[["date", "vnq_return"]], on="date", how="left")
panel = panel.merge(mort[["date", "mortgage_ret"]], on="date", how="left")

# Z-score of new variables
for col in ["vnq_return", "mortgage_ret"]:
    mu = panel[col].mean()
    sd = panel[col].std()
    panel[f"{col}_z"] = (panel[col] - mu) / sd

# ETH shock
eth_raw = pd.read_csv(ETH_FILE)
eth_raw["date"] = pd.to_datetime(
    eth_raw["snapped_at"].str.replace(" UTC", ""), format="mixed"
).dt.normalize()
eth_raw = (
    eth_raw[["date", "price"]]
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)
eth_raw["eth_return_raw"] = eth_raw["price"] / eth_raw["price"].shift(1) - 1

# 5-day cumulative ETH return
eth_raw["eth_cum_return_5d_before"] = (
        (1 + eth_raw["eth_return_raw"])
        .shift(1)
        .rolling(window=5, min_periods=5)
        .apply(np.prod, raw=True)
        - 1
)

# Dummy = 1 if cumulative 5-day pre-return < -5%
eth_raw["eth_shock"] = (eth_raw["eth_cum_return_5d_before"] < -0.05).astype(float)
eth_raw.loc[eth_raw["eth_cum_return_5d_before"].isna(), "eth_shock"] = np.nan

panel = panel.merge(
    eth_raw[["date", "eth_cum_return_5d_before", "eth_shock"]],
    on=DATE, how="left"
)

shock_days = int(eth_raw["eth_shock"].sum())
total_days = eth_raw["eth_shock"].notna().sum()

# Save panel as V6
new_cols = ["vnq_return", "vnq_return_z", "mortgage_ret", "mortgage_ret_z",
            "eth_cum_return_5d_before", "eth_shock"]
v5_cols = [c for c in panel.columns if c not in new_cols]
panel = panel[v5_cols + new_cols]

panel.to_excel(OUTPUT_FILE, index=False)
print(f"Saved {OUTPUT_FILE}")