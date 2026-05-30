import pandas as pd
import numpy as np

DATA_FILE = "data_v8.xlsx"
TX_FILE = "transactions_v7_good.csv"
PANEL_V4 = "transaction_panel_v4_clean.xlsx"
OUTPUT_FILE = "transaction_panel_v5_clean.xlsx"

LOFTY_RESERVE = "LOFTYRITC3QUX6TVQBGT3BARKWAZDEB2TTJWYQMH6YITKNH7IOMWRLC7SA"

# Identify management wallets (>15% of tokens_total in any property)
holders = pd.read_excel(DATA_FILE, sheet_name="holders_v7")
props = pd.read_excel(DATA_FILE, sheet_name="properties_v7_with_ratios")

merged = holders.merge(props[["asa_id", "tokens_total"]], on="asa_id", how="left")
merged["pct"] = merged["tokens_held"] / merged["tokens_total"]

mgmt_addresses = set(
    merged.loc[
        (merged["pct"] > 0.15) & (merged["holder_address"] != LOFTY_RESERVE),
        "holder_address"
    ].unique()
)

PLATFORM = mgmt_addresses | {LOFTY_RESERVE}
tx = pd.read_csv(TX_FILE)
tx["round_time"] = pd.to_datetime(tx["round_time"], format="mixed", dayfirst=False)
tx["date"] = tx["round_time"].dt.normalize()

# Use real_transfer_amount > 0 which includes both transfers and close-outs
tx_flows = tx[tx["real_transfer_amount"] > 0].copy()

# Classify each transaction
tx_flows["sender_is_platform"] = tx_flows["sender"].isin(PLATFORM)
tx_flows["receiver_is_platform"] = tx_flows["receiver"].isin(PLATFORM)

# Inflow: tokens arriving at investor wallets; platform to investor or investor to investor (receiver gains)
inflow_mask = ~tx_flows["receiver_is_platform"]
inflow_df = tx_flows[inflow_mask][["asa_id", "date", "real_transfer_amount"]].copy()
inflow_df = inflow_df.rename(columns={"real_transfer_amount": "inflow_amount"})

# Outflow: tokens leaving investor wallets; investor to platform, investor to investor (sender loses), or close-out (investor exits)
outflow_mask = ~tx_flows["sender_is_platform"]
outflow_df = tx_flows[outflow_mask][["asa_id", "date", "real_transfer_amount"]].copy()
outflow_df = outflow_df.rename(columns={"real_transfer_amount": "outflow_amount"})

# Aggregate per asa_id and date
daily_inflow = inflow_df.groupby(["asa_id", "date"])["inflow_amount"].sum().reset_index()
daily_outflow = outflow_df.groupby(["asa_id", "date"])["outflow_amount"].sum().reset_index()

# Merge with panel v4
panel = pd.read_excel(PANEL_V4)
panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

panel = panel.merge(daily_inflow,  on=["asa_id", "date"], how="left")
panel = panel.merge(daily_outflow, on=["asa_id", "date"], how="left")

# Fill NaN with 0 (no outflow or inflow on that day)
panel["inflow_amount"] = panel["inflow_amount"].fillna(0)
panel["outflow_amount"] = panel["outflow_amount"].fillna(0)

# Net flow: positive = net investor accumulation, negative = net investor exit
panel["net_flow"] = panel["inflow_amount"] - panel["outflow_amount"]

# Log transforms
panel["log_inflow"] = np.log1p(panel["inflow_amount"])
panel["log_outflow"] = np.log1p(panel["outflow_amount"])

# net_flow can be negative thus signed log
panel["log_net_flow"] = np.sign(panel["net_flow"]) * np.log1p(panel["net_flow"].abs())

new_cols = [
    "inflow_amount", "outflow_amount", "net_flow",
    "log_inflow", "log_outflow", "log_net_flow"
]

# Insert new cols after existing Ys
base_cols = list(panel.columns[:panel.columns.get_loc("log_transactions_rel") + 1])
rest_cols = [c for c in panel.columns if c not in base_cols and c not in new_cols]
panel = panel[base_cols + new_cols + rest_cols]

# Save as xlsx
panel.to_excel(OUTPUT_FILE, index=False)
print(f"Saved → {OUTPUT_FILE}")
