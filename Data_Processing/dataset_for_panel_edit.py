import pandas as pd
import numpy as np

INPUT  = "transaction_panel_v4.xlsx"
OUTPUT = ("transaction_panel_v4_clean.xlsx")

df = pd.read_excel(INPUT)

first_date = df.groupby("asa_id")["date"].min().rename("first_date")
df = df.join(first_date, on="asa_id")
df["is_first_day"] = df["date"] == df["first_date"]

first_day_data = (
    df[df["is_first_day"]][["asa_id", "total_tokens_transferred", "n_transactions"]]
    .rename(columns={
        "total_tokens_transferred": "first_day_tokens",
        "n_transactions":           "first_day_txns",
    })
)

df_clean = df[~df["is_first_day"]].drop(columns=["first_date", "is_first_day"])

# Ensure no NaN values and that all values are positive
df_clean = df_clean.dropna()
df_clean = df_clean[
    (df_clean["total_tokens_transferred"] > 0) &
    (df_clean["n_transactions"] > 0)
]

df_clean = df_clean.merge(first_day_data, on="asa_id", how="left")

# Calculate relative variables
df_clean["tokens_rel"] = df_clean["total_tokens_transferred"] / df_clean["first_day_tokens"]
df_clean["transactions_rel"] = df_clean["n_transactions"] / df_clean["first_day_txns"]

# Log-transform variables
df_clean["log_total_tokens_transferred"] = np.log(df_clean["total_tokens_transferred"])
df_clean["log_n_transactions"] = np.log(df_clean["n_transactions"])
df_clean["log_tokens_rel"] = np.log(df_clean["tokens_rel"])
df_clean["log_transactions_rel"] = np.log(df_clean["transactions_rel"])

# Z-scores
xs = ["delta_vix", "delta_dgs10", "delta_dgs1mo", "ads", "eth_return"]
for x in xs:
    mu = df_clean[x].mean()
    sd = df_clean[x].std()
    df_clean[f"{x}_z"] = (df_clean[x] - mu) / sd

cols = [
    "asa_id", "date",
    "total_tokens_transferred", "n_transactions",
    "tokens_rel", "transactions_rel",
    "log_total_tokens_transferred", "log_n_transactions",
    "log_tokens_rel", "log_transactions_rel",
    "delta_vix", "delta_dgs10", "delta_dgs1mo", "ads", "eth_return",
    "delta_vix_z", "delta_dgs10_z", "delta_dgs1mo_z", "ads_z", "eth_return_z",
    "first_day_tokens", "first_day_txns",
]

# Create final dataset
df_final = (
    df_clean[cols]
    .drop_duplicates()
    .sort_values(["asa_id", "date"])
    .reset_index(drop=True)
)

df_final.to_excel(OUTPUT, index=False)
print(f"Saved {OUTPUT}")