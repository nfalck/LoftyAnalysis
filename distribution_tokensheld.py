import pandas as pd

MAX_PERCENTAGE_TOKEN_OWNERSHIP = 0.15 # >15% is assumed to be platform-controlled

properties = pd.read_csv("properties_v7_with_ratios.csv")
holders = pd.read_csv("holders_v7 copy.csv")
transactions = pd.read_csv("transactions_v7.csv")

# Add property_type and tokens_total column from properties.csv
holders = holders.merge(
    properties[["property_id", "asa_id", "property_type", "tokens_total"]],
    on=["property_id", "asa_id"],
    how="left"
)

# Convert to numeric columns
holders["tokens_held"] = pd.to_numeric(holders["tokens_held"])
holders["tokens_total"] = pd.to_numeric(holders["tokens_total"])

# Calculate the holder share per property
holders["holder_share"] = holders["tokens_held"] / holders["tokens_total"]

# Label the wallets with >15% token ownership and/or is_application_account flag as platform-controlled wallets
holders["is_platform_wallet"] = (holders["holder_share"] > MAX_PERCENTAGE_TOKEN_OWNERSHIP)

# Make one copy with all wallets and one copy excluding the platform-controlled wallets
holders_investors_only = holders[~holders["is_platform_wallet"]].copy()
holders_all = holders.copy()

# Aggregate to the average and median tokens held and group by property_id, asa_id and property_type
per_property_all = (
    holders_all.groupby(["property_id", "asa_id", "property_type"])["tokens_held"]
    .agg(mean_tokens_per_holder="mean", median_tokens_per_holder="median")
    .reset_index()
)

per_property_investors = (
    holders_investors_only.groupby(["property_id", "asa_id", "property_type"])["tokens_held"]
    .agg(mean_tokens_per_holder="mean", median_tokens_per_holder="median")
    .reset_index()
)

# Create the two CSVs, one including all wallets and one with the filtered wallets
per_property_all.to_csv("per_property_tokens_all_v7.csv", index=False)
per_property_investors.to_csv("per_property_tokens_investors_v7.csv", index=False)

# Round the average and median tokens held and create a rounded version of the CSVs
per_property_all["mean_tokens_per_holder"] = (per_property_all["mean_tokens_per_holder"].round(0))
per_property_investors["mean_tokens_per_holder"] = (per_property_investors["mean_tokens_per_holder"].round(0))
per_property_all["median_tokens_per_holder"] = (per_property_all["median_tokens_per_holder"].round(0))
per_property_investors["median_tokens_per_holder"] = (per_property_investors["median_tokens_per_holder"].round(0))

per_property_all.to_csv("per_property_tokens_all_rounded_v7.csv", index=False)
per_property_investors.to_csv("per_property_tokens_investors_rounded_v7.csv", index=False)