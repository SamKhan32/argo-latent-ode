import pandas as pd
import matplotlib.pyplot as plt

LOW_DRIFT_PFL_PATH = "data/processed/PFL1_low_drift_devices.csv"
PFL1_PATH = "data/processed/PFL1_preprocessed.csv"

low_drift_df = pd.read_csv(LOW_DRIFT_PFL_PATH)
pfl1_df = pd.read_csv(PFL1_PATH)

low_drift_wmo_ids = low_drift_df["WMO_ID"].unique()
pfl1_filtered = pfl1_df[pfl1_df["WMO_ID"].isin(low_drift_wmo_ids)].copy()

cast_meta = (
    pfl1_filtered[["WMO_ID", "castIndex", "date"]]
    .drop_duplicates(subset=["castIndex"])
    .copy()
)
cast_meta["date"] = pd.to_datetime(cast_meta["date"], format="%Y%m%d")
cast_meta["month"] = cast_meta["date"].dt.month

# how many seasons does each float actually cover?
def n_seasons(months):
    def s(m):
        if m in [12,1,2]: return "winter"
        elif m in [3,4,5]: return "spring"
        elif m in [6,7,8]: return "summer"
        else: return "autumn"
    return len(set(s(m) for m in months))

float_season_coverage = (
    cast_meta.groupby("WMO_ID")["month"]
    .apply(n_seasons)
    .reset_index(name="n_seasons_covered")
)

print(float_season_coverage["n_seasons_covered"].value_counts().sort_index())

# how long are floats deployed?
float_duration = (
    cast_meta.groupby("WMO_ID")["date"]
    .agg(lambda x: (x.max() - x.min()).days)
    .reset_index(name="duration_days")
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

float_season_coverage["n_seasons_covered"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], title="Seasons covered per float"
)
axes[0].set_xlabel("Number of seasons")

float_duration["duration_days"].hist(bins=30, ax=axes[1])
axes[1].set_title("Float deployment duration (days)")
axes[1].set_xlabel("Days")

# cast count per float
cast_counts = cast_meta.groupby("WMO_ID").size().reset_index(name="n_casts")
cast_counts["n_casts"].hist(bins=30, ax=axes[2])
axes[2].set_title("Casts per float")
axes[2].set_xlabel("N casts")

plt.tight_layout()
plt.savefig("data_visualization/float_season_coverage.png", dpi=150)
plt.show()