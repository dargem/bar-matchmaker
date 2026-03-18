# %%
import numpy as np
import pandas as pd

df = pd.read_csv("match_players.csv")

# %%
# Drop unnecessary columns and nan
df = df.drop(columns=["faction", "new_uncertainty", "left_after", "party_id", "user_id"])
df = df.dropna()

# %%
# Create a new won column, 1 if the player won 0 otherwise
df['won'] = 0
# If the new skill rating is higher than the old skill rating, they've won
df.loc[df["new_skill"] > df["old_skill"], "won"] = 1
df = df.drop(columns=["new_skill", "team_id"])

# %%
# Group by match_id and won (teams effectively)
df = df.groupby(["match_id", "won"])[["old_skill", "old_uncertainty"]].agg(list).reset_index()
df = df.rename(columns={"old_skill": "skill", "old_uncertainty": "uncertainty"})

# %%
# merge winners and losers together
filtered = (
    df.set_index(["match_id", "won"])[["skill", "uncertainty"]]
      .unstack("won")
)

filtered.columns = [
    f"{'win' if won == 1 else 'loss'}_{col}"
    for col, won in filtered.columns
]

filtered = filtered.reset_index()
filtered = filtered.dropna(subset=[
    "win_skill",
    "loss_skill",
    "win_uncertainty",
    "loss_uncertainty",
])

# Convert list-valued columns into numeric team-level features.
features = filtered.assign(
    win_skill=filtered["win_skill"].map(lambda xs: np.asarray(xs, dtype=float)),
    loss_skill=filtered["loss_skill"].map(lambda xs: np.asarray(xs, dtype=float)),
    win_uncertainty=filtered["win_uncertainty"].map(lambda xs: np.asarray(xs, dtype=float)),
    loss_uncertainty=filtered["loss_uncertainty"].map(lambda xs: np.asarray(xs, dtype=float)),
)

features = features.assign(
    win_skill_mean=features["win_skill"].map(np.mean),
    loss_skill_mean=features["loss_skill"].map(np.mean),
    win_uncertainty_mean=features["win_uncertainty"].map(np.mean),
    loss_uncertainty_mean=features["loss_uncertainty"].map(np.mean),
    win_team_size=features["win_skill"].map(len),
    loss_team_size=features["loss_skill"].map(len),
)

# keep only a specific team size
# features = features.query("win_team_size == 2 and loss_team_size == 2").copy()

# Randomize which side is team_0 vs team_1 so the label isn't always team_0.
rng = np.random.default_rng(42)
swap = rng.random(len(features)) < 0.5

# %%
X = pd.DataFrame({
    # drop match ID
    "team_0_skill": np.where(swap, features["loss_skill_mean"], features["win_skill_mean"]),
    "team_1_skill": np.where(swap, features["win_skill_mean"], features["loss_skill_mean"]),
    "team_0_uncertainty": np.where(swap, features["loss_uncertainty_mean"], features["win_uncertainty_mean"]),
    "team_1_uncertainty": np.where(swap, features["win_uncertainty_mean"], features["loss_uncertainty_mean"]),
    "team_0_size": np.where(swap, features["loss_team_size"], features["win_team_size"]),
    "team_1_size": np.where(swap, features["win_team_size"], features["loss_team_size"]),
})

y = pd.DataFrame({
    # y = 1 if team_0 wins else 0
    "y": np.where(swap, 0, 1).astype(int)
})

# Build our train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.75, random_state=42
)

# %%
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

# %%
predictions = clf.predict(X_test)

# accuracy = accuracy_score(y_test, predictions)
# %%
