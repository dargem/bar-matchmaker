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
max_players = int(max(features["win_team_size"].max(), features["loss_team_size"].max()))

def pad_sorted_desc(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.sort(arr)[::-1]
    out = np.full(n, np.nan, dtype=float)
    out[: min(n, len(arr))] = arr[:n]
    return out

team0_skill_lists = np.where(swap, features["loss_skill"], features["win_skill"])
team1_skill_lists = np.where(swap, features["win_skill"], features["loss_skill"])
team0_uncertainty_lists = np.where(swap, features["loss_uncertainty"], features["win_uncertainty"])
team1_uncertainty_lists = np.where(swap, features["win_uncertainty"], features["loss_uncertainty"])

team0_skill_mat = np.vstack([pad_sorted_desc(a, max_players) for a in team0_skill_lists])
team1_skill_mat = np.vstack([pad_sorted_desc(a, max_players) for a in team1_skill_lists])
team0_uncertainty_mat = np.vstack(
    [pad_sorted_desc(a, max_players) for a in team0_uncertainty_lists]
)
team1_uncertainty_mat = np.vstack(
    [pad_sorted_desc(a, max_players) for a in team1_uncertainty_lists]
)

X = pd.DataFrame(
    np.hstack([team0_skill_mat, team1_skill_mat, team0_uncertainty_mat, team1_uncertainty_mat]),
    columns=(
        [f"team_0_skill_{i+1}" for i in range(max_players)] +
        [f"team_1_skill_{i+1}" for i in range(max_players)] +
        [f"team_0_uncertainty_{i+1}" for i in range(max_players)] +
        [f"team_1_uncertainty_{i+1}" for i in range(max_players)]
    ),
)

y = pd.DataFrame({
    # y = 1 if team_0 wins else 0
    "y": np.where(swap, 0, 1).astype(int)
})

# Build our train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.75, random_state=42, stratify=y
)

# %%
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

base = HistGradientBoostingClassifier()

clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
clf.fit(X_train, y_train["y"])

# %%
p_team0_win = clf.predict_proba(X_test)[:, 1]
y_true = y_test["y"].to_numpy()

print("brier:", brier_score_loss(y_true, p_team0_win))
print("logloss:", log_loss(y_true, p_team0_win))
print("accuracy:", accuracy_score(y_true, (p_team0_win >= 0.5).astype(int)))
print("roc_auc:", roc_auc_score(y_true, p_team0_win))

# Baseline: always predict the empirical win rate in the test set
p_base = np.full_like(p_team0_win, fill_value=y_true.mean(), dtype=float)
print("baseline_brier:", brier_score_loss(y_true, p_base))

# %%
