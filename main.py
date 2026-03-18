# %%
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
df.drop(columns=["new_skill", "team_id"])

# %%
# Group by match_id and won (teams effectively)
df = df.groupby(["match_id", "won"])[["old_skill", "old_uncertainty"]].agg(list).reset_index()
df = df.rename(columns={"old_skill": "skill", "old_uncertainty": "uncertainty"})

# %%
# merge winners and losers together
out = (
    df.set_index(["match_id", "won"])[["skill", "uncertainty"]]
      .unstack("won")
)

out.columns = [
    f"{'win' if won == 1 else 'loss'}_{col}"
    for col, won in out.columns
]

out = out.reset_index()