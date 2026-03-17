# %%
import pandas as pd
df = pd.read_csv("match_players.csv")

# %%
# Drop unnecessary columns and nan
df = df.drop(columns=["faction", "new_uncertainty", "left_after", "party_id", "user_id"])
df = df.dropna()
df.head()

# %%
# Create a new won column, 1 if the player won 0 otherwise
df['won'] = 0
# If the new skill rating is higher than the old skill rating, they've won
df.loc[df["new_skill"] > df["old_skill"], "won"] = 1
df.drop(columns=["new_skill", "team_id"])

# %%
# Group by match_id and won (teams effectively)
df = df.groupby(["match_id", "won"])[["old_skill", "old_uncertainty"]].agg(list).reset_index()
df.head()
# %%
