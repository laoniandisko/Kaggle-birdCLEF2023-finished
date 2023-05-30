import random

import pandas as pd

df = pd.read_csv("upsample_flod40.csv")
dic = {}
for index in range(len(df)):
    if df.iloc[index,:].primary_label not in dic:
        dic[df.iloc[index,:].primary_label] = []
        dic[df.iloc[index,:].primary_label].append(df.iloc[index, :].filename)
    else:
        dic[df.iloc[index, :].primary_label].append(df.iloc[index,:].filename )

is_mixup = []
is_mixup2 = []
for index in range(len(df)):
    is_mixup.append(random.choice(dic[df.iloc[index,:].primary_label] ))
    is_mixup2.append(random.choice(dic[df.iloc[index,:].primary_label] ))

df["is_mixup"] = is_mixup
df["is_mixup2"] = is_mixup2
df.to_csv("upsample_fold40_mixup.csv",index=False)

print(index)


