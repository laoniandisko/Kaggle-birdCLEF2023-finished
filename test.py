import pandas as pd

df = pd.read_csv("upsample_flod40.csv")

import random


#添加是否增强
is_zengqiang = []

for i in range(len(df)):
    if random.random()>0.5:
        is_zengqiang.append(1)
    else:
        is_zengqiang.append(0)
df["is_zengqiang"] = is_zengqiang

print(df.value_counts())

#添加fold
fold = []
for i in range(len(df)):
    fold.append(random.randint(0,4))
df["fold"] = fold

#添加年份
df["data_year"] = 2023

df.to_csv("upsample_flod40.csv",index=False)


