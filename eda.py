import pandas as pd

df = pd.read_csv("dataset/Merged01.csv")

print(df.head())
print(df.info())
print(df.shape)
print(df.columns.tolist())
print(df['Label'].value_counts())