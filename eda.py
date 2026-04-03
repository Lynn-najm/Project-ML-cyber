import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/Merged01.csv")

# convert labels first
df["Label"] = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")

# print only the useful result
print(df["Label"].value_counts())

print("Duplicates:", df.duplicated().sum())
print("\nMissing values:\n", df.isnull().sum())
print("\nLow-unique columns:\n", df.nunique().sort_values().head(10))

# plot binary distribution
df["Label"].value_counts().plot(kind="bar", figsize=(6, 4))
plt.title("Binary Class Distribution")
plt.show()