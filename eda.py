import pandas as pd
import matplotlib.pyplot as plt
import json

# =========================================================
# 1) LOAD DATASET
# =========================================================
df = pd.read_csv("dataset/Merged01.csv")

# =========================================================
# 2) CONVERT LABELS TO BINARY
#    BENIGN stays BENIGN
#    every other attack class becomes ATTACK
# =========================================================
df["Label"] = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")

# =========================================================
# 3) BASIC EDA ON ORIGINAL DATASET
# =========================================================
print("Class distribution in original dataset:")
print(df["Label"].value_counts())

print("\nDuplicates in original dataset:", df.duplicated().sum())
print("\nMissing values in original dataset:\n", df.isnull().sum())
print("\nLow-unique columns in original dataset:\n", df.nunique().sort_values().head(10))

# =========================================================
# 4) PLOT 1: ORIGINAL DATASET DISTRIBUTION
# =========================================================
df["Label"].value_counts().plot(kind="bar", figsize=(6, 4))
plt.title("Original Dataset Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# =========================================================
# 5) REMOVE DUPLICATES
# =========================================================
df = df.drop_duplicates()

print("\nShape after dropping duplicates:", df.shape)
print("\nClass distribution after dropping duplicates:")
print(df["Label"].value_counts())

# =========================================================
# 6) PLOT 2: DISTRIBUTION AFTER REMOVING DUPLICATES
# =========================================================
df["Label"].value_counts().plot(kind="bar", figsize=(6, 4))
plt.title("Distribution After Removing Duplicates")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# =========================================================
# 7) SEPARATE THE TWO CLASSES
# =========================================================
df_attack = df[df["Label"] == "ATTACK"]
df_benign = df[df["Label"] == "BENIGN"]

# =========================================================
# 8) CREATE A BALANCED SAMPLE
#    take the same number of rows from each class
# =========================================================
n = 10000

df_sample = pd.concat([
    df_attack.sample(n=n, random_state=42),
    df_benign.sample(n=n, random_state=42)
])

# shuffle the rows so ATTACK and BENIGN are mixed
df_sample = df_sample.sample(frac=1, random_state=42)

print("\nClass distribution in balanced sample:")
print(df_sample["Label"].value_counts())
print("\nBalanced sample shape:", df_sample.shape)

# =========================================================
# 9) PLOT 3: BALANCED SAMPLE DISTRIBUTION
# =========================================================
df_sample["Label"].value_counts().plot(kind="bar", figsize=(6, 4))
plt.title("Balanced Sample Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

df_sample.to_csv("clean_sample.csv", index=False)


features = {
    "target": "Label",
    "features": [col for col in df_sample.columns if col != "Label"]
}

with open("feature_list.json", "w") as f:
    json.dump(features, f, indent=4)


import os
print("Current working directory:", os.getcwd())