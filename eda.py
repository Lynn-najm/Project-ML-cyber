import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/Merged01.csv")

print(df.head())
print(df.info())
print(df.shape)
print(df.columns.tolist())
print(df['Label'].value_counts())


df['Label'].value_counts().plot(kind='bar', figsize=(12,6))
plt.title("Class Distribution")
plt.show()