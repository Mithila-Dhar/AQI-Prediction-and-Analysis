import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('Ariyalur_AQIBulletins.csv')
df.head()
print(df.isnull().sum())
print(df.describe())
plt.figure(figsize=(12,5))
plt.plot(df['date'], df['Index Value'])
plt.title("Time-Series Trend of Index Value")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.grid(True)
plt.show()
abnormal = df[(df['Index Value'] < 0) | (df['Index Value'] > df['Index Value'].quantile(0.99))]
abnormal
plt.figure(figsize=(8,5))
sns.countplot(x=df['Prominent Pollutant'])
plt.title("Distribution of Prominent Pollutants")
plt.xticks(rotation=45)
plt.show()
df['Prominent Pollutant'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6,6))
plt.title("Prominent Pollutant Share")
plt.ylabel("")
plt.show()
