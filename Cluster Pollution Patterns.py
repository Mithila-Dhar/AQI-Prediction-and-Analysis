import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('Ariyalur_AQIBulletins.csv')
df.head()

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['Month'] = df['date'].dt.month
df['Month_Name'] = df['date'].dt.strftime('%B')

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date')

X = df[['Index Value']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10,5))
plt.scatter(df['date'], df['Index Value'], c=df['Cluster'], cmap='viridis')
plt.title("Pollution Clusters Over Time")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.colorbar(label="Cluster")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Cluster'], y=df['Index Value'], palette='Set2')
plt.title("Cluster wise Index Value Distribution")
plt.xlabel("Cluster")
plt.ylabel("Index Value")
plt.show()

df['Cluster'].value_counts()

centroids = kmeans.cluster_centers_
scaled_centroids = scaler.inverse_transform(centroids)

cluster_info = pd.DataFrame({
    "Cluster": [0,1,2],
    "Index Value Range (Centroid)": scaled_centroids.flatten()
})

cluster_info

df['Pollution Level'] = df['Index Value'].apply(
    lambda x: "Low" if x < scaled_centroids.flatten().min() else
              "High" if x > scaled_centroids.flatten().max() else
              "Medium"
)
