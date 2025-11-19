import pandas as pd
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('Ariyalur_AQIBulletins.csv')
df.head()

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['Month'] = df['date'].dt.month
df['Month_Name'] = df['date'].dt.strftime('%B')

monthly_avg = df.groupby('Month')[['Index Value']].mean().reset_index()
monthly_avg['Month_Name'] = monthly_avg['Month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))

monthly_avg

plt.figure(figsize=(12,5))
plt.plot(monthly_avg['Month_Name'], monthly_avg['Index Value'], marker='o')
plt.title("Monthly Average Pollution (Index Value)")
plt.xlabel("Month")
plt.ylabel("Average Index Value")
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

def categorize(val):
    if val < 50:
        return "Clean"
    elif val < 100:
        return "Moderate"
    else:
        return "High Pollution"

monthly_avg['Category'] = monthly_avg['Index Value'].apply(categorize)

monthly_avg

high_pollution_months = monthly_avg[monthly_avg['Category'] == "High Pollution"]
high_pollution_months

insight = ""

if len(high_pollution_months) == 0:
    insight = "No months show consistently high pollution. The region remains relatively clean throughout the year."
else:
    months_list = ", ".join(high_pollution_months['Month_Name'].values)
    insight = f"High pollution months detected: {months_list}. These months show elevated Index Values indicating seasonal pollution spikes."

clean_months = monthly_avg[monthly_avg['Category'] == "Clean"]['Month_Name'].values
moderate_months = monthly_avg[monthly_avg['Category'] == "Moderate"]['Month_Name'].values

insight += f"\n\nClean months: {', '.join(clean_months)}."
insight += f"\nModerate pollution months: {', '.join(moderate_months)}."

print("📌 Seasonal Pollution Insight\n")
print(insight)
