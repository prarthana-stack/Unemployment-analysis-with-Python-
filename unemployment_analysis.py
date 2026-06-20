import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check current folder
print("Current Working Directory:")
print(os.getcwd())

# Load datasets
df1 = pd.read_csv("Unemployment in India (3).csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Show first 5 rows
print("\nDataset 1:")
print(df1.head())

print("\nDataset 2:")
print(df2.head())

# Dataset information
print("\nDataset 1 Info:")
df1.info()

print("\nDataset 2 Info:")
df2.info()

# Check missing values
print("\nMissing Values in Dataset 1:")
print(df1.isnull().sum())

print("\nMissing Values in Dataset 2:")
print(df2.isnull().sum())

# Remove extra spaces from column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Remove missing values
df1 = df1.dropna()
df2 = df2.dropna()

# Check missing values after cleaning
print("\nMissing Values After Cleaning Dataset 1:")
print(df1.isnull().sum())

print("\nMissing Values After Cleaning Dataset 2:")
print(df2.isnull().sum())

# Show column names
print("\nDataset 1 Columns:")
print(df1.columns)

print("\nDataset 2 Columns:")
print(df2.columns)

# Statistical summary
print("\nDataset 1 Statistical Summary:")
print(df1["Estimated Unemployment Rate (%)"].describe())

print("\nDataset 2 Statistical Summary:")
print(df2["Estimated Unemployment Rate (%)"].describe())

# Convert Date column
df1["Date"] = pd.to_datetime(df1["Date"], dayfirst=True)
df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)

# ----------------------------
# Unemployment Trend Over Time
# ----------------------------
trend1 = df1.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(12, 6))
plt.plot(
    trend1.index,
    trend1.values,
    marker="o",
    linewidth=2
)

plt.title("Unemployment Trend in India")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Covid Impact Analysis
# ----------------------------
df2["Covid Period"] = df2["Date"].apply(
    lambda x: "Before Covid"
    if x < pd.Timestamp("2020-03-01")
    else "During Covid"
)

covid_impact = df2.groupby(
    "Covid Period"
)["Estimated Unemployment Rate (%)"].mean()

print("\nCovid Impact on Unemployment:")
print(covid_impact)

plt.figure(figsize=(8, 5))
plt.bar(
    covid_impact.index,
    covid_impact.values
)

plt.title("Covid-19 Impact on Unemployment in India")
plt.xlabel("Period")
plt.ylabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# ----------------------------
# State-wise Unemployment
# ----------------------------
state_unemployment = df1.groupby(
    "Region"
)["Estimated Unemployment Rate (%)"].mean()

top_states = state_unemployment.sort_values(
    ascending=False
).head(10)

print("\nTop 10 States with Highest Unemployment:")
print(top_states)

plt.figure(figsize=(12, 6))
plt.bar(
    top_states.index,
    top_states.values
)

plt.title("Top 10 States with Highest Unemployment")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# Monthly Trend
# ----------------------------
df1["Month"] = df1["Date"].dt.month_name()

monthly_order = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

monthly_trend = df1.groupby(
    "Month"
)["Estimated Unemployment Rate (%)"].mean()

monthly_trend = monthly_trend.reindex(monthly_order)

print("\nMonthly Unemployment Trend:")
print(monthly_trend)

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_trend.index,
    monthly_trend.values,
    marker="o"
)

plt.title("Monthly Unemployment Trend")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Correlation Heatmap
# ----------------------------
plt.figure(figsize=(8, 6))

numeric_df = df1.select_dtypes(
    include=["float64", "int64"]
)

sns.heatmap(
    numeric_df.corr(),
    annot=True
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nAnalysis Completed Successfully!")