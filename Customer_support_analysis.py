import pandas as pd
import matplotlib.pyplot as plt

# === E-Commerce Customer Support Efficiency & Satisfaction Analysis ===

# Load dataset
df = pd.read_csv("Customer_support_data.csv")

# Convert datetime columns
df["Issue_reported at"] = pd.to_datetime(df["Issue_reported at"], errors="coerce")
df["issue_responded"] = pd.to_datetime(df["issue_responded"], errors="coerce")

# Compute response time in minutes
df["response_time_min"] = (df["issue_responded"] - df["Issue_reported at"]).dt.total_seconds() / 60

# Clean data
df = df.dropna(subset=["CSAT Score", "response_time_min", "channel_name", "category"])

# 1️⃣ Average CSAT by Channel
csat_by_channel = df.groupby("channel_name")["CSAT Score"].mean().sort_values(ascending=False)
plt.figure()
csat_by_channel.plot(kind="bar", color="skyblue")
plt.title("Average CSAT by Channel")
plt.ylabel("CSAT Score")
plt.tight_layout()
plt.savefig("csat_by_channel.png")

# 2️⃣ Top 10 Issue Categories
issues_by_cat = df["category"].value_counts().head(10)
plt.figure()
issues_by_cat.plot(kind="barh", color="orange")
plt.title("Top 10 Issue Categories")
plt.xlabel("Number of Issues")
plt.tight_layout()
plt.savefig("issues_by_category.png")

# 3️⃣ Response Time vs CSAT
avg_response_csat = df.groupby("channel_name")[["response_time_min", "CSAT Score"]].mean()
plt.figure()
plt.scatter(avg_response_csat["response_time_min"], avg_response_csat["CSAT Score"], s=100, alpha=0.7)
plt.title("Response Time vs CSAT by Channel")
plt.xlabel("Avg Response Time (min)")
plt.ylabel("Avg CSAT")
plt.tight_layout()
plt.savefig("response_time_vs_csat.png")

# === Summary ===
print("=== Summary Insights ===")
print(f"Channels analyzed: {df['channel_name'].nunique()}")
print("Highest CSAT Channel:", csat_by_channel.idxmax())
print("Top Issue Category:", issues_by_cat.index[0])
print("Correlation Response Time ↔ CSAT:", df['response_time_min'].corr(df['CSAT Score']))
