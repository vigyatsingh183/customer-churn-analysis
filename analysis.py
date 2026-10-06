import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

# Convert churn into numerical value
df["Churn_Flag"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("===== CUSTOMER CHURN ANALYSIS =====")

# Overall churn rate
churn_rate = df["Churn_Flag"].mean() * 100

print(f"\nOverall Churn Rate: {churn_rate:.2f}%")

# Churn by contract
contract_churn = (
    df.groupby("Contract")["Churn_Flag"]
    .mean()
    .sort_values(ascending=False) * 100
)

print("\nChurn Rate by Contract:")
print(contract_churn)

# Churn by tenure
df["Tenure_Group"] = pd.cut(
    df["Tenure_Months"],
    bins=[0, 6, 12, 24, 100],
    labels=["0-6 Months", "7-12 Months", "13-24 Months", "25+ Months"]
)

tenure_churn = (
    df.groupby("Tenure_Group", observed=True)["Churn_Flag"]
    .mean() * 100
)

print("\nChurn Rate by Tenure:")
print(tenure_churn)

# Average monthly charges
print("\nAverage Monthly Charges:")
print(
    df.groupby("Churn")["Monthly_Charges"]
    .mean()
)

# Average support calls
print("\nAverage Support Calls:")
print(
    df.groupby("Churn")["Support_Calls"]
    .mean()
)

# Visualization
contract_churn.plot(
    kind="bar",
    title="Customer Churn Rate by Contract"
)

plt.ylabel("Churn Rate (%)")
plt.xlabel("Contract")
plt.tight_layout()
plt.show()
