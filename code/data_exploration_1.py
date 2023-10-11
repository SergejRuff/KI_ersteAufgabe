"""
this script explores the variables
in the churn.csv.
goal: diciding on which variables should
be fed to the decision-tree
"""

import pandas as pd
import matplotlib.pyplot as plt

# import churn csv
churn_df = pd.read_csv("../data/Churn.csv")

# summary statistics for df
print(churn_df.describe())

churn_df["Churn"] = churn_df["Churn"].replace(["No", "Yes"], [0, 1])

# generate a 2x2 frequency table
partner_churn_table = pd.crosstab(churn_df["Partner"], churn_df["Churn"])

print(partner_churn_table)

# barplot for partner destribution by churn-status

partner_churn_table.plot.bar()

plt.xlabel("partner")
plt.ylabel("churn count")
plt.title("partner distribution by churn-status")

plt.show()

# generate a 2x2 frequency table
partner_churn_table = pd.crosstab(churn_df["Dependents"], churn_df["Churn"])

print(partner_churn_table)

# barplot for partner distribution by churn-status

partner_churn_table.plot.bar()

plt.xlabel("Dependents")
plt.ylabel("churn count")
plt.title("Dependents distribution by churn-status")

plt.show()
