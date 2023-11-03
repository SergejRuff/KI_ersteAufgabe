"""
this script explores the variables
in the churn.csv.
goal: diciding on which variables should
be fed to the decision-tree
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import churn csv
churn_df = pd.read_csv("data/Churn.csv")


# total charges is not numeric.
# how datatypes before transformation.
print("dataframe before datatype change for Total charges coloumn:\n")
print(churn_df.info())
# Replace spaces with NaN in the "TotalCharges" column
churn_df["TotalCharges"] = churn_df["TotalCharges"].replace(' ', np.nan)
# attempt to convert the column to float
churn_df["TotalCharges"] = churn_df["TotalCharges"].astype(float)
# return datatype: Totalcharges should be float now
print("\n")
print("dataframe after datatype change for Total charges coloumn:\n")
print(churn_df.info())

churn_df = churn_df.dropna()

# print how many values are NA
print(churn_df.isnull().sum())

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


# Create the boxplot for tenure vs churn-status.
plt.boxplot([churn_df[churn_df["Churn"] == 0]["tenure"], churn_df[churn_df["Churn"] == 1]["tenure"]])
# Set labels for the boxes
plt.xticks([1, 2], ["No Churn", "Churn"])
plt.ylabel("duration of tenure")
plt.title("boxplot for churn or no churn depending on tenure length")
plt.show()


# generate a 2x2 frequency table
partner_churn_table = pd.crosstab(churn_df["gender"], churn_df["Churn"])

print(partner_churn_table)

# barplot for partner distribution by churn-status

partner_churn_table.plot.bar()

plt.xlabel("gender")
plt.ylabel("churn count")
plt.title("gender distribution by churn-status")

plt.show()

# generate a 2x2 frequency table
partner_churn_table = pd.crosstab(churn_df["SeniorCitizen"], churn_df["Churn"])

# Normalize the crosstab to get percentages
partner_churn_table = partner_churn_table.div(partner_churn_table.sum(1), axis=0) * 100

print(partner_churn_table)

# barplot for partner distribution by churn-status

partner_churn_table.plot.bar()

plt.xlabel("SeniorCitizen")
plt.ylabel("churn count")
plt.title("SeniorCitizen distribution by churn-status")

plt.show()

sns.kdeplot(churn_df[churn_df['Churn'] == 0]['MonthlyCharges'], label='No Churn', fill=True)
sns.kdeplot(churn_df[churn_df['Churn'] == 1]['MonthlyCharges'], label='Churn', fill=True)
plt.title('Density Plot of MonthlyCharges by Churn')
plt.xlabel('MonthlyCharges')
plt.ylabel('Density')
plt.legend()
plt.show()
