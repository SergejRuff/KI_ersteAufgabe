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

# barplot for parters (x) and Churn (y)
plt.bar(courses, values, color ='gray',
        width = 0.4)

plt.xlabel("partner")
plt.ylabel("Churn count")
plt.title("Distribution of partners by Churn")

plt.show()

