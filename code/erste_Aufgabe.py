# import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from convert_column_to_integers import convert_string_columns_to_integers

# import churn csv
churn_df = pd.read_csv("../data/Churn.csv")

# filter data first

# total charges is not numeric.
churn_df["TotalCharges"] = churn_df["TotalCharges"].replace(' ', np.nan)
# attempt to convert the column to float
churn_df["TotalCharges"] = churn_df["TotalCharges"].astype(float)
# now 11 NAs are introduced by transformation. should be removed
churn_df = churn_df.dropna()

# drop coloumns with useless information
churn_df = churn_df.drop(labels=["customerID", "gender", "tenure",
                                 "MonthlyCharges","TotalCharges"],axis=1)

# call function to convert every unique character into integer
convert_string_columns_to_integers(churn_df)


# divide into training and test data: 80/20 %
churnx_train, churnx_test, churny_train, churny_test = train_test_split(
    churn_df.drop("Churn", axis=1), churn_df["Churn"], test_size=0.2, random_state=0
)

print("The training dataset is {}% of the total dataset".format(round(len(churnx_train)/len(churn_df)*100)))
print("The test dataset is {}% of the total dataset".format(round(len(churnx_test)/len(churn_df)*100)))
