# import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# import churn csv
churn_df = pd.read_csv("../data/Churn.csv")

# filter data first

# replace Male/female with 0/1 for desicion tree
churn_df["gender"] = churn_df["gender"].replace(["Male","Female"],[0,1])

# replace yes/no with 0/1 for desicion tree
churn_df = churn_df.replace(["No","Yes"],[0,1])

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

# divide into training and test data: 80/20 %
churnx_train, churnx_test, churny_train, churny_test = train_test_split(
    churn_df.drop("Churn", axis=1), churn_df["Churn"], test_size=0.2, random_state=0
)

print("The training dataset is {}% of the total dataset".format(round(len(churnx_train)/len(churn_df)*100)))
print("The test dataset is {}% of the total dataset".format(round(len(churnx_test)/len(churn_df)*100)))

