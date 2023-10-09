# import packages
import pandas as pd
from sklearn.model_selection import train_test_split

# import churn csv
churn_df = pd.read_csv("../data/Churn.csv")

# filter data first

# divide into training and test data: 80/20 %
churnx_train, churnx_test,churny_train, churny_test  = train_test_split(
    churn_df.drop("Churn",axis=1),churn_df["Churn"],test_size=0.2,random_state=0
)

print("Der Trainingsdatensatz ist:{}% des Gesamten Datensatzes".format(round(len(churnx_train)/len(churn_df)*100)))
print("Der Testdatensatz ist:{}% des Gesamten Datensatzes".format(round(len(churnx_test)/len(churn_df)*100)))