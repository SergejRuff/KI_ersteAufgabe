# import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from convert_column_to_integers import convert_string_columns_to_integers
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from numpy import sqrt
from numpy import argmax

# import churn csv
churn_df = pd.read_csv("../data/Churn.csv")

# filter data first

# total charges is not numeric.
churn_df["TotalCharges"] = churn_df["TotalCharges"].replace(' ', np.nan)
# attempt to convert the column to float
churn_df["TotalCharges"] = churn_df["TotalCharges"].astype(float)
# now 11 NAs are introduced by transformation. should be removed
churn_df = churn_df.dropna()

# option True: bin the continues variables for decision tree
# option 2: remove the continues variables

option_ = False

if option_:
    print("option 1 was selected: keep continues variables")
    # bin the continues variables using sturges bin-rule
    # 1 + log2(N) where N is the number of data points
    sturges_bin = int(1 + np.log2(len(churn_df)))  # need integer, not float : 13
    churn_df["MonthlyCharges"] = pd.cut(churn_df["MonthlyCharges"], bins=sturges_bin).apply(lambda x: x.mid)
    churn_df["tenure"] = pd.cut(churn_df["tenure"], bins=sturges_bin).apply(lambda x: x.mid)
    churn_df["TotalCharges"] = pd.cut(churn_df["TotalCharges"], bins=sturges_bin).apply(lambda x: x.mid)

    # drop coloumns with useless information
    churn_df = churn_df.drop(labels=["customerID", "gender"], axis=1)
else:
    print("option 2 was selected: remove continues variables")
    # drop coloumns with useless information
    churn_df = churn_df.drop(labels=["customerID", "gender", "MonthlyCharges", "tenure",
                                     "TotalCharges"], axis=1)

class_names_ = churn_df["Churn"]

# call function to convert every unique character into integer
convert_string_columns_to_integers(churn_df)

# divide into training and test data: 80/20 %
churnx_train, churnx_test, churny_train, churny_test = train_test_split(
    churn_df.drop("Churn", axis=1), churn_df["Churn"], test_size=0.2, random_state=0
)

print("The training dataset is {}% of the total dataset".format(round(len(churnx_train) / len(churn_df) * 100)))
print("The test dataset is {}% of the total dataset".format(round(len(churnx_test) / len(churn_df) * 100)))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(churnx_train, churny_train)

#fig = plt.figure(figsize=(50, 40))
#tree.plot_tree(clf, feature_names=churn_df.drop("Churn", axis=1).columns,
#               class_names=class_names_, filled=True)

print("accuracy of training data: {:.3f}".format(
    clf.score(churnx_train, churny_train)
))
print("accuracy of test data: {:.3f}".format(
    clf.score(churnx_test, churny_test)
))

#fig.savefig("../output/churn_tree.svg")

# second task

# Use the classifier to make predictions on the test data
#test_predictions = clf.predict(churnx_test)
test_predictions = clf.predict_proba(churnx_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(churny_test, test_predictions)


# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)
print('Churn: AUROC = %.3f' % (roc_auc))

optimal_treshhold = thresholds[np.argmax(tpr-fpr)]
print("Best Threshold=",optimal_treshhold)
# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


