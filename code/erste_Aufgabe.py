# import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from convert_column_to_integers import convert_string_columns_to_integers
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix


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

# fig = plt.figure(figsize=(50, 40))
# tree.plot_tree(clf, feature_names=churn_df.drop("Churn", axis=1).columns,
#               class_names=class_names_, filled=True)

print("accuracy of training data: {:.3f}".format(
    clf.score(churnx_train, churny_train)
))
print("accuracy of test data: {:.3f}".format(
    clf.score(churnx_test, churny_test)
))

# fig.savefig("../output/churn_tree.svg")

# second task

# Use the classifier to make predictions on the test data
# test_predictions = clf.predict(churnx_test)
test_predictions = clf.predict_proba(churnx_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(churny_test, test_predictions)


# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)
print('Churn: AUROC = %.3f' % roc_auc)


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

# Create a list of different cutoff thresholds to iterate over
cutoffs = np.linspace(0, 1, 100)
# Lists to store performance metrics
f1_scores = []
accuracy_scores = []

for cutoff in cutoffs:
    # Convert probabilities to binary predictions using the current cutoff
    binary_predictions = np.where(test_predictions >= cutoff, 1, 0)

    # Calculate F1-score and accuracy for each cutoff
    f1 = f1_score(churny_test, binary_predictions)
    accuracy = accuracy_score(churny_test, binary_predictions)

    f1_scores.append(f1)
    accuracy_scores.append(accuracy)

# Find the index of the best F1-score
best_f1_index = np.argmax(f1_scores)

# Get the corresponding cutoff and metrics
best_cutoff = cutoffs[best_f1_index]
best_f1 = f1_scores[best_f1_index]
best_accuracy = accuracy_scores[best_f1_index]

# Plot the F1-score and accuracy vs. cutoff
plt.figure(figsize=(8, 6))
plt.plot(cutoffs, f1_scores, label='F1-Score')
plt.plot(cutoffs, accuracy_scores, label='Accuracy')
plt.axvline(x=best_cutoff, color='red', linestyle='--', label=f'Best Cutoff = {best_cutoff:.2f}')
plt.xlabel('Cutoff')
plt.ylabel('Score')
plt.title('F1-Score and Accuracy vs. Cutoff')
plt.legend()
plt.show()

print('Best Cutoff:', best_cutoff)
print('Best F1-Score:', best_f1)
print('Accuracy at Best Cutoff:', best_accuracy)

# Use the classifier to make predictions on the test data
test_predictions = clf.predict_proba(churnx_test)[:, 1]

# Convert probabilities to binary predictions using the best cutoff
binary_predictions = np.where(test_predictions >= best_cutoff, 1, 0)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(churny_test, binary_predictions)

# Calculate specificity and sensitivity
true_negative = conf_matrix[0, 0]
false_positive = conf_matrix[0, 1]
false_negative = conf_matrix[1, 0]
true_positive = conf_matrix[1, 1]

specificity = true_negative / (true_negative + false_positive)
sensitivity = true_positive / (true_positive + false_negative)

# Print the specificity and sensitivity
print('Specificity:', specificity)
print('Sensitivity:', sensitivity)