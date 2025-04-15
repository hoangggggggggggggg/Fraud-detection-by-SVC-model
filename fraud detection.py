import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
credit_card_data = pd.read_csv("C:\\Users\\laptop\\Downloads\\archive_fraud\\creditcard.csv")
credit_card_data.head()
credit_card_data.isnull().sum()
credit_card_data['Class'].value_counts()
credit_card_data = credit_card_data.drop("Time", axis=1)


import imblearn
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy=0.5)
cols = credit_card_data.columns.tolist()
cols = [c for c in cols if c not in ["Class"]]
target = "Class"
#define X and Y
X = credit_card_data[cols]
Y = credit_card_data[target]

#undersample
X_under, Y_under = undersample.fit_resample(X, Y)
from pandas import DataFrame
test = pd.DataFrame(Y_under, columns = ['Class'])
#visualizing undersampling results
fig, axs = plt.subplots(ncols=2, figsize=(13,4.5))
sns.countplot(x="Class", data=credit_card_data, ax=axs[0])
sns.countplot(x="Class", data=test, ax=axs[1])

fig.suptitle("Class repartition before and after undersampling")
a1=fig.axes[0]
a1.set_title("Before")
a2=fig.axes[1]
a2.set_title("After")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size=0.2, random_state=1, stratify=Y_under)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
#standard scaling
X_train['std_Amount'] = scaler.fit_transform(X_train['Amount'].values.reshape (-1,1))
X_test['std_Amount'] = scaler.transform(X_test['Amount'].values.reshape(-1, 1))
#removing Amount
X_test = X_test.drop("Amount", axis=1)
X_train = X_train.drop("Amount", axis = 1)
sns.countplot(x="Class", data=credit_card_data)
from sklearn.svm import SVC

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

#train the model
model = SVC(probability=True, random_state=2)
svm = model.fit(X_train, y_train)
#predictions
y_pred_svm = model.predict(X_test)
#scores
print("Accuracy SVM:",metrics.accuracy_score(y_test, y_pred_svm))
print("Precision SVM:",metrics.precision_score(y_test, y_pred_svm))
print("Recall SVM:",metrics.recall_score(y_test, y_pred_svm))
print("F1 Score SVM:",metrics.f1_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))



