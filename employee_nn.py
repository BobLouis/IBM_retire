from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import random


data = pd.read_csv("./data/HR-Employee.csv")




data.info()


data.describe()
data.shape
print(data.columns)


# drop the EmployeeCount column
data.drop('EmployeeCount', axis=1, inplace=True)
# drop the EmployeeNumber column
data.drop('EmployeeNumber', axis=1, inplace=True)
data.drop('Over18', axis=1, inplace=True)  # drop the EmployeeNumber column


data.duplicated().sum()  # check for duplicates


data.Attrition.value_counts()


data.Gender.value_counts()


data.Education.value_counts()


data.MaritalStatus.value_counts()







plt.figure(figsize=(50, 7))






# train = data[(data['AGE'] <= 60) & (data['PAY_0'] <=2) & (data['PAY_2'] <=2) & (data['PAY_3'] <=2) & (data['PAY_4'] <=2) & (data['PAY_5'] <=2) & (data['PAY_6'] <=2) & (data['LIMIT_BAL'] <=600000)]
train = data


train.info()


train.Attrition.value_counts()



train = pd.get_dummies(train, columns=['BusinessTravel', 'Department', 'EducationField', 'JobRole',
                       'MaritalStatus'], dtype=int, drop_first=True)  # change education and marriage to categorical variables


train.Attrition.value_counts()


encoders_nums = {
    "Gender": {"Female": 0, "Male": 1},
    "OverTime": {"No": 0, "Yes": 1},
    "Attrition": {"No": 0, "Yes": 1},

}
train = train.replace(encoders_nums)





X = train.drop('Attrition', axis=1)
y = train.Attrition


y.value_counts()


columns = train.drop('Attrition', axis=1).columns






sm = SMOTE(random_state=42)

X, y = sm.fit_resample(X, y)  # make the x y balanced


y.value_counts()


X.value_counts()


scaler = StandardScaler()
# scale the data to make it easier for the model to learn
X = scaler.fit_transform(X)



logistic_accuracy_array = []
decision_tree_accuracy_array = []
random_forest_accuracy_array = []
xgboost_accuracy_array = []
MLP_accuracy_array = []

for i in range(5):

    test_ratio = 0.2

    random_seed = random.randint(1, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42)



    
    # # NN




    # MLP = MLPClassifier(solver='sgd', hidden_layer_sizes=(150, 100, 50), learning_rate='adaptive',
    #                     verbose=1, alpha=0.05, max_iter=200, n_iter_no_change=10, tol=0.0001,
    #                     activation='relu')
    # MLP.fit(X_train, y_train)

    param_grid = {
        'hidden_layer_sizes': [(100,), (150, 100, 50), (200, 100)],
        'alpha': [0.0001, 0.001, 0.01, 0.05],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'activation': ['tanh', 'relu']
    }

    # Create an MLPClassifier instance
    mlp = MLPClassifier(solver='sgd', max_iter=200, n_iter_no_change=10, tol=0.0001, verbose=1)

    # Create a GridSearchCV instance
    grid_mlp = GridSearchCV(mlp, param_grid, scoring='accuracy', n_jobs=-1, verbose=1, cv=5)

    # Fit the grid search to the data
    grid_mlp.fit(X_train, y_train)

    optimized_mlp = grid_mlp.best_estimator_
    


    print("mlp best param",grid_mlp.best_params_)

    MLP_train_pred = optimized_mlp.predict(X_train)
    MLP_test_pred = optimized_mlp.predict(X_test)


    print("The accuracy on train data is ",
        accuracy_score(MLP_train_pred, y_train))
    MLP_accuracy = accuracy_score(MLP_test_pred, y_test)
    print("The accuracy on test data is ", accuracy_score(MLP_test_pred, y_test))
    print("The precision on test data is ", precision_score(MLP_test_pred, y_test))
    print("The recall on test data is ", recall_score(MLP_test_pred, y_test))
    print("The f1 on test data is ", f1_score(MLP_test_pred, y_test))
    print("The roc_score on test data is ", roc_auc_score(MLP_test_pred, y_test))



    labels = ['Not Defaulter', 'Defaulter']
    cm = confusion_matrix(y_test, MLP_test_pred)

    y_pred_proba = optimized_mlp.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.title('MLP')
    plt.legend(loc=4)
    # plt.show()


    # print models accuracy
    print('test_ratio', test_ratio)
    print("MLP Accuracy: ", MLP_accuracy)

    MLP_accuracy_array.append(MLP_accuracy)




    with open('record.txt', 'a') as file:
        file.write(f'test_ratio: {test_ratio}\n')

        file.write(f"MLP Accuracy: {MLP_accuracy}\n")



print('MLP_accuracy_avg', sum(MLP_accuracy_array) / len(MLP_accuracy_array))

with open('record_nn.txt', 'a') as file:
    file.write(f'average\n')
    file.write(f'MLP_accuracy_avg: {sum(MLP_accuracy_array) / len(MLP_accuracy_array)}\n')
