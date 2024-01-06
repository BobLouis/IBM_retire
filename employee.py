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

for i in range(10):

    test_ratio = 0.2

    random_seed = random.randint(1, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42)






    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print(gnb.score(X_test, y_test))


    param_lr = {'penalty': ['none', 'l2'], 'C': [
        0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    grid_lr = GridSearchCV(LogisticRegression(), param_lr,
                        scoring='accuracy', n_jobs=-1, verbose=1, cv=5)
    grid_lr.fit(X_train, y_train)


    optimized_lr = grid_lr.best_estimator_
    optimized_lr


    grid_lr.best_score_


    grid_lr.best_params_


    lr_train_pred = optimized_lr.predict(X_train)
    lr_test_pred = optimized_lr.predict(X_test)



    print("The accuracy on train data is ", accuracy_score(lr_train_pred, y_train))
    logistic_accuracy = accuracy_score(lr_test_pred, y_test)
    print("The accuracy on test data is ", accuracy_score(lr_test_pred, y_test))
    print("The precision on test data is ", precision_score(lr_test_pred, y_test))
    print("The recall on test data is ", recall_score(lr_test_pred, y_test))
    print("The f1 on test data is ", f1_score(lr_test_pred, y_test))
    print("The roc_score on test data is ", roc_auc_score(lr_test_pred, y_test))


    labels = ['Not Defaulter', 'Defaulter']
    cm = confusion_matrix(y_test, lr_test_pred)




    y_preds_proba_lr = optimized_lr.predict_proba(X_test)[::, 1]
    y_pred_proba = y_preds_proba_lr
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.title('Logistic Regression')
    plt.legend(loc=4)
    # # plt.show()


    dt = DecisionTreeClassifier(max_depth=8, max_features=15, random_state=42)
    dt.fit(X_train, y_train)
    print(dt.score(X_test, y_test))

    param_dt = {
        # 'max_depth': [5, 10, 15],  # Maximum depth of each tree
        'max_depth': range(5, 15),  # Maximum depth of each tree
        # Minimum number of samples required to split an internal node
        'min_samples_split': range(2, 10),
        # Minimum number of samples required to be at a leaf node
        'min_samples_leaf': range(1, 4),
        'max_features': range(5, 18)
    }
    # param_dt = {
    #     'max_depth': [5, 10, 15],  # Maximum depth of each tree
    #     # Minimum number of samples required to split an internal node
    #     'min_samples_split': [2, 5, 10],
    #     # Minimum number of samples required to be at a leaf node
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': [5, 10, 15, 18]
    # }


    grid_dt = GridSearchCV(estimator=dt, param_grid=param_dt,
                        cv=4, scoring='accuracy', n_jobs=-1, verbose=1)

    grid_dt.fit(X_train, y_train)


    optimized_dt = grid_dt.best_estimator_


    grid_dt.best_score_


    grid_dt.best_params_


    dt_train_pred = optimized_dt.predict(X_train)
    dt_test_pred = optimized_dt.predict(X_test)


    print("The accuracy on train data is ", accuracy_score(dt_train_pred, y_train))
    decision_tree_accuracy = accuracy_score(dt_test_pred, y_test)
    print("The accuracy on test data is ", accuracy_score(dt_test_pred, y_test))
    print("The precision on test data is ", precision_score(dt_test_pred, y_test))
    print("The recall on test data is ", recall_score(dt_test_pred, y_test))
    print("The f1 on test data is ", f1_score(dt_test_pred, y_test))
    print("The roc_score on test data is ", roc_auc_score(dt_test_pred, y_test))


    labels = ['Not Defaulter', 'Defaulter']
    cm = confusion_matrix(y_test, dt_test_pred)




    y_preds_proba_dt = optimized_dt.predict_proba(X_test)[::, 1]
    y_pred_proba = y_preds_proba_dt
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.title('Decision Tree')
    plt.legend(loc=4)
    # plt.show()


    rf = RandomForestClassifier(max_depth=10, max_features=15, random_state=42)
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))

    # param_grid = {
    #     'n_estimators': [100, 200, 300, 350],  # Number of trees in the forest
    #     'max_depth': [5, 10, 15],  # Maximum depth of each tree
    #     # 'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    #     # 'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    #     'max_features': [5, 10, 15, 18]
    # }

    param_grid = {
        # Number of trees in the forest
        'n_estimators': [100, 170, 180, 190, 200, 210, 220, 230, 240, 300, 350],
        # Maximum depth of each tree
        'max_depth': [5, 10, 12, 13, 14, 15, 16, 17, 18],
        # 'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        # 'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        'max_features': [3, 4, 5, 6, 7, 8, 10, 15, 18]
    }
    # param_grid = {
    #     'n_estimators': range(300, 300),  # Number of trees in the forest
    #     'max_depth': range(5, 15),  # Maximum depth of each tree
    #     # 'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    #     # 'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    #     'max_features': range(5, 18)
    # }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                            cv=4, scoring='accuracy', n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)


    optimized_rf = grid_search.best_estimator_


    grid_search.best_score_


    grid_search.best_params_


    rf_train_pred = optimized_rf.predict(X_train)
    rf_test_pred = optimized_rf.predict(X_test)


    print("The accuracy on train data is ", accuracy_score(rf_train_pred, y_train))
    random_forest_accuracy = accuracy_score(rf_test_pred, y_test)
    print("The accuracy on test data is ", accuracy_score(rf_test_pred, y_test))
    print("The precision on test data is ", precision_score(rf_test_pred, y_test))
    print("The recall on test data is ", recall_score(rf_test_pred, y_test))
    print("The f1 on test data is ", f1_score(rf_test_pred, y_test))
    print("The roc_score on test data is ", roc_auc_score(rf_test_pred, y_test))


    labels = ['Not Defaulter', 'Defaulter']
    cm = confusion_matrix(y_test, rf_test_pred)




    y_preds_proba_rf = optimized_rf.predict_proba(X_test)[::, 1]
    y_pred_proba = y_preds_proba_rf
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.title('Random Forest')
    plt.legend(loc=4)
    # plt.show()


    xgb = XGBClassifier(objective='binary:logistic')
    # parm_xgb = {'max_depth': [15, 18, 22, 25], 'n_estimators': [
    #     250, 300, 400], 'learning_rate': [0.01, 0.05, 0.1]}
    parm_xgb = {'max_depth': [5, 6, 7, 8, 9, 10, 11, 12], 'n_estimators': [
        280, 290, 300, 305, 310, 315, 320], 'learning_rate': [0.05, 0.07, 0.09, 0.1, 0.11, 0.13, 0.15]}
    grid_xgb = GridSearchCV(estimator=xgb, param_grid=parm_xgb,
                            scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
    grid_result = grid_xgb.fit(X_train, y_train)


    optimized_xgb = grid_xgb.best_estimator_


    grid_xgb.best_score_


    xgb_best_params = grid_xgb.best_params_


    xgb_train_pred = optimized_xgb.predict(X_train)
    xgb_test_pred = optimized_xgb.predict(X_test)


    y_train.shape


    print("The accuracy on train data is ",
        accuracy_score(xgb_train_pred, y_train))
    xgboost_accuracy = accuracy_score(xgb_test_pred, y_test)
    print("The accuracy on test data is ", accuracy_score(xgb_test_pred, y_test))
    print("The precision on test data is ", precision_score(xgb_test_pred, y_test))
    print("The recall on test data is ", recall_score(xgb_test_pred, y_test))
    print("The f1 on test data is ", f1_score(xgb_test_pred, y_test))
    print("The roc_score on test data is ", roc_auc_score(xgb_test_pred, y_test))


    labels = ['Not Defaulter', 'Defaulter']
    cm = confusion_matrix(y_test, xgb_test_pred)




    y_preds_proba_xgb = optimized_xgb.predict_proba(X_test)[::, 1]
    y_pred_proba = y_preds_proba_xgb
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.title('XGBoost')
    plt.legend(loc=4)
    # plt.show()


    # Feature Importance
    feature_importances_xgb = pd.DataFrame(optimized_xgb.feature_importances_,
                                        index=columns,
                                        columns=['importance_xgb']).sort_values('importance_xgb',
                                                                                ascending=False)[:20]

    plt.subplots(figsize=(17, 6))
    plt.title("Feature importances from XGBoost")
    plt.bar(feature_importances_xgb.index, feature_importances_xgb['importance_xgb'],
            color="b",  align="center")
    plt.xticks(feature_importances_xgb.index, rotation=45)
    # plt.xlim([-1, X.shape[1]])
    # plt.show()

    
    # # NN




    MLP = MLPClassifier(solver='sgd', hidden_layer_sizes=(150, 100, 50), learning_rate='adaptive',
                        verbose=1, alpha=0.05, max_iter=200, n_iter_no_change=10, tol=0.0001,
                        activation='relu')
    MLP.fit(X_train, y_train)


    MLP_train_pred = MLP.predict(X_train)
    MLP_test_pred = MLP.predict(X_test)


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


    y_pred_proba = MLP.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.title('MLP')
    plt.legend(loc=4)
    # plt.show()


    # print models accuracy
    print('test_ratio', test_ratio)
    print("Logistic Regression Accuracy: ", logistic_accuracy)
    print("Decision Tree Accuracy: ", decision_tree_accuracy)
    print("Random Forest Accuracy: ", random_forest_accuracy)
    print("XGBoost Accuracy: ", xgboost_accuracy)
    print("MLP Accuracy: ", MLP_accuracy)

    logistic_accuracy_array.append(logistic_accuracy)
    decision_tree_accuracy_array.append(decision_tree_accuracy)
    random_forest_accuracy_array.append(random_forest_accuracy)
    xgboost_accuracy_array.append(xgboost_accuracy)
    MLP_accuracy_array.append(MLP_accuracy)




    with open('record.txt', 'a') as file:
        file.write(f'test_ratio: {test_ratio}\n')
        file.write(f"Logistic Regression Accuracy: {logistic_accuracy}\n")
        file.write(f"Decision Tree Accuracy: {decision_tree_accuracy}\n")
        file.write(f"Random Forest Accuracy: {random_forest_accuracy}\n")
        file.write(f"XGBoost Accuracy: {xgboost_accuracy}\n")
        file.write(f"MLP Accuracy: {MLP_accuracy}\n")

print('logistic_accuracy_avg', sum(logistic_accuracy_array) / len(logistic_accuracy_array))
print('decision_tree_accuracy_avg', sum(decision_tree_accuracy_array) / len(decision_tree_accuracy_array))
print('random_forest_accuracy_avg', sum(random_forest_accuracy_array) / len(random_forest_accuracy_array))
print('xgboost_accuracy_avg', sum(xgboost_accuracy_array) / len(xgboost_accuracy_array))
print("xgb_best_params", xgb_best_params)
print('MLP_accuracy_avg', sum(MLP_accuracy_array) / len(MLP_accuracy_array))

with open('record.txt', 'a') as file:
    file.write(f'average\n')
    file.write(f'logistic_accuracy_avg: {sum(logistic_accuracy_array) / len(logistic_accuracy_array)}\n')
    file.write(f'decision_tree_accuracy_avg: {sum(decision_tree_accuracy_array) / len(decision_tree_accuracy_array)}\n')
    file.write(f'random_forest_accuracy_avg: {sum(random_forest_accuracy_array) / len(random_forest_accuracy_array)}\n')
    file.write(f'xgboost_accuracy_avg: {sum(xgboost_accuracy_array) / len(xgboost_accuracy_array)}\n')
    file.write(f'MLP_accuracy_avg: {sum(MLP_accuracy_array) / len(MLP_accuracy_array)}\n')
