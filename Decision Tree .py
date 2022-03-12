import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Đọc dữ liệu từ file .csv
data_train = pd.read_csv("Training Data.csv")



data_train = data_train.drop(columns=['Id','CITY','STATE'], axis = 1)


col_train = data_train.columns[data_train.dtypes=='object']

for i in col_train:
    le = LabelEncoder()
    col_train = le.fit_transform(data_train[i])
    data_train[i] = col_train

X = data_train.iloc[:,0:9]
Y = data_train.iloc[:,9]

#print(pd.value_counts(Y))


for j in range(0,10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3.0, random_state=3)
    print("min_samples_leaf=",j+1, " max_depth=", j+5);
    
    tree = DecisionTreeClassifier(criterion= "entropy", random_state =0, min_samples_leaf=j+1, max_depth=j+5)
    tree.fit(X_train, Y_train)
    Y_pre = tree.predict(X_test)
    print("Accuracy Tree= ", np.round(accuracy_score(Y_test, Y_pre)*100,2))


acc_tree  = 0;
acc_bayes = 0;

for i in range(0,10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3.0, random_state=i)
    
    tree = DecisionTreeClassifier(criterion= "entropy", random_state =0, min_samples_leaf=10)
    tree.fit(X_train, Y_train)
    Y_pre = tree.predict(X_test)
    print("Accuracy Tree= ", np.round(accuracy_score(Y_test, Y_pre)*100,2))
    acc_tree += accuracy_score(Y_test, Y_pre);
    
    #Sử dụng giải thuật Naive Bayes
    bayes = GaussianNB()
    bayes.fit(X_train, Y_train)
    y_pred_bayes = bayes.predict(X_test)
    print("Accuracy Bayes= ", np.round(accuracy_score(Y_test, y_pred_bayes)*100,2))
    acc_bayes += accuracy_score(Y_test, y_pred_bayes);

        


print("Do chinh xac trung binh cua Decision Tree la: ", np.round(acc_tree/10, 2)*100)
print("Do chinh xac trung binh cua Naive Bayes la: ", np.round(acc_bayes/10, 2)*100)


