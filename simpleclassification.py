import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import streamlit as st
import os
import plotly.express as px

class classification:
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        

    def accuracy(self, confusion_matrix):
        sum, total = 0,0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[0])):
                if i == j: 
                    sum += confusion_matrix[i,j]
                total += confusion_matrix[i,j]
        return sum/total


    def classification_report_plot(self, clf_report):
        fig = px.imshow(pd.DataFrame(clf_report).iloc[:-1, :].T)
        st.plotly_chart(fig)
    
    
    def LR(self):
        from sklearn.linear_model import LogisticRegression
        lr_classifier = LogisticRegression()
        lr_classifier.fit(self.X_train, self.y_train)
        #joblib.dump(lr_classifier, "model/lr.sav")
        y_pred = lr_classifier.predict(self.X_test)

        st.write("\n")
        st.write("--------------------------------------")
        st.write("### Logistic Regression Classifier ###")
        st.write("--------------------------------------")
        st.write('Classification Report: ')
        clf = classification_report(self.y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(clf))
        st.write('Confusion Matrix: ')
        st.table(pd.DataFrame(confusion_matrix(self.y_test, y_pred)))
        st.write('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')

        self.classification_report_plot(clf)
        
        
        
    def KNN(self):
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier()
        knn_classifier.fit(self.X_train, self.y_train)
        #joblib.dump(knn_classifier, "model/knn.sav")
        y_pred = knn_classifier.predict(self.X_test)
        
        st.write("\n")
        st.write("-------------------------------")
        st.write("### K-Neighbors Classifier ###")
        st.write("-------------------------------")
        st.write('Classification Report: ')
        clf = classification_report(self.y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(clf))
        st.write('Confusion Matrix: ')
        st.table(pd.DataFrame(confusion_matrix(self.y_test, y_pred)))
        st.write('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')

        self.classification_report_plot(clf)
        
        
    
    # kernel type could be 'linear' or 'rbf' (Gaussian)
    def SVM(self, kernel_type):
        from sklearn.svm import SVC
        svm_classifier = SVC(kernel = kernel_type)
        svm_classifier.fit(self.X_train, self.y_train)
        #joblib.dump(svm_classifier, "model/svm.sav")
        y_pred = svm_classifier.predict(self.X_test)
        
        st.write("\n")
        st.write("--------------------------------------")
        st.write("### Support Vector Classifier (" + kernel_type + ") ###")
        st.write("--------------------------------------")
        st.write('Classification Report: ')
        clf = classification_report(self.y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(clf))
        st.write('Confusion Matrix: ')
        st.table(pd.DataFrame(confusion_matrix(self.y_test, y_pred)))
        st.write('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')

        self.classification_report_plot(clf)
        
        
        
    def NB(self):
        from sklearn.naive_bayes import GaussianNB
        nb_classifier = GaussianNB()
        nb_classifier.fit(self.X_train, self.y_train)
        #joblib.dump(nb_classifier, "model/nb.sav")
        y_pred = nb_classifier.predict(self.X_test)
        
        st.write("\n")
        st.write("------------------------------")
        st.write("### Naive Bayes Classifier ###")
        st.write("------------------------------")
        st.write('Classification Report: ')
        clf = classification_report(self.y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(clf))
        st.write('Confusion Matrix: ')
        st.table(pd.DataFrame(confusion_matrix(self.y_test, y_pred)))
        st.write('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')

        self.classification_report_plot(clf)
        

        
        
    def DT(self):
        from sklearn.tree import DecisionTreeClassifier
        tree_classifier = DecisionTreeClassifier()
        tree_classifier.fit(self.X_train, self.y_train)
        #joblib.dump(tree_classifier, "model/tree.sav")
        y_pred = tree_classifier.predict(self.X_test)
        
        st.write("\n")
        st.write("--------------------------------")
        st.write("### Decision Tree Classifier ###")
        st.write("--------------------------------")
        st.write('Classification Report: ')
        clf = classification_report(self.y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(clf))
        st.write('Confusion Matrix: ')
        st.table(pd.DataFrame(confusion_matrix(self.y_test, y_pred)))
        st.write('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')

        self.classification_report_plot(clf)
        

        
        
    def RF(self):
        from sklearn.ensemble import RandomForestClassifier
        rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
        rf_classifier.fit(self.X_train, self.y_train)
        #joblib.dump(rf_classifier, "model/rf.sav")
        y_pred = rf_classifier.predict(self.X_test)
        
        st.write("\n")
        st.write("--------------------------------")
        st.write("### Random Forest Classifier ###")
        st.write("--------------------------------")
        st.write('Classification Report: ')
        clf = classification_report(self.y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(clf))
        st.write('Confusion Matrix: ')
        st.table(pd.DataFrame(confusion_matrix(self.y_test, y_pred)))
        st.write('Precision: ', self.accuracy(confusion_matrix(self.y_test, y_pred))*100,'%')

        self.classification_report_plot(clf)

def st_classification():
    df = pd.read_csv("temp_data/test.csv")

    # select features/columns
    col_names = []
    col_names_y=[]
    feature_list = list(df.columns)

    st.sidebar.write("Select Feature Columns- X)")
    for col_name in feature_list:
        check_box1 = st.sidebar.checkbox(col_name,key= 0)
        if check_box1:
            col_names.append(col_name)

    st.sidebar.write("Select Target Column- Y)")
    for col_name in feature_list:
        check_box2 = st.sidebar.checkbox(col_name,key=1)
        if check_box2:
            col_names_y.append(col_name)

    df_X = df[col_names]
    df_Y = df[col_names_y]

    testSize=st.sidebar.slider("Enter Test Data Size (default 0.2)", 0.0,0.4,0.2,0.1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_X,df_Y, test_size = testSize, random_state = 0)


    model = st.sidebar.selectbox(
                'Choose Model', ["LR", "KNN", "SVM", "NB", "DT", "RF"])

    classifier = classification(X_train, X_test, y_train, y_test)

    if model == "LR":
        try:
            classifier.LR()
        except Exception as e:
            st.write(e)

    if model == "KNN":
        try:
            classifier.KNN()
        except Exception as e:
            st.write(e)

    if model == "SVM":
        kernel_choice = st.sidebar.selectbox('Select Feature Selection Method',\
                                                ["linear", "rbf"])
        try:
            classifier.SVM(kernel_choice)
        except Exception as e:
            st.write(e)

    if model == "NB":
        try:
            classifier.NB()
        except Exception as e:
            st.write(e)

    if model == "DT":
        try:
            classifier.DT()
        except Exception as e:
            st.write(e)

    if model == "RF":
        try:
            classifier.RF()
        except Exception as e:
            st.write(e)

    