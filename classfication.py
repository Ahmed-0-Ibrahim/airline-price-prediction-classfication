from calendar import month
from re import X
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
#from sympy import true
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# import plotly.express as px
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
import joblib


data = pd.read_csv("airline-price-classification.csv")
dataTest = pd.read_csv("airline-test-samples.csv")

print(data.head())
def preprocessingfun(dtatSet):
    proc = preprocessing.LabelEncoder()
    
    dtatSet['TicketCategory'] = proc.fit_transform(dtatSet['TicketCategory'])
    print(dtatSet)
    
    x = dtatSet.iloc[:, :-1]  # Futures
    y = dtatSet.iloc[:, -1]  # Goal
    
    # print(y)
    
    
    # drop airline // ch_code == airline
    x.drop(["airline"], axis=1, inplace=True)
    
    
    # drop arr_time // arr_time == dept_time + time_taken
    x.drop(["arr_time"], axis=1, inplace=True)
    
    
    #  dates preprocessing
    dates = dtatSet["date"].str.replace("/", "-")
    dates = dates.str.replace("-0", "-")
    
    days = dates.str.split("-", expand=True)[0]
    days = pd.to_numeric(days, downcast="integer")
    days = days.to_frame()
    days.columns = ['days']
    
    months = dates.str.split("-", expand=True)[1]
    months = pd.to_numeric(months, downcast="integer")
    months = months.to_frame()
    months.columns = ['months']
    # dates = days + "/" + months
    # x["date"] = dates
    x.drop(["date"], axis=1, inplace=True)
    x = pd.concat([x, days, months], axis=1)
    # print(x)
    
    
    # stop preprocessing
    stop = dtatSet['stop'].str.strip()
    stop.replace("\s+",'',regex=True, inplace = True)
    stop.replace('^non.*','0',regex=True, inplace = True)
    stop.replace('^1-st.*','1',regex=True, inplace = True)
    stop.replace('^2.*','2',regex=True, inplace = True)
    stop = pd.to_numeric(stop, downcast="integer")
    x['stop'] = stop
    
    
    # dep_time preprocessing
    hours = dtatSet["dep_time"].str.split(":", expand=True)[0]
    minutes = dtatSet["dep_time"].str.split(":", expand=True)[1]
    for i in range(len(minutes)):
        minutes[i] = float(minutes[i]) / 60
        hours[i] = float(hours[i]) + float(minutes[i])
    
    # print(hours)
    hours = pd.to_numeric(hours, downcast="float")
    x["dep_time"] = hours
    # print(x)
    
    
    # time_taken preprocessing
    hours = dtatSet["time_taken"].str.split("h ", expand=True)[0]
    minutes = dtatSet["time_taken"].str.split("h ", expand=True)[1]
    minutes = minutes.str.split("m", expand=True)[0]
    minutes = minutes.str.replace("", "0")
    
    for i in range(len(minutes)):
        hours[i] = float(hours[i]) * 60
        minutes[i] = float(minutes[i]) + hours[i]
    
    minutes = pd.to_numeric(minutes, downcast="float")
    x["time_taken"] = minutes
    
    # Encoding dtatSet
    x_obj = x.select_dtypes(include=["object"])
    x_non_obj = x.select_dtypes(exclude=["object"])
    la = LabelEncoder()
    
    for i in range(x_obj.shape[1]):
        x_obj.iloc[:, i] = la.fit_transform(x_obj.iloc[:, i])
    
    x_encoded = pd.concat([x_non_obj, x_obj], axis=1)
    print('X encoded\n', x_encoded)
    # x = x_encoded
    
    # correlation
    corr = x_encoded.corrwith(y)
    print('correlation with TicketCategory\n', corr)
    
    
    
    final_data = pd.concat([x_encoded, y], axis=1)
    corr = final_data.corr()
    topFeature = corr.index[abs(corr['TicketCategory']) > 0.05]
    top_corr = final_data[topFeature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    topFeature = topFeature.delete(-1)
    x = x_encoded[topFeature]
    print('feature selection with correlation\n', x.head())
    return x,y
# Split the data to training and testing sets
x,y=preprocessingfun(data)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, shuffle=True)


# #Logistic Regression Model
# logreg = LogisticRegression()
# #fit the data model
# logreg.fit(X_train, y_train)
#
# #predict
# y_predict = logreg.predict(X_test)
#
# print("Accuracy using Logistic Regression = ", metrics.accuracy_score(y_test, y_predict))

# #Suppor vector machine
# svm_linear_ovo = OneVsOneClassifier(LinearSVC(C=0.4),).fit(X_train, y_train)
# svm_linear_ovr = OneVsRestClassifier(LinearSVC(C=1)).fit(X_train, y_train)

# y_predict = svm_linear_ovo.predict(X_test)
# print('svm_linear_ovo')
# print("Mean Square Error", metrics.mean_squared_error(y_test, y_predict))
# # print("True value  in the test set: " + str(trueValue))
# # print("Predicted value  in the test set: " + str(predictedValue))
# print('score:', r2_score(y_test, y_predict))
# print("Accuracy using svm_linear_ovo= ", metrics.accuracy_score(y_test, y_predict))
# print('---------------------------------------------------------------')

# # model accuracy for svc model
# accuracy = svm_linear_ovr.score(X_test, y_test)
# print('LinearSVC OneVsRest SVM accuracy: ' + str(accuracy))
# accuracy = svm_linear_ovo.score(X_test, y_test)
# print('LinearSVC OneVsOne SVM accuracy: ' + str(accuracy))

#Decision Tree Model
filenameDicision = "finalModelDicision.sav"
dicision = tree.DecisionTreeClassifier()
dicision = dicision.fit(X_train, y_train)
joblib.dump(dicision, filenameDicision)
tree.plot_tree(dicision)
# loadDicisionModel = joblib.load(filenameDicision)
y_predict   = dicision.predict(X_test)
# y_predictload = loadDicisionModel.predict(X_test)

for i in range(20):
    trueValue = np.asarray(y_test)[i]
    predictedValue = y_predict[i]
    print (trueValue ,' : ',predictedValue)
    
print('dicision tree')
print("Mean Square Error", metrics.mean_squared_error(y_test, y_predict))
print("True value  in the test set: " + str(trueValue))
print("Predicted value  in the test set: " + str(predictedValue))
print('score:', r2_score(y_test, y_predict))
print("Accuracy using Decision Tree = ", metrics.accuracy_score(y_test, y_predict))
print('---------------------------------------------------------------')




# #Random Forest Classifier
filenameRandomForest = "finalModelRandomForest.sav"
rand_forest = RandomForestClassifier()
rand_forest.fit(X_train, y_train)
joblib.dump(rand_forest, filenameRandomForest)
y_predict = rand_forest.predict(X_test)

print('Random Forest Classifier')
print("Mean Square Error", metrics.mean_squared_error(y_test, y_predict))
print("True value  in the test set: " + str(trueValue))
print("Predicted value  in the test set: " + str(predictedValue))
print('score:', r2_score(y_test, y_predict))
print("Accuracy using Random Forest = ", metrics.accuracy_score(y_test, y_predict))
print('---------------------------------------------------------------')

# #Gradient Boost Classifier
# gb = GradientBoostingClassifier()
# gb.fit(X_train, y_train)
# y_predict = gb.predict(X_test)
# print("Accuracy using Gradient Boost = ", metrics.accuracy_score(y_test, y_predict))


# #Gaussian Naive Bayes
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_predict = gnb.predict(X_test)
# print("Accuracy using Guassian Naive Bayes = ", metrics.accuracy_score(y_test, y_predict))


#K Nearest neighbours
filenameKNearest = "finalModelKNearest.sav"
neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)
joblib.dump(neigh, filenameKNearest)
y_predict = neigh.predict(X_test)

print('K Nearest neighbours')
print("Mean Square Error", metrics.mean_squared_error(y_test, y_predict))
print("True value  in the test set: " + str(trueValue))
print("Predicted value  in the test set: " + str(predictedValue))
print('score:', r2_score(y_test, y_predict))
print("Accuracy using K Nearest Neighbours = ", metrics.accuracy_score(y_test, y_predict))
print('---------------------------------------------------------------')
#print("\ncompleted\n")

#svm_kernel_ovo = OneVsOneClassifier(SVC(kernel='linear', C=1)).fit(X_train, y_train)
# svm_kernel_ovr = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_train, y_train)

# svm_linear_ovo = OneVsOneClassifier(LinearSVC(C=1),).fit(X_train, y_train)
# svm_linear_ovr = OneVsRestClassifier(LinearSVC(C=1)).fit(X_train, y_train)

# model accuracy for svc model
# accuracy = svm_kernel_ovr.score(X_test, y_test)
# print('Linear Kernel OneVsRest SVM accuracy: ' + str(accuracy))
#accuracy = svm_kernel_ovo.score(X_test, y_test)
#print('Linear Kernel OneVsOne SVM accuracy: ' + str(accuracy))

# model accuracy for svc model
# accuracy = svm_linear_ovr.score(X_test, y_test)
# print('LinearSVC OneVsRest SVM accuracy: ' + str(accuracy))
# accuracy = svm_linear_ovo.score(X_test, y_test)
# print('LinearSVC OneVsOne SVM accuracy: ' + str(accuracy))

xTest ,yTest = preprocessingfun(dataTest)
d = joblib.load(filenameDicision)
k = joblib.load(filenameKNearest)
f = joblib.load(filenameRandomForest)

pr=d.predict(xTest)
print(pr," : ",yTest)
pr=k.predict(xTest)
print(pr," : ",yTest)
pr=f.predict(xTest)
print(pr," : ",yTest)

