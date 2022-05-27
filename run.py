import pandas as pd
import joblib
import prepro as p


print("test model for by load")

dataTest = pd.read_csv("airline-test-samples.csv")

filenameDicision = "finalModelDicision.sav"
filenameKNearest = "finalModelKNearest.sav"
filenameRandomForest = "finalModelRandomForest.sav"

xTest ,yTest = p.preprocessingfunTest(dataTest)

loadDicisionModel = joblib.load(filenameDicision)
k = joblib.load(filenameKNearest)
f = joblib.load(filenameRandomForest)

pr=loadDicisionModel.predict(xTest)
pr1=k.predict(xTest)
pr2=f.predict(xTest)
print(loadDicisionModel.score(xTest,yTest))
print(k.score(xTest,yTest))
print(f.score(xTest,yTest))