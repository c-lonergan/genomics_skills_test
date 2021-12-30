# import packages
import xgboost as xgb
import pandas as pd
# read data
train = pd.read_csv("data/wisconsin_training.csv", header=0, index_col=False)
test = pd.read_csv("data/wisconsin_testing.csv", header=0, index_col=False)
# remove index
train = train.drop(labels="Unnamed: 0", axis=1)
test = test.drop(labels="Unnamed: 0", axis=1)
# define labels and predictor data
y_train = train["diagnosis"]
x_train = train.iloc[:,1:31]
x_test = test.iloc[:,1:31]
# fit model 
model = xgb.XGBClassifier()
model.fit(x_train, y_train)
# predict test set results
y_pred = model.predict(x_test)
# write predictions to csv
y_pred = pd.DataFrame(y_pred) 
y_pred.to_csv("./data/wisconsin_predictions.csv")
