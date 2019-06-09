import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection

def predict():

  # Read the whole data from CSV
  data = pd.read_csv("math-students.csv", sep=";")

  # The column whose values must be predicted
  predict_column = "G3"

  # Limit the data in a set of columns. These are 
  # the important columns for this test
  study_data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
  
  # Set the features (attributes) and the label
  # for the test. The label is the data we want
  # to predict
  features = np.array(study_data.drop([predict_column], axis=1))
  labels = np.array(data[predict_column])

  # Given the whole read data, split it into a set 
  # of trains and tests.
  # Trains: the data used to train the model
  # Tests: the data used to compare the results later
  features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size = 0.1)
  
  # Train the model drawing the line using Linear 
  # Regression concept. We use different concepts 
  # for different data (entry) formats
  linear = linear_model.LinearRegression()
  linear.fit(features_train, labels_train)

  # Check the accuracy of our model by comparing 
  # the results with the expected data (tests)
  accuracy = linear.score(features_test, labels_test)

  # Since we now have our model trained, we predict 
  # the labels (test ones) based on the features 
  # (also the test ones)
  predictions = linear.predict(features_test)

  # Print details
  for i in range(len(predictions)):
    print("features: .........", features_test[i])
    print("expected label: ...", labels_test[i])
    print("predicted label: ..", predictions[i])
    print("-")
  print("Accuracy:", (accuracy * 100), "%")

# Execute the program
predict()