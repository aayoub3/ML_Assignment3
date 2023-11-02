# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# Importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

# hyperparameter values
c_values = [1, 5, 10, 100]
degree_values = [1, 2, 3]
kernel_values = ["linear", "poly", "rbf"]
decision_function_shape_values = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) 

X_training = np.array(df.values)[:, :64] 
y_training = np.array(df.values)[:, -1] 

df = pd.read_csv('optdigits.tes', sep=',', header=None) 

X_test = np.array(df.values)[:, :64]
y_test = np.array(df.values)[:, -1] 

highest_accuracy = 0.0
best_parameters = {}

for c in c_values:
    for degree in degree_values:
        for kernel in kernel_values:
            for decision_function_shape in decision_function_shape_values:
                clf = svm.SVC(C=c, degree=degree, kernel=kernel, decision_function_shape=decision_function_shape)

                clf.fit(X_training, y_training)

                correct_predictions = 0
                total_predictions = len(X_test)

                for x_testSample, y_testSample in zip(X_test, y_test):
                    prediction = clf.predict([x_testSample])
                    if prediction == y_testSample:
                        correct_predictions += 1

                accuracy = correct_predictions / total_predictions

                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_parameters = {
                        'C': c,
                        'degree': degree,
                        'kernel': kernel,
                        'decision_function_shape': decision_function_shape
                    }
                    print(f"Highest SVM accuracy: {highest_accuracy:.2f}, Parameters: {best_parameters}")