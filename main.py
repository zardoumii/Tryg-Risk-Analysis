import sys
import os
from util.basic_analysis import run_complete_analysis
from util.advanced_analysis import advanced_analysis
from util.pca import rpca
from util.clustering import cluster_data
from M11 import train_and_evaluate_decision_tree
from M12 import train_and_evaluate_sklearn_tree
from M21 import train_and_evaluate_custom_ffnn
from M22 import train_and_evaluate_sklearn_ffnn
from M30 import train_and_evaluate_gbm

"""Main script to run data processing, model training, and evaluation CHECK README FOR DETAILS"""

filepathtrain = "data/claims_train.csv"
filepathtest = "data/claims_test.csv"
datasettest = run_complete_analysis(filepathtest)
print(f"Dataset successfully cleaned. Final shape: {datasettest.shape}")
datasettrain = run_complete_analysis(filepathtrain)
print(f"Dataset successfully cleaned. Final shape: {datasettrain.shape}")

model, y_test, y_pred_test = train_and_evaluate_decision_tree()
print("Train and Tested successfully.")

model, y_test, y_pred_test = train_and_evaluate_sklearn_tree()
print("Train and Tested successfully.")

advanced_analysis()
rpca()
cluster_data()

model, y_test, y_test_pred = train_and_evaluate_custom_ffnn()
print("Train and Tested successfully.")

model, y_test, y_test_pred = train_and_evaluate_sklearn_ffnn()
print("Train and Tested successfully.")

train_and_evaluate_gbm()
print("Everything ran successfully.")