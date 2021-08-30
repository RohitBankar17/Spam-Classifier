import numpy as np
import csv
import sys
import json,math

from collections import defaultdict 
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the model learned during training.
Writes the predicted values to the file named "predicted_test_Y_nb.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n', dtype=str)
    with open(model_file_path, "r") as read_file:
        model = json.load(read_file)
    return test_X, model



def compute_likelihood(test_X, c,class_wise_frequency_dict,class_wise_denominators):
    cc = defaultdict(int,class_wise_frequency_dict[c])
    lk = 0
    for x in test_X.split(" "):
        num = cc[x]+1
        lk += np.log(num/(class_wise_denominators[c]))
    return lk


def preprocessing(X):
    result = []
    for s in X:
        S = s
        for i in range(len(S)):
            if not(S[i].isalpha()) and not(S[i] == " "):
                s = s.replace(S[i], "")
        n = s.strip()
        n = n.lower()
        while not(n.find("  ") == -1):
            i = n.find("  ")
            n = n[0:i] + n[i+1::]
        result.append(n)
    return result
def predict_target_values(test_X, model):
    li=-math.inf
    best_class=-math.inf
    [classes,class_wise_frequency_dict,class_wise_denominators,prior_probabilities]=model
    ans=[]
    
    prior_probabilities=defaultdict(int,prior_probabilities)
    for t in test_X:  
        li=-math.inf
        for c in classes:
            temp=compute_likelihood(t,c,class_wise_frequency_dict,class_wise_denominators)+np.log(prior_probabilities[c])
            if temp>li:
                li=temp
                best_class=c
        ans.append(best_class)
    return np.array(ans)


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.
    """
    test_X, model = import_data_and_model(test_X_file_path, "./MODEL_FILE.json")
    pred_Y = predict_target_values(preprocessing(test_X), model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 