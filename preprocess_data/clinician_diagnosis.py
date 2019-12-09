"""
This scripts is to get the accuracy, F1 score and confusion matrix of clinician diagnosis
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def read_features(data_path, early_stage_definition):
    """
    Read feature files.
    Args:
        - data_path: directory containing all feature files
        - early_stage_definition: "all_visits_features", "first_visit_features", or "last_visit_features"
    Returns:
        - train_features, valid_features, test_features, features_concat
    """
    data_path = data_path + '/' + early_stage_definition + '/'
    
    with open(data_path+"train_"+early_stage_definition+'.csv', "rt") as fin:
        train_features = pd.read_csv(fin, low_memory=False)
    
    with open(data_path+"valid_"+early_stage_definition+'.csv', "rt") as fin:
        valid_features = pd.read_csv(fin, low_memory=False)    
    
    with open(data_path+"test_"+early_stage_definition+'.csv', "rt") as fin:
        test_features = pd.read_csv(fin, low_memory=False)

    return train_features, valid_features, test_features, pd.concat([train_features, valid_features, test_features])


def label_clinical_diagnosis(row):
    """
    Define the clinical diagnosis
    Args: 
        - row: pandas row
    """
    if (row["NACCALZD"] == 1) and (row["NACCLBDE"] != 1):
        return 0
    elif (row["NACCALZD"] != 1) and (row["NACCLBDE"] == 1):
        return 1
    elif (row["NACCALZD"] == 1) and (row["NACCLBDE"] == 1):
        return 2
    else:
        return 3    


def feature_statistics(features_csv):
    """
    Given features csv, return accuracy, F1 score and confusion matrix
    """
    # add clinician diagnosis column
    features_csv["clinician diagnosis"] = features_csv.apply(lambda row: label_clinical_diagnosis(row), axis=1)
    
    # get values of two columns
    clinician_diagnosis = features_csv["clinician diagnosis"].values
    label = features_csv["label"].values
    
    # get accuracy, F1 score and confusion matrix
    mtx = confusion_matrix(label, clinician_diagnosis)
    f1_macro = f1_score(label, clinician_diagnosis, average="macro")
    f1_micro = f1_score(label, clinician_diagnosis, average="micro")
    f1 = f1_score(label, clinician_diagnosis, average=None)
    mtx = confusion_matrix(label, clinician_diagnosis)
    
    return f1_macro, f1_micro, f1, mtx

def get_feature_statistics(data_path, early_stage_definition):
    """
    Get all statistics for all three level csv.
    """
    # read in the features
    train_features, valid_features, test_features, features_concat = read_features(data_path, early_stage_definition)

    # get the feature statistics for three level
    overall_f1_macro, overall_f1_micro, overall_f1, overall_mtx = feature_statistics(features_concat)
    valid_f1_macro, valid_f1_micro, valid_f1, valid_mtx = feature_statistics(valid_features)
    test_f1_macro,  test_f1_micro, test_f1, test_mtx = feature_statistics(test_features)

    # print the information 
    print('*'*40 + early_stage_definition + '*'*40 + '\n')
    print("overall F1_macro: {} | overall F1_micro: {} | overall F1: {}".format(overall_f1_macro, overall_f1_micro, overall_f1))
    print("overall mtx:")
    print(overall_mtx)
    print('\n')

    print("valid F1_macro: {} | valid F1_micro: {} | valid F1: {}".format(valid_f1_macro, valid_f1_micro, valid_f1))
    print("valid mtx:")
    print(valid_mtx)
    print('\n')

    print("test F1_macro: {} | test F1_micro: {} | test F1: {}".format(test_f1_macro, test_f1_micro, test_f1))
    print("test mtx:")
    print(test_mtx)
    print('\n')


if __name__ == "__main__":

    data_path = "./processed_data/processed_csv"
    
    definitions = ["all_visits_features", "first_visit_features", "last_visit_features"]

    for early_def in definitions:
        get_feature_statistics(data_path, early_def)
    
