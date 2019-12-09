"""
This script contains all steps of doing the feature engineering.

The strategy is:
(1) for continuous columns, normalize them first according to training statistics and keep as it is.
(2) for categorical columns, encode them as one-hot vectors. 
"""

import numpy as np
import pandas as pd
import pickle


def normalize_continuous_data(continuous_columns, train_csv, valid_csv, test_csv):
    """
    Normalize the continuous data according to training statistics. 
    Then concatenate three sets together for the next step.
    
    Args:
        - continuous_columns: python list containing continuous column names
        - train_csv: training rows after filling out the missing data
        - valid_csv: validation rows after filling out the missing data
        - test_csv: testing rows after filling out the missing data 
    Returns:
        - nacc_csv_concat: concatenation of train_csv, valid_csv and test_csv after normalization    
    """
    ### get all continuous columns names of train_csv ###
    train_continuous_cols = train_csv[continuous_columns]
    
    ### Normalize three datasets using statistics of the training dataset ###
    # first store the the mean and std of the training dataset, since after we normalize it, 
    # their values are changed
    train_csv_mean = train_csv[continuous_columns].mean()
    train_csv_std = train_csv[continuous_columns].std()
    # normalize three sets
    train_csv[continuous_columns] = (train_csv[continuous_columns] - train_csv_mean) / train_csv_std 
    valid_csv[continuous_columns] = (valid_csv[continuous_columns] - train_csv_mean) / train_csv_std 
    test_csv[continuous_columns] = (test_csv[continuous_columns] - train_csv_mean) / train_csv_std
    
    ### concatenate them together ###
    nacc_csv_concat = pd.concat([train_csv, valid_csv, test_csv])

    return nacc_csv_concat
    #return train_csv, valid_csv, test_csv

def make_features_for_each_patient(column_names, continuous_names, categorical_values, nacc_csf_csv):
    """
    Do the feature engineering for each patient.
    Args:
        - column_names: all columns names in nacc_csf_csv
        - continuous_names: list of all continuous column names in nacc_csf_csv
        - categorical_values: dictionary containing categorical names and the corresponding possible values
        - nacc_csf_csv: normalized and concatenate nacc_csf_csv rows
    Returns:
        - nacc_csf_features_each_record: list of format [{"NACCID": ..., "records":[{"DATE": ..., "feature": ...}]}]
    """    
    # preprocess the column names
    column_names.remove("NACCID")
    column_names.remove("DATE")

    # do the feature engineering and store them into a dictionary
    nacc_csf_features_each_record = []  # list storing the features of all records for all patients
    
    for patient in pd.unique(nacc_csf_csv["NACCID"]):
        # for each patient 
        patient_records = nacc_csf_csv[nacc_csf_csv["NACCID"]==patient]
        patient_records_dict = {"NACCID": patient, "records":[]}  # dictionary storing the features of all records for one patient
        
        for index in patient_records.index:
            # for each record
            record_dict = {"DATE": patient_records.loc[index, :]["DATE"]}
            record = patient_records.loc[index, :]
            record_feature = np.array([])  # numpy array to store the feature for the record
            
            for col in column_names:
                if col in continuous_names:
                    # if it is continuous, then just keep as it is
                    record_feature = np.concatenate((record_feature, np.array([record[col]])), axis=0)
                else:
                    # if it is categorical, then compose a one-hot vector for it
                    column_values = categorical_values[col]
                    print(patient, index, col, column_values, record[col])
                    idx = int(np.where(column_values==record[col])[0])
                    # create one-hot vector
                    one_hot = np.zeros(column_values.shape[0])
                    one_hot[idx] = 1.
                    record_feature = np.concatenate((record_feature, one_hot), axis=0)
            
            record_dict["feature"] = record_feature  # feature for each record
            patient_records_dict["records"].append(record_dict) 
        
        nacc_csf_features_each_record.append(patient_records_dict)

    return nacc_csf_features_each_record   


def change_feature_format(nacc_csf_features_each_record, train_labels, valid_labels, test_labels):
    """
    Change the format of features by adding labels for each record.
    Args:
        - nacc_csf_features_each_record: the original format dictionary storing the nacc_csf_features
        - train_labels, valid_labels, test_labels: labels for train, valid or test set.
    Returns:
        - train_features, valid_features, test_features: lists of format: [(feature_vector, label)]
    """
    # concatenate all labels
    labels_concat = pd.concat([train_labels, valid_labels, test_labels])
    
    # get patient list for each set
    train_patient_list = list(pd.unique(train_labels["NACCID"]))
    valid_patient_list = list(pd.unique(valid_labels["NACCID"]))
    test_patient_list = list(pd.unique(test_labels["NACCID"]))

    # construct train, valid, test, feature list
    train_feature_list = []
    valid_feature_list = []
    test_feature_list = []

    # change the format for each record
    for patient_records in nacc_csf_features_each_record:
        # find the label for each patient
        patient_id = patient_records["NACCID"] 
        # if we contain all the patient record, we need to continue to use DATE to get the corresponding label
        for record in patient_records["records"]:
            diagnosis_date = record["DATE"]
            label = labels_concat.loc[(labels_concat["NACCID"] == patient_id) & (labels_concat["DATE"] == diagnosis_date), "label"].values[0]
            # combine feature and label
            if patient_id in train_patient_list:
                train_feature_list.append((record["feature"], label))
            elif patient_id in valid_patient_list:
                valid_feature_list.append((record["feature"], label))
            elif patient_id in test_patient_list:
                test_feature_list.append((record["feature"], label))
            else:
                raise ValueError("{} has no label!!".format(patient_id))

    return train_feature_list, valid_feature_list, test_feature_list                         
    

def feature_engineering(column_names, continuous_names, categorical_values, train_csv, valid_csv, test_csv, train_labels, valid_labels, test_labels):
    """
    The aggregated feature engineering steps.
    Args:
        - column_names: list containing all column names
        - continuous_names: list containing all continuous names
        - categorical_values: dictionary containing categorical names and the values of them
        - train_csv, valid_csv, test_csv: rows after filling out the missing data
        - train_labels, valid_labels, test_labels: labels for three sets
    Returns:
        - train_features, valid_features, test_features: list containing all features 
          as format [(feature_vector, label)]
    """
    ### normalize the continuous columns for three sets  ###
    nacc_csf_concat = normalize_continuous_data(continuous_names, train_csv, valid_csv, test_csv)

    ### make features for each patient ###
    nacc_csf_features = make_features_for_each_patient(column_names, continuous_names, categorical_values, nacc_csf_concat)

    ### change the format and split the features into three sets ###
    train_features, valid_features, test_features = change_feature_format(nacc_csf_features, train_labels, valid_labels, test_labels)    
    
    return train_features, valid_features, test_features

        
if __name__ == "__main__":

    ### normalize continuous columns ###
    # read the continuous column_names
    with open("./intermediate_files/continuous_column_names.pkl", "rb") as fin:
        continuous_column_names = pickle.load(fin)
    
    # read filled nacc_csv rows
    with open("./intermediate_files/train_csv_filled.csv", "rt") as fin:
        train_csv = pd.read_csv(fin, low_memory=False)
    with open("./intermediate_files/valid_csv_filled.csv", "rt") as fin:
        valid_csv = pd.read_csv(fin, low_memory=False)
    with open("./intermediate_files/test_csv_filled.csv", "rt") as fin:
        test_csv = pd.read_csv(fin, low_memory=False)  

    # normalize the continuous columns for three sets
    #nacc_csf_concat = normalize_continuous_data(continuous_column_names, train_csv, valid_csv, test_csv)
    #train_csv_norm, valid_csv_norm, test_csv_norm = normalize_continuous_data(continuous_column_names, train_csv, valid_csv, test_csv)
    
    ### feature engineering ###
    # read in the column_names
    with open("./intermediate_files/all_column_names.pkl", "rb") as fin:
        column_names = pickle.load(fin)

    # read in continuous names
    with open("./intermediate_files/continuous_column_names.pkl", "rb") as fin:
        continuous_names = pickle.load(fin)     
    
    # read in categorical_values
    with open("./intermediate_files/categorical_values.pkl", "rb") as fin:
        categorical_values = pickle.load(fin) 

    # make features for each patient
    #nacc_csf_features = make_features_for_each_patient(column_names, continuous_names, categorical_values, nacc_csf_concat)
    #with open("./intermediate_files/nacc_csf_features_each_patient.pkl", "wb") as fout:
    #    pickle.dump(nacc_csf_features, fout)

    # read in labels of three sets
    with open("./processed_data/labels/train_labels.csv", "rt") as fin:
        train_labels = pd.read_csv(fin, low_memory=False)
    with open("./processed_data/labels/valid_labels.csv", "rt") as fin:
        valid_labels = pd.read_csv(fin, low_memory=False)
    with open("./processed_data/labels/test_labels.csv", "rt") as fin:
        test_labels = pd.read_csv(fin, low_memory=False)        
 
    # change the format and split the features into three sets
    #train_features, valid_features, test_features = change_feature_format(nacc_csf_features, train_labels, valid_labels, test_labels)    
    
    train_features_list, valid_features_list, test_features_list = feature_engineering(column_names, continuous_names, categorical_values, train_csv, valid_csv, test_csv, train_labels, valid_labels, test_labels)
    
    # write it into files
    with open("./intermediate_files/train_features_aggregated.pkl", "wb") as fout:
        pickle.dump(train_features_list, fout)
    with open("./intermediate_files/valid_features_aggregated.pkl", "wb") as fout:
        pickle.dump(valid_features_list, fout)
    with open("./intermediate_files/test_features_aggregated.pkl", "wb") as fout:
        pickle.dump(test_features_list, fout)        
