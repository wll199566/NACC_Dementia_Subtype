"""
This script is to aggregate all step 2 and step 3 to get the features from the 
unpreprocessed nacc_csf csv files.
"""

import numpy as np
import pandas as pd
import pickle

from step_2_process_nacc_csf_rows import drop_columns, combine_nacc_csf, fill_missing_data
from step_3_feature_engineering import feature_engineering


#mild_stage_definition = "first_visit_features"

### process nacc and csf rows  ###
data_path = "./data_augmentation/all_stages/csv/features/"

# read all train, valid and test
#with open(data_path + "train_" + mild_stage_definition + ".csv", "rt") as fin:
#    train_csv = pd.read_csv(fin, low_memory=False)
#with open(data_path + "valid_" + mild_stage_definition + ".csv", "rt") as fin:
#    valid_csv = pd.read_csv(fin, low_memory=False)
#with open(data_path + "test_" + mild_stage_definition + ".csv", "rt") as fin:
#    test_csv = pd.read_csv(fin, low_memory=False)

with open(data_path + "train_features.csv", "rt") as fin:
    train_csv = pd.read_csv(fin, low_memory=False)
with open(data_path + "valid_features.csv", "rt") as fin:
    valid_csv = pd.read_csv(fin, low_memory=False)
with open(data_path + "test_features.csv", "rt") as fin:
    test_csv = pd.read_csv(fin, low_memory=False)

# concatenate them together
nacc_csv = pd.concat([train_csv, valid_csv, test_csv])

# read in csf file
with open("../data/data_with_date/csf_csv_with_dates.csv", "rt") as fin:
    csf_csv = pd.read_csv(fin, low_memory=False)

# drop the columns for nacc and csf csv files 
nacc_csv_dropped = drop_columns(nacc_csv, "nacc")
csf_csv_dropped = drop_columns(csf_csv, "csf")

# combine NACC and CSF together
nacc_csf_csv_combined = combine_nacc_csf(nacc_csv_dropped, csf_csv_dropped)

# fill the missing data 
# load in continuous column names
with open("./processed_data/processed_pkl/column_names/continuous_column_names.pkl", "rb") as fin:
    continuous_names = pickle.load(fin)

# fill the missing data for all three sets
train_csv_filled, valid_csv_filled, test_csv_filled = fill_missing_data( nacc_csf_csv_combined, 
                                                                         continuous_names, 
                                                                         list(pd.unique(train_csv["NACCID"])), 
                                                                         list(pd.unique(valid_csv["NACCID"])), 
                                                                         list(pd.unique(test_csv["NACCID"])) )

### do the feature engineering ###

# read in the column_names
with open("./processed_data/processed_pkl/column_names/all_column_names.pkl", "rb") as fin:
    column_names = pickle.load(fin)

# read in categorical_values
with open("./processed_data/processed_pkl/column_names/categorical_values.pkl", "rb") as fin:
    categorical_values = pickle.load(fin) 

# read in labels of three sets
with open("./data_augmentation/all_stages/csv/labels/train_labels.csv", "rt") as fin:
    train_labels = pd.read_csv(fin, low_memory=False)
with open("./data_augmentation/all_stages/csv/labels/valid_labels.csv", "rt") as fin:
    valid_labels = pd.read_csv(fin, low_memory=False)
with open("./data_augmentation/all_stages/csv/labels/test_labels.csv", "rt") as fin:
    test_labels = pd.read_csv(fin, low_memory=False)        
 
# do the feature engineering and get the list for three sets
train_features_list, valid_features_list, test_features_list = feature_engineering(column_names, 
                                                                                   continuous_names, 
                                                                                   categorical_values, 
                                                                                   train_csv_filled, 
                                                                                   valid_csv_filled, 
                                                                                   test_csv_filled, 
                                                                                   train_labels, 
                                                                                   valid_labels, 
                                                                                   test_labels)

# write it into files
with open("./data_augmentation/all_stages/pkl/train_features.pkl", "wb") as fout:
    pickle.dump(train_features_list, fout)
with open("./data_augmentation/all_stages/pkl/valid_features.pkl", "wb") as fout:
    pickle.dump(valid_features_list, fout)
with open("./data_augmentation/all_stages/pkl/test_features.pkl", "wb") as fout:
    pickle.dump(test_features_list, fout)        
