"""
This scripts contains all utility functions
"""

import numpy as np
import pandas as pd
import pickle

from step_2_process_nacc_csf_rows import preprocess_missing_data

def get_all_categorical_values(categorical_names, csv_file, file_type="nacc"):
    """
    Get all values for each categorical column for one-hot embedding.
    Args:
        - categorical_names: python list containing all categorical column names
        - csv_file: the csv file we need to get he categorical values from 
        - file_type: "nacc" or "csf"
    Returns:
        - categorical_values_dict: the dictionary {"categorical_name": [possible values]} 
    """
    # construct the categorical_values_dict
    categorical_values_dict = {}
    
    if file_type == "nacc":
        # combine missing values as -4 for all categorical columns
        csv_file = preprocess_missing_data(csv_file)
        categorical_names_list = list(set(categorical_names) - set(["CSFABMD", "CSFPTMD", "CSFTTMD"]))
    
    elif file_type == "csf":
        for col in ["CSFABMD", "CSFPTMD", "CSFTTMD"]:    
            csv_file.loc[pd.isnull(csv_file[col]), col] = -4
        categorical_names_list = ["CSFABMD", "CSFPTMD", "CSFTTMD"]    
    
    else:
        raise ValueError("{} is not predefined!!".format(file_type))
    
    # get unique values for each categorical_names
    for col in categorical_names_list:
        values = list(pd.unique(csv_file[col]))
        values.sort()
        # since when concatenate nacc and csf rows, there will add NaNs in CSFABMD column
        if col == "CSFABMD":
            values.insert(0, -4)
        categorical_values_dict[col] = np.array(values)

    return categorical_values_dict



if __name__ == "__main__":

    # read in nacc_csv_with_dates and csf_csv_with_dates
    with open("../data/data_with_date/nacc_csv_with_dates.csv", "rt") as fin:
        nacc_csv = pd.read_csv(fin, low_memory=False)
    with open("../data/data_with_date/csf_csv_with_dates.csv", "rt") as fin:
        csf_csv = pd.read_csv(fin, low_memory=False)
    
    # read in the categorical names
    with open("./intermediate_files/categorical_column_names.pkl", "rb") as fin:
        categorical_names = pickle.load(fin)

    # get categorical_values for nacc_csv
    nacc_categorical_values_dict = get_all_categorical_values(categorical_names, nacc_csv, "nacc")                
    csf_categorical_values_dict = get_all_categorical_values(categorical_names, csf_csv, "csf")

    # merge them together
    categorical_values_dict = {**nacc_categorical_values_dict, **csf_categorical_values_dict}
    
    #for key, value in categorical_values_dict.items():
    #    print(key, value)

    # write it into file
    with open("./intermediate_files/categorical_values.pkl", "wb") as fout:
        pickle.dump(categorical_values_dict, fout)
