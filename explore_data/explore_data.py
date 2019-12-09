"""
This script is to get the statisitcs extracted from the data
"""
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix



def get_patient_visit_num(feature_df):
    """
    Get the total number of patients and visits in feature_df
    """
    
    return len(pd.unique(feature_df["NACCID"])), len(feature_df)


def get_average_age_init_visit(feature_df):
    """
    Get the average age of initial visits of all patients
    """
    
    # get the last record for each patient and the inital ages array
    init_visit_ages_arr = feature_df.groupby("NACCID").tail(1)["NACCAGEB"].values 
    
    return np.mean(init_visit_ages_arr), init_visit_ages_arr


def get_cdr_score_init_visit(feature_df):
    """
    Get the statistics for CDR score of initial visits of all patients
    """
    
    cdr_score_init_visit_arr = feature_df.groupby("NACCID").head(1)["CDRGLOB"].values

    return np.mean(cdr_score_init_visit_arr), cdr_score_init_visit_arr


def get_cdr_transitions(feature_df):
    """
    Get the transition matrix, in which the CDR score converts from the row value 
    to the column value
    """
    # get all last and next labels for each patient
    last_labels_list = []  # list containing all labels except for the last one for each patient
    next_labels_list = []  # list containing all labels except for the first one
    for naccid in pd.unique(feature_df["NACCID"]):
        patient_record = feature_df.loc[feature_df["NACCID"]==naccid]
        last_labels_list.extend(list(patient_record["CDRGLOB"][:-1]))
        next_labels_list.extend(list(patient_record["CDRGLOB"][1:]))
    
    # convert raw CDRGLOB score into continuous categories
    label_relationships = {0.0: 0, 0.5: 1, 1.0: 2, 2.0:3, 3.0: 4}
    last_labels_list = [label_relationships[item] for item in last_labels_list]
    next_labels_list = [label_relationships[item] for item in next_labels_list]
    
    return confusion_matrix(y_true=last_labels_list, y_pred=next_labels_list)


def get_dead_patients(feature_df):
    """
    Get all dead patients in the feature_df
    """
    # only dead persons have autopsy
    dead_patients_df = feature_df.loc[feature_df["NACCDIED"]==1]
    
    return dead_patients_df, len(pd.unique(dead_patients_df["NACCID"]))


def get_dementia_patients(dead_patients_df):
    """
    Get all dead patients who have dementia 
    """     
    dementia_patients_df = dead_patients_df[dead_patients_df["CDRGLOB"].isin([0.5, 1, 2, 3])]
    dementia_patients_list = list(pd.unique(dementia_patients_df["NACCID"]))
    # the reason we don't return dementia_patients_df is that it does not contain
    # CDRGLOB == 0 visits for those demented patients
    return dementia_patients_list, len(dementia_patients_list)


def get_early_stage_patients(dead_patients_df, dementia_patients_list):
    """
    Get all dead demented patients who have records on their early stages (CDRGLOB==0.5 or 1)
    """
    dementia_patients_df = dead_patients_df.loc[dead_patients_df["NACCID"].isin(dementia_patients_list)]
    early_stage_patients_df = dementia_patients_df.loc[dementia_patients_df["CDRGLOB"].isin([0.5, 1])]
    early_stage_patients_list = list(pd.unique(early_stage_patients_df["NACCID"]))
    # the reason we don't return early_stage_patients_df is that it does no contain
    # CDRGLOB == 0 visits for those early staged patients
    return early_stage_patients_list, len(early_stage_patients_list)



if __name__ == "__main__":
    # open the nacc.csv file
    with open("../data/data_with_date/nacc_csv_with_dates.csv", "rt") as fin:
        nacc_csv = pd.read_csv(fin, low_memory=False)
    
    # get the total number of patients and visits
    patient_num, visit_num = get_patient_visit_num(nacc_csv)
    print("There are {} patients in total, with {} total visits.\n".format(patient_num, visit_num))

    # get the average age of inital visit of all patients
    average_age_init_visit, _ = get_average_age_init_visit(nacc_csv)
    print("The average age of initial visits of all patients is", average_age_init_visit)
    print('\n')

    # get the average CDRGLOB score of the inital visit of all patients
    average_cdr_init_visit, _ = get_cdr_score_init_visit(nacc_csv)
    print("The average CDR score of initial visits of all patients is", average_cdr_init_visit)
    print('\n')

    # get the transition matrix of Dementia
    cdr_transition_matrix = get_cdr_transitions(nacc_csv)
    print("The CDR score transition matrix is\n", cdr_transition_matrix)
    print("\n")

    # get total number of dead patients
    dead_patients_df, dead_patients_num = get_dead_patients(nacc_csv)
    print("There are {} dead patients\n".format(dead_patients_num))
    
    # get total number of demented patients among them
    demented_patients_list, demented_patients_num = get_dementia_patients(dead_patients_df)
    print("There are {} demented patients out of {} dead patients\n".format(demented_patients_num, dead_patients_num))

    # get total number of early ages patients among them
    early_stage_patients_list, early_stage_patients_num = get_early_stage_patients(dead_patients_df, demented_patients_list)
    print("There are {} demented patients have early stage records\n".format(early_stage_patients_num))

    # get all patients who are (1) dead (2) demented (3) have early stages
    qualified_patients_df = dead_patients_df.loc[dead_patients_df["NACCID"].isin(early_stage_patients_list)]
    print("There are {} qualified patients with {} total visits\n".format(len(pd.unique(qualified_patients_df["NACCID"])), len(qualified_patients_df)))

    # write qualified_patients_df into csv file
    qualified_patients_df.to_csv("./intermediate_files/patients_dead_demented_early_stages.csv", index=False) 



