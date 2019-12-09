"""
This script is to 1) define the bins and missing values for each key factors
2) show the probability of p(wrong | bin) for each category for categorical data or each bin for continuous data 

Dataframes we want to generate the distribution:
(1) whole_csv_with_diag: concatenation of training, validation and test dataset with clinician diagnosis.
(2) test_csv_with_diag: only the test set with clinician diagnosis, which is used for comparing with the model.
"""
import glob
import pandas as pd
from scipy import stats
from show_dist import ctg_cont_wrong_prob

### define the bins and missing values for each key factor ###
pure_alz_bins_missing_values = {
                                 "NACCAGE": (list(range(10, 130, 10)), [], False), 
                                 "SEX": ([1, 2], [], True), 
                                 "WAIS": (list(range(0, 110, 10)), [-4, 95, 96, 97, 98], False)
                               }

pure_lbd_bins_missing_values = {
                                "HALL": ([0, 1], [-4, 9], True), 
                                "NACCAANX": ([0, 1], [-4], True), 
                                "BOSTON": (list(range(0, 35, 5)), [-4, 95, 96, 97, 98], False),
                                "NACCMMSE": ([0, 10, 21, 25, 30], [-4, 88, 95, 96, 97, 98], False), 
                                "TRAILA": ([0, 29, 78, 150], [-4, 995, 996, 997, 998], False), 
                                "NITE": ([0, 1], [-4, 9], True), 
                                "SEX": ([1, 2], [], True), 
                                "INRELTO": (list(range(1, 8)), [-4], True)
                               }

mix_bins_missing_values = {
                            "CDRSUM": (list(range(0, 24, 6)), [], False) , 
                            "EDUC": ([0, 12, 16, 18, 20, 36], [99], False), 
                            "NACCAGE": (list(range(10, 130, 10)), [], False), 
                            "BOSTON": (list(range(0, 35, 5)), [-4, 95, 96, 97, 98], False), 
                            "NACCAGEB": (list(range(10, 130, 10)), [], False), 
                            "DIGIF": (list(range(0, 13)), [-4, 95, 96, 97, 98], False), 
                            "HYPERCHO": ([0, 1, 2], [-4, 9], True), 
                            "TRAILA": ([0, 29, 78, 150], [-4, 995, 996, 997, 998], False), 
                            "TRAILB": ([0, 75, 273, 300], [-4, 995, 996, 997, 998], False), 
                            "HALL": ([0, 1], [-4, 9], True), 
                            "HALLSEV": ([1,2,3,8], [-4, 9], True)
                          }

### read in all three datasets with clinician diagnosis ###
csv_folder_with_diag = "../feat_csv_with_diagnosis/"

with open(csv_folder_with_diag + "train_csv_with_diag.csv", "rt") as fin:
    train_csv_with_diag = pd.read_csv(fin, low_memory=False)

with open(csv_folder_with_diag + "valid_csv_with_diag.csv", "rt") as fin:
    valid_csv_with_diag = pd.read_csv(fin, low_memory=False)

with open(csv_folder_with_diag + "test_csv_with_diag.csv", "rt") as fin:
    test_csv_with_diag = pd.read_csv(fin, low_memory=False)

### concatenate all three dataframes into whole_csv ###
whole_csv = pd.concat([train_csv_with_diag, valid_csv_with_diag, test_csv_with_diag])        

### configure for printing distribution ###
label_list = [0, 1, 2]  # stands for PURE AD, PURE LBD, MIX AD+LBD
print_dist_dataframe_list = [whole_csv, test_csv_with_diag]
output_path = "../clinician_wrong_prob/"

### for each dataframe, we need to print each key factor distribution for each disease
for index, df in enumerate(print_dist_dataframe_list):
    
    # get the output_filename and output file object
    if index == 0:  # whole_csv
        output_filename = output_path + "whole_csv_with_diag_wrong_prob_with_transition.txt"
    else:
        output_filename = output_path + "test_csv_with_diag_wrong_prob_with_transition.txt"    

    # begin to print
    with open(output_filename, "wt") as fout:         
    
        # for each label, obtain the corresponding columns stated in corresponding dictionary
        for label in label_list:
            
            # we need to choose the config_dict
            if label == 0:
                config_dict = pure_alz_bins_missing_values
                separate_line = " PURE AD "
            elif label == 1:
                config_dict = pure_lbd_bins_missing_values
                separate_line = " PURE LBD "
            else:
                config_dict = mix_bins_missing_values
                separate_line = " MIX AD+LBD "
            
            print('*'*20 + separate_line +'*'*20, file=fout)   
    
            # We need to iterate each column (feature) we want to inspect 
            for col, config in config_dict.items():
                print('\n', file=fout)
                print('$'*10 + ' '+ col + ' ' +'$'*10 + '\n', file=fout)

                ctg_cont_wrong_prob(fout, 
                                    df, 
                                    col, 
                                    label, 
                                    config[0], 
                                    config[1], 
                                    is_ctg=config[2])    
            
                print('^' * 50, file=fout)
                print('\n', file=fout)                                        
    
    print("{} is Finished!!".format(output_filename))                                                                      
