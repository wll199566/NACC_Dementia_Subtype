"""
This script is to get the wrong probability of each bin for both categorical and
continuous features. 

For continuous feature, we show the proportion of the values falling into each bin, 
while for categorical feature, we show the number of each category.
"""

import numpy as np
import pandas as pd
import pickle

def ctg_cont_wrong_prob(file_obj, whole_csv, col, label, ctg_bins_list, missing_values_list, is_ctg=True):
    """
    Compute the wrong probability p(wrong | bin) of each bin for this column (feature).

    Args:
        - file_obj: file object which defining which file we need to print out to
        - whole_csv: csv file containing all features of all samples we want to get the statistics 
        - col: the column whose value we want to get the statistics from
        - label: {0, 1, 2, or 3} which corresponds to {PURE AD, PURE LBD, MIXED AD+LBD, OTHERS}
        - ctg_bins_list: list of numbers indicating each category or edges of bins
        - missing_values_list: list of numbers indicating the missing values
        - is_ctg: bool value, whether it is categorical feature
    """

    # get the corresponding label name for each label
    if label == 0:
        label_name = "PURE AD"
    elif label == 1:
        label_name = "PURE LBD"
    elif label == 2:
        label_name = "MIX AD + LBD"
    else:
        label_name = "OTHERS"
    
    # print some important information
    print("Disease: {} | Feature: {}".format(label_name, col), file=file_obj)

    # to get all the samples for that label
    all_samples_for_this_label = whole_csv.loc[whole_csv["label"]==label]
    print("there are {} samples for {}".format(len(all_samples_for_this_label), label_name), file=file_obj)

    # remove the missing values
    samples_with_missing_values = all_samples_for_this_label.loc[all_samples_for_this_label[col].isin(missing_values_list)]
    print("there are {} missing values ({})".format(len(samples_with_missing_values), len(samples_with_missing_values) / len(all_samples_for_this_label)), file=file_obj)
    all_samples_for_this_label = all_samples_for_this_label.loc[~all_samples_for_this_label[col].isin(missing_values_list)]

    # for each group (bin), get the probability of wrong diagnosis
    ### categorical feature ###
    if is_ctg:
        for i in range(len(ctg_bins_list)):
            #print(ctg_bins_list[i], file=file_obj)
            # get samples  in this category
            samples_in_this_bin = all_samples_for_this_label.loc[all_samples_for_this_label[col]==ctg_bins_list[i]]
        
            ### obtain and print the probability p(wrong | this bin)
            total_num_samples_in_this_bin = len(samples_in_this_bin)
            #print("total num in this bin: {}".format(total_num_samples_in_this_bin), file=file_obj)
            # to get those whose diagnose are wrong
            sample_with_wrong_diag_in_this_bin = samples_in_this_bin.loc[samples_in_this_bin["clinician_diagnosis"]!=samples_in_this_bin["label"]]
            num_wrong_diag_in_this_bin = len(sample_with_wrong_diag_in_this_bin)
            if total_num_samples_in_this_bin == 0:
                print("{} has zero samples".format(ctg_bins_list[i]), file=file_obj)
            else:
                print("p(wrong | {}) = {} ({} / {})".format(ctg_bins_list[i], num_wrong_diag_in_this_bin / total_num_samples_in_this_bin, num_wrong_diag_in_this_bin, total_num_samples_in_this_bin), file=file_obj)
                if num_wrong_diag_in_this_bin != 0:
                    class_transition_wrong_samples([ctg_bins_list[i]], sample_with_wrong_diag_in_this_bin, "clinician_diagnosis", file_obj)
    ### continuous feature ###
    else:
        for i in range(len(ctg_bins_list)-1):
            #print(ctg_bins_list[i], file=file_obj)
            ### continuous feature ###
            # get the lower and upper bound for each 
            bin_lower_bound = ctg_bins_list[i]
            bin_upper_bound = ctg_bins_list[i+1]
            # for this bin, we get the total number of samples in this group
            # and the number of wrongly diagnosed sample to get the probability 
            # of p(wrong diagnosed | this_bin)
            if i == len(ctg_bins_list)-2:
                samples_in_this_bin = all_samples_for_this_label.loc[(all_samples_for_this_label[col]>=bin_lower_bound) & (all_samples_for_this_label[col]<=bin_upper_bound)]
            else:
                samples_in_this_bin = all_samples_for_this_label.loc[(all_samples_for_this_label[col]>=bin_lower_bound) & (all_samples_for_this_label[col]<bin_upper_bound)]
            
            ### obtain and print the probability p(wrong | this bin)
            total_num_samples_in_this_bin = len(samples_in_this_bin)
            #print("total num in this bin: {}".format(total_num_samples_in_this_bin), file=file_obj)
            # to get whose whose diagnose are wrong
            sample_with_wrong_diag_in_this_bin = samples_in_this_bin.loc[samples_in_this_bin["clinician_diagnosis"]!=samples_in_this_bin["label"]]
            num_wrong_diag_in_this_bin = len(sample_with_wrong_diag_in_this_bin)
            if total_num_samples_in_this_bin == 0:
                print("{} ~ {} has zero samples".format(bin_lower_bound, bin_upper_bound), file=file_obj)
            else:
                print("p(wrong | {}~{}) = {} ({} / {})".format(bin_lower_bound, bin_upper_bound, num_wrong_diag_in_this_bin / total_num_samples_in_this_bin, num_wrong_diag_in_this_bin, total_num_samples_in_this_bin), file=file_obj)
                if num_wrong_diag_in_this_bin != 0:
                    class_transition_wrong_samples([bin_lower_bound, bin_upper_bound], sample_with_wrong_diag_in_this_bin, "clinician_diagnosis", file_obj)


def class_transition_wrong_samples(bin_bound_list, wrong_samples_df_in_this_bin, col4pred, file_obj):
    """
    Print to which disease wrongly diagnosed samples are assigned to by doctors or the best model.
    Args:
        - bin_bound_list: the list storing the upper and lower bound for continuous feature, or the category for categorical feature.
        - wrong_samples_df_in_this_bin: dataframe storing all wrong samples for some disease
        - col4pred: string, for doctors, it is "clinician_diagnosis", for the best model, it is "pred_label"
    """
    num_pure_ad_wrong = len(wrong_samples_df_in_this_bin.loc[wrong_samples_df_in_this_bin[col4pred]==0])
    num_pure_lbd_wrong = len(wrong_samples_df_in_this_bin.loc[wrong_samples_df_in_this_bin[col4pred]==1])
    num_mix_wrong = len(wrong_samples_df_in_this_bin.loc[wrong_samples_df_in_this_bin[col4pred]==2])
    num_others_wrong = len(wrong_samples_df_in_this_bin.loc[wrong_samples_df_in_this_bin[col4pred]==3])
    
    ### categorical feature ###
    if len(bin_bound_list) == 1:
        print("{}: PURE AD: {}, PURE LBD: {}, MIX: {}, OTHERS: {}\n".format(bin_bound_list[0], num_pure_ad_wrong, num_pure_lbd_wrong, num_mix_wrong, num_others_wrong), file=file_obj)
    ### continuous feature ###
    elif len(bin_bound_list) == 2:
        print("{}~{}: PURE AD: {}, PURE LBD: {}, MIX: {}, OTHERS: {}\n".format(bin_bound_list[0], bin_bound_list[1], num_pure_ad_wrong, num_pure_lbd_wrong, num_mix_wrong, num_others_wrong), file=file_obj)
    else:
        raise ValueError("there are more than two bounds for this bin!!", fout=file_obj)
        