"""
This script is to add the time information for nacc and csf csv files.
"""
import pandas as pd
import numpy as np
import math

def get_date(mon, day, year):
    """
    Convert number in .csv files to time of format yyyy-mm-dd
    Args:
        - mon: month
        - day: day
        - year: year
    Returns:
        - date: yyyy-mm-dd    
    """
    return str(int(year)).zfill(4) + '-' + str(int(mon)).zfill(2) + '-' + str(int(day)).zfill(2)


def get_dates(mon_col, day_col, year_col):
    """
    Convert the date columns in pandas to a python list.
    Args:
        - mon_col: month column in pandas dataframe
        - day_col: day column in pandas dataframe
        - year_col: year column in pandas dataframe
    Returns:
        - date_list: the list containing all the converted dates
    """    
    # convert the pandas dataframe column into a python list
    months = list(mon_col)
    days = list(day_col)
    years = list(year_col)

    # add the dates into date_list
    date_list = []
    for mon, day, year in zip(months, days, years):
        if math.isnan(mon):
            # if there is nan in pandas dataframe, then append "NAN"
            date_list.append("NAN")
        else:
            date_list.append(get_date(mon, day, year))   
    
    return date_list

def add_time_info(csv_filename):
    """
    Args:
        - csv_filename: nacc_csv or csf_csv filename
    Returns:
        - df_with_dates: dataframe with dates 
                         and removed the original date columns
    """
    # read in the csv file
    with open(csv_filename, "rt") as fin:
        df_csv = pd.read_csv(fin, low_memory=False)

    # tell if it is nacc_csv or csf_csv
    if "VISITMO" in list(df_csv.columns):
        # nacc_csv
        removed_column_list = [ "VISITMO", "VISITDAY", "VISITYR" ]
    elif "CSFLPMO" in list(df_csv.columns):
        # csf_csv
        removed_column_list = [ "CSFLPMO", "CSFLPDY", "CSFLPYR",
                                "CSFABMO", "CSFABDY", "CSFABYR",
                                "CSFPTMO", "CSFPTDY", "CSFPTYR",
                                "CSFTTMO", "CSFTTDY", "CSFTTYR" ]      
    else:
        raise ValueError("Unpredefined file: {}!".format(csv_filename))

    # get the date column and add into the dataframe
    df_csv["DATE"] = get_dates(df_csv[removed_column_list[0]], 
                               df_csv[removed_column_list[1]],
                               df_csv[removed_column_list[2]])

    # remove the original time columns 
    df_csv = df_csv.drop(columns=removed_column_list)
    
    # reorder the column name list 
    column_names = list(df_csv.columns)
    column_names.remove("DATE")
    column_names.insert(1, "DATE")
    df_csv = df_csv[column_names] 

    # reorder the rows according to NACCID and DATE
    df_csv.sort_values(by=["NACCID", "DATE"], axis=0, inplace=True)
    df_csv = df_csv.reset_index(drop=True)

    return df_csv


if __name__ == "__main__":

    nacc_csv_filename = "../data/razavian08202019.csv"
    csf_csv_filename = "../data/razavian08202019csf.csv"

    nacc_csv = add_time_info(nacc_csv_filename)
    csf_csv = add_time_info(csf_csv_filename)

    # write them into csv file
    nacc_csv.to_csv("../data/data_with_date/nacc_csv_with_dates.csv", index=False)
    csf_csv.to_csv("../data/data_with_date/csf_csv_with_dates.csv", index=False)