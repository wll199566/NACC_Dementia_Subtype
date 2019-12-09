# Contents

This directory containes all codes to explore our data.

1. add_time_info: add `Date` column to the raw .csv files containing UDS data and CSF data. It is derived from Month, Day, Year from original columns. 
2. explore_data.py: get data statistics (total number of patients and visits, average age at the initial visits, average CDR score at the inital visits, CDR transition matrix of two consecutive visits) and process patients to select those patients who are (1) dead (2) demented (3) have early stages (CDRGLOB = 0.5 or 1).
3. explore_labels: define qualified patients and get Normal + Mild CDR windows. They are defined as visits from the initial visits to the first time patients show later stages (CDRGLOB = 2 or 3). When exploring the data, we used this window to define different tasks (first early stage visit, last early stage visit, all early visits) and finally used the task of first visits. 
