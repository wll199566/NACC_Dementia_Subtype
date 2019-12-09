"""
This script includes all codes for using bootstrapping to get the 
confident interval for the best model
"""

# include necessary modules
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml

from models.linear_regression import LinearRegression

import numpy as np
from sklearn.utils import resample

from models.Dataset import NACCDataset
from models.evaluation import evaluation


### define a function to load in configuration  ###
def load_config(filename):
    params = {}

    with open(filename) as f:
        params = yaml.load(f)

    return params

### define a function to return the resampled test_features ###
def construct_bootstrap_test_samples(feature_list, sample_size):
    """
    Args:
        - feature_list: the list containing all test features
        - sample_size: the number of samples in each bootstrap sample
    """
    index_array = np.arange(len(feature_list))  # get the index array for test_features
    resampled_index_array = resample(index_array, n_samples=sample_size)
    resampled_index_list = resampled_index_array.tolist()
    test_features_resampled = [feature_list[i] for i in resampled_index_list]
    
    return test_features_resampled, resampled_index_list


######################### main #########################

### load in the test set ###
with open("../data_augmentation_models/data/all_early_stages/pkl/test_features.pkl", "rb") as fin:
    test_features = pickle.load(fin)


### define the model ###

# to get the arguments 
params = load_config('./models/config.yaml')       

# define the model
bootstrap_model = LinearRegression(params["feature_dim"], params["output_dim"])

# load in the pre-trained model
PATH_pretrained = "./models/model_6.pth"
print("read in", PATH_pretrained)

bootstrap_model.load_state_dict(torch.load(PATH_pretrained))

# define the device and move the model into the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the device is", device)

bootstrap_model.to(device)


### bootstrap to get confidence interval ###

# configure bootstrap
n_iterations = params["bootstrap_num_iterations"]
n_size = int(len(test_features)*params["bootstrap_test_sample_ratio"])

# run bootstrap
stats_list = []  # python list containing eval_stats for different bootstrap samples
sample_indices_list = []  # randomly bootstrap sample indices 

for iteration in range(n_iterations):
    
    # prepare bootstrap test set
    test_set, sample_indices = construct_bootstrap_test_samples(test_features, n_size)
    sample_indices_list.append(sample_indices)
    
    # get the DataLoader
    data_loader = torch.utils.data.DataLoader(NACCDataset(test_set), batch_size=params["batch_size"], shuffle=False)
    eval_stats = evaluation(bootstrap_model, params["output_dim"], data_loader, device)
    stats_list.append(eval_stats)


### store the results ###
with open("./bootstrap_results/f1_score_dict_n_size="+str(params["bootstrap_test_sample_ratio"])+".pickle", "wb") as fout:
    pickle.dump(stats_list, fout)

with open("./bootstrap_results/sample_indices_lists_n_size="+str(params["bootstrap_test_sample_ratio"])+".pikcle", "wb") as fout:
    pickle.dump(sample_indices_list, fout)    

