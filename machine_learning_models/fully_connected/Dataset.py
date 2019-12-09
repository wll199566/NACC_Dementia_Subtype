# import necessary modules
import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data.dataset import Dataset

##################### Customized Dataset for NACC dataset ################################################
class NACCDataset(Dataset):
    def __init__(self, nacc_csf_features_path, dataset="train"):
        """
        Args:
            - nacc_csf_features_path: path to the folder which contains all the pre-computed nacc_csf features
            - dataset: "train", "valid" or "test" dataset
        """
        # read in the nacc_csf_features according to dataset
        nacc_csf_features_filename = nacc_csf_features_path + "/" + dataset + "_features.pkl"
        with open(nacc_csf_features_filename, "rb") as fin:
            self.nacc_csf_features = pickle.load(fin)
        
    def __getitem__(self, index):
        feature = torch.from_numpy(self.nacc_csf_features[index][0]).float()
        #print(feature)
        label = self.nacc_csf_features[index][1]
        return feature, label
    
    def __len__(self):
        return len(self.nacc_csf_features)

if __name__ == "__main__":
    
    root_path = "../../features/all_early_stages"
    data_loader = torch.utils.data.DataLoader(NACCDataset(root_path, dataset="valid"), batch_size=16, shuffle=False)
         
    for idx, (features, labels) in enumerate(data_loader):
        if idx == 1:
            break
        print(features.shape)
        print(labels)
