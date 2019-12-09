# import necessary modules
import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data.dataset import Dataset

##################### Customized Dataset for NACC dataset ################################################
class NACCDataset(Dataset):
    def __init__(self, test_dictionary):
        """
        Args:
            - test_dictionary: python dictionary containing all the test samples
        """
        
        self.nacc_csf_features = test_dictionary
        
    def __getitem__(self, index):
        feature = torch.from_numpy(self.nacc_csf_features[index][0]).float()
        #print(feature)
        label = self.nacc_csf_features[index][1]
        return feature, label
    
    def __len__(self):
        return len(self.nacc_csf_features)

if __name__ == "__main__":
    
    with open("../../data_augmentation_models/data/all_early_stages/pkl/test_features.pkl", "rb") as fin:
        test_features = pickle.load(fin)
    
    data_loader = torch.utils.data.DataLoader(NACCDataset(test_features), batch_size=16, shuffle=False)
         
    for idx, (features, labels) in enumerate(data_loader):
        if idx == 0:
            break
    
    print(features.shape)
    print(labels.shape)
