### Here is one of the baselines for the prediction task ###

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    """
    Linear Regression Model implemented by Pytorch
    """
    def __init__(self, feature_dim, output_dim):
        """
        Args:
        - feature_dim: the dimension of the input feature
        - output_dim: the dimension of the output size, which is the number of classes
        """
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# test
if __name__ == "__main__":
    # read one of the feature files
    import pickle
    
    feature_filename = "./preprocess_data/NACC_prediction/window_width=1/first_definition/valid_features_first_def.pkl"
    with open(feature_filename, "rb") as fin:
        features = pickle.load(fin)

    # get one of the features
    sample_feature = torch.from_numpy(features[0]["feature"]).float()

    # make a model
    model = LinearRegression(sample_feature.shape[2], 3)

    # feed the sample feature into the model
    output = model(sample_feature)

    print(output.shape)

    # print the number of model parameter
    for parameter in model.parameters():
        print(parameter.shape)