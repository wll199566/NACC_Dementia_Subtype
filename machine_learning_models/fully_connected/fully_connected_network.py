### Here is the second baseline model for the prediction task ###

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):
    """
    Fully Connected Network including one hidden layer
    """
    def __init__(self, feature_dim, hidden_dim, output_dim, leaky_slop, dropout_rate):
        """
        Args:
        - feature_dim: the dimension of the input features
        - hidden_dim: the dimension of the hidden layer
        - output_dim: the dimension of the output, which is the number of classes
        - leaky_slop: the negative slop of the leaky ReLU layer
        - dropout_rate: the probability of dropout
        """
        super(FullyConnectedNet, self).__init__()
        self.hidden = nn.Linear(feature_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(negative_slope=leaky_slop)
        self.dropout = nn.Dropout(p=dropout_rate)

    # As suggested, we don't put dropout to the output layer
    def forward(self, x):
        return self.output(self.dropout(self.leakyrelu(self.hidden(x))))    

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
    model = FullyConnectedNet(sample_feature.shape[2], 1024,  3, 0.1, 0.2)

    # feed the sample feature into the model
    output = model(sample_feature)

    print(output.shape)
