# import the necessary modules

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# model
#from Transformer.attend_diagnose import make_model
#from linear_regression import LinearRegression
#from Dataset import NACCDataset

# utility
#from utils.config_utils import load_config
#from utils.fs_utils import create_folder
#from utils.Timer import Timer
#from utils import torch_utils

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

import time
import pickle


def evaluation(model, output_size, loader, device):
    """
    Args:
        - model: deep learning model we want to evaluation
        - output_size: the number of classes
        - loader: validation data loader 
        - device: cpu or gpu
    Returns:
        - eval_stats: dictionary containing macro, micro F1 scores,
                      and F1 scores for all four classes respectively.
    """

    # initialize the statistics
    y_true_array = np.array([])  # numpy array to store all true labels
    y_pred_array = np.array([])  # numpy array to store all predicted labels  
    y_prob_array = np.empty((1, output_size))  # numpy array to store all predicted probability
    # to set the sigmoid function to compute the correctness
    #sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    # Note here we don't need to keep track of gradients
    with torch.no_grad():
        # the output of the dataloader is (features, labels)
        # the dimension of feature is [num_batch, max_time_steps, feature_dim]
        # labels is a nn.LongTensor type tensor of shape (batch_size,)
        for batch_idx, (features, labels) in enumerate(loader):
            # put features and labels into the device
            features = features.to(device)
            labels = labels.to(device)

            # compute the preds
            preds = model(features)

            # to compute the loss
            #loss = criterion(preds, labels.long())  # redunction = "mean" as default 
            # add L1 regularization
            #linear_params = [x for x in model.linear.parameters()][0].view(-1)
            #l1_regularization = torch.norm(linear_params, 1)
            #loss += alpha * l1_regularization
            
            # compute loss
            #running_loss += loss.item()

            # store true and predictive labels to numpy array for computing F1 score
            y_true = labels.cpu().numpy()
            y_pred = (torch.argmax(softmax(preds), dim=1)).cpu().numpy()
            y_prob = softmax(preds).cpu().numpy()
            y_true_array = np.concatenate((y_true_array, y_true), axis=0)
            y_pred_array = np.concatenate((y_pred_array, y_pred), axis=0)
            y_prob_array = np.concatenate((y_prob_array, y_prob), axis=0)

        
        # print the information into the screen   
        #print("eval_loss: {:.6f} | eval_macro_f1: {:.6f} | eval_micro_f1: {:.6f} | eval_f1: {}".format(running_loss / len(loader), f1_score(y_true_array, y_pred_array, average="macro"), f1_score(y_true_array, y_pred_array, average="micro"), f1_score(y_true_array, y_pred_array, average=None)))
        #print("eval_mtx:")
        #print(confusion_matrix(y_true_array, y_pred_array))

        # put statistics arrays to a python dictionary
        #eval_stat = {"eval_f1_macro": f1_score(y_true_array, y_pred_array, average="macro"), 
        #             "eval_f1_micro": f1_score(y_true_array, y_pred_array, average="micro"),
        #             "eval_f1s": f1_score(y_true_array, y_pred_array, average=None)}
        
        #return eval_stat

        # return the confusion matrix
        return confusion_matrix(y_true_array, y_pred_array)


if __name__ == "__main__":

    # initialize a model
    params = load_config('config.yaml')
    #model = make_model(h=params["head_number"], d_model=params["d_model"], d_ff=params["d_ff"], dropout=params["dropout"], max_len=params["max_len"], record_dim=params["record_dim"], d_ff_hidden=params["d_ff_hidden"], N=params["encoder_num"], de_factor=params["de_factor"])
    model = LinearRegression(params["feature_dim"], params["output_dim"])
    # load in the pre-trained model
    PATH_pretrained = "./models/session_2019-09-08[17_52_06]/model_24.pth"
    model.load_state_dict(torch.load(PATH_pretrained))

    # move the model into the corresponding device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # define the criterion
    # define the criterion
    # How to get weights:
    # e.g., weight of NORM = (# all training samples) / (# normal samples)
    training_loss_weights = [2.488, 43.346, 3.295, 3.683]
    weights = torch.FloatTensor(training_loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)  # reduction = "mean"
    
    # define data loader
    eval_dir = params["val_dir"]
    eval_loader = torch.utils.data.DataLoader(NACCDataset(eval_dir, "test"), batch_size=16, shuffle=False)

    # call the evaluation
    eval_f1_macro, eval_f1_micro, eval_stat = evaluation(model, params["output_dim"], criterion, eval_loader, device)
    
    ### store eval_true_array, eval_pred_array and eval_prob_array for error analysis ###
    
    # create folder to store them
    create_folder("statistics")
    stat_folder = "statistics" + '/' + PATH_pretrained.split('/')[2] + "/eval"
    create_folder(stat_folder)
    
    # store them
    with open(stat_folder+"/eval_statistics.pkl", "wb") as fout:
        pickle.dump(eval_stat, fout)

